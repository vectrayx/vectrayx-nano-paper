"""Curriculum pre-training driver for VectraYX-Nano v2.

Phase 1: 100% conversational  (LR 3e-4 from scratch)
Phase 2: 75% tech + 25% conv  (LR 1.5e-4, resumed from phase 1)
Phase 3: 70% tools + 20% tech + 10% conv  (LR 8e-5, resumed from phase 2)

Run example:
    python -m training_v2.train.pretrain \
        --config training_v2/configs/nano.json \
        --bins training_v2/data/bins \
        --out training_v2/checkpoints \
        --phase 1 --max-steps 8000 --batch-size 16 --grad-accum 8

Then:
    --phase 2 --resume training_v2/checkpoints/phase1/last.pt
    --phase 3 --resume training_v2/checkpoints/phase2/last.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.data.curriculum_dataset import (
    MixedCurriculumDataset, make_phase_mix, load_phase_summary,
)
from training_v2.model.transformer import VectraYXNano, ModelConfig
from training_v2.train.utils import (
    cosine_with_warmup, make_optimizer, save_checkpoint, load_checkpoint, log_jsonl,
)


PHASE_LR = {1: 3.0e-4, 2: 1.5e-4, 3: 8.0e-5}
PHASE_WARMUP_FRAC = {1: 0.05, 2: 0.02, 3: 0.02}


def build_dataloader(args, mix, block_size):
    phase_dirs = {
        "phase1_conv": Path(args.bins) / "phase1_conv",
        "phase2_tech": Path(args.bins) / "phase2_tech",
        "phase3_tools": Path(args.bins) / "phase3_tools",
    }
    ds = MixedCurriculumDataset(
        phase_dirs={k: v for k, v in phase_dirs.items() if mix.get(k, 0) > 0},
        weights=mix,
        block_size=block_size,
        dtype=np.uint16,
        seed=args.seed,
    )

    def collate(batch):
        xs = torch.stack([b[0] for b in batch], 0)
        ys = torch.stack([b[1] for b in batch], 0)
        return xs, ys

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def estimate_phase_tokens(phase_idx, mix, summary):
    total = 0.0
    for k, w in mix.items():
        n = summary.get(k, {}).get("n_tokens", 0)
        if w > 0 and n > 0:
            total += n
    return int(total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--bins", required=True, help="root of binary shard dirs")
    p.add_argument("--out", required=True, help="checkpoint output root")
    p.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--epochs", type=float, default=2.0,
                   help="estimate steps as epochs*phase_tokens/(batch*ga*block)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay-conv", type=float, default=None,
                   help="override replay ratio of conversational data in phase 2/3")
    p.add_argument("--replay-tech", type=float, default=None,
                   help="override replay ratio of technical data in phase 3")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig.from_json(args.config)
    model = VectraYXNano(cfg).to(args.device)
    n_params = model.num_params()
    print(f"[model] {n_params/1e6:.2f}M params · cfg={cfg}")

    mix = make_phase_mix(args.phase, replay_conv=args.replay_conv, replay_tech=args.replay_tech)
    summary = load_phase_summary(args.bins)
    phase_tokens = estimate_phase_tokens(args.phase, mix, summary)
    tokens_per_step = args.batch_size * args.grad_accum * cfg.max_seq_len
    if args.max_steps is None:
        args.max_steps = max(1000, int(args.epochs * phase_tokens / tokens_per_step))
    print(f"[phase {args.phase}] mix={mix}")
    print(f"[phase {args.phase}] phase_tokens={phase_tokens:,}  tokens/step={tokens_per_step:,}  steps={args.max_steps}")

    lr = args.lr if args.lr is not None else PHASE_LR[args.phase]
    warmup = max(50, int(PHASE_WARMUP_FRAC[args.phase] * args.max_steps))
    optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(args.resume, model, optimizer=None, map_location=args.device)
        print(f"[resume] loaded weights from {args.resume} (step={start_step})")
        start_step = 0  # fresh optimizer for new phase

    loader = build_dataloader(args, mix, cfg.max_seq_len)
    data_iter = iter(loader)

    out_dir = Path(args.out) / f"phase{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_amp = args.device == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[compile] skipped: {e}")

    model.train()
    t_start = time.time()
    tokens_seen = 0
    running_loss = 0.0
    running_n = 0

    for step in range(start_step, args.max_steps):
        cur_lr = cosine_with_warmup(step, warmup, args.max_steps, lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            xs, ys = batch[0], batch[1]
            xs = xs.to(args.device, non_blocking=True)
            ys = ys.to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                _, loss = model(xs, targets=ys)
                loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item() * args.grad_accum

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        tokens_seen += tokens_per_step
        running_loss += loss_accum / args.grad_accum
        running_n += 1

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            tps = tokens_seen / max(1.0, elapsed)
            avg_loss = running_loss / running_n
            print(f"[p{args.phase} step {step+1:>6}/{args.max_steps}] "
                  f"loss={avg_loss:.4f} lr={cur_lr:.2e} gnorm={gnorm:.2f} "
                  f"tok/s={tps:>7,.0f} elapsed={elapsed/60:.1f}min")
            log_jsonl(log_path, {
                "phase": args.phase, "step": step + 1, "loss": avg_loss,
                "lr": cur_lr, "gnorm": float(gnorm), "tok_per_s": tps,
                "tokens_seen": tokens_seen,
            })
            running_loss = 0.0
            running_n = 0

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = out_dir / "last.pt"
            save_checkpoint(ckpt_path, model, optimizer, {"step": step + 1}, step + 1,
                            extra={"phase": args.phase, "mix": mix, "lr": lr})
            print(f"[save] {ckpt_path}")

    final = out_dir / "last.pt"
    save_checkpoint(final, model, optimizer, {"step": args.max_steps}, args.max_steps,
                    extra={"phase": args.phase, "mix": mix, "lr": lr, "done": True})
    print(f"[done] phase {args.phase} → {final}")


if __name__ == "__main__":
    main()
