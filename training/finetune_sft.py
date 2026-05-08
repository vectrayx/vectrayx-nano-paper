"""SFT fine-tuning with assistant-only loss masking and an internal mini-curriculum.

Mini-curriculum (within SFT):
  Epoch 1-2:  60% conversational (OASST1 ES + sft_conv) + 40% CVE Q&A
  Epoch 3:    add tool-use (50% conv + 25% CVE + 25% tool_use)

This avoids drowning the chat behavior in JSON tool-call patterns the way SFT v3 did.

Run example:
    python -m training_v2.train.finetune_sft \
        --config training_v2/configs/nano.json \
        --tokenizer training_v2/tokenizer/out/vectrayx_bpe.model \
        --resume training_v2/checkpoints/phase3/last.pt \
        --out training_v2/checkpoints/sft_v4 \
        --batch-size 16 --grad-accum 4 --epochs 3 --lr 2e-5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.data.sft_dataset import SFTDataset
from training_v2.model.transformer import VectraYXNano, ModelConfig
from training_v2.train.utils import (
    cosine_with_warmup, make_optimizer, save_checkpoint, load_checkpoint, log_jsonl,
)


SFT_FILES = {
    "conversational": [
        "corpus/sft_conversational.jsonl",
        "sft_v2_data/oasst1_es.jsonl",
    ],
    "cve_qa": [
        "corpus/sft_v2_dataset.jsonl",
    ],
    "tool_use": [
        "corpus/tooluse_dataset.jsonl",
    ],
}


def load_sft_corpus_config(path):
    global SFT_FILES
    cfg = json.loads(Path(path).read_text())
    SFT_FILES = {
        "conversational": cfg.get("sft_conversational", SFT_FILES["conversational"]),
        "cve_qa":         cfg.get("sft_cve_qa",         SFT_FILES["cve_qa"]),
        "tool_use":       cfg.get("sft_tool_use",        SFT_FILES["tool_use"]),
    }


def discover(paths, root):
    found = []
    for rel in paths:
        full = Path(root) / rel
        if full.exists():
            found.append(full)
        else:
            print(f"  [skip missing] {full}")
    return found


def build_dataset(args, sp, include_tools):
    block_size = ModelConfig.from_json(args.config).max_seq_len
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0

    conv = discover(SFT_FILES["conversational"], args.corpus_root)
    cve = discover(SFT_FILES["cve_qa"], args.corpus_root)
    tools = discover(SFT_FILES["tool_use"], args.corpus_root)

    parts = []
    if conv:
        parts.append(("conv", SFTDataset(conv, sp, block_size, pad_id=pad_id, seed=args.seed)))
    if cve:
        parts.append(("cve", SFTDataset(cve, sp, block_size, pad_id=pad_id, seed=args.seed + 1)))
    if include_tools and tools:
        parts.append(("tools", SFTDataset(tools, sp, block_size, pad_id=pad_id, seed=args.seed + 2)))
    return parts, pad_id


def make_loader(parts, weights, batch_size, num_workers):
    """Weighted sampling across the named parts."""
    sizes = [len(d) for _, d in parts]
    names = [n for n, _ in parts]
    datasets = [d for _, d in parts]
    big = ConcatDataset(datasets)

    offsets = np.cumsum([0] + sizes)
    weight_per_idx = np.zeros(offsets[-1], dtype=np.float64)
    for i, n in enumerate(names):
        w = weights.get(n, 1.0) / max(1, sizes[i])
        weight_per_idx[offsets[i]:offsets[i + 1]] = w
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weight_per_idx,
        num_samples=int(sum(sizes)),
        replacement=True,
    )

    def collate(batch):
        xs = torch.stack([b[0] for b in batch], 0)
        ys = torch.stack([b[1] for b in batch], 0)
        ms = torch.stack([b[2] for b in batch], 0)
        return xs, ys, ms

    return DataLoader(
        big, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--resume", required=True, help="pre-training checkpoint to fine-tune")
    p.add_argument("--out", required=True)
    p.add_argument("--corpus-root", default=".")
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    if args.corpus_config:
        load_sft_corpus_config(args.corpus_config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig.from_json(args.config)
    model = VectraYXNano(cfg).to(args.device)
    print(f"[model] {model.num_params()/1e6:.2f}M params")
    load_checkpoint(args.resume, model, optimizer=None, map_location=args.device)
    print(f"[resume] {args.resume}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    parts, pad_id = build_dataset(args, sp, include_tools=True)
    if not parts:
        raise RuntimeError("no SFT files found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_amp = args.device == "cuda" and dtype != torch.float32

    epoch_plans = [
        {"conv": 1.00, "cve": 0.00, "tools": 0.0},   # epoch 1: SOLO conversacional
        {"conv": 0.70, "cve": 0.30, "tools": 0.00},  # epoch 2: + CVE Q&A
        {"conv": 0.55, "cve": 0.30, "tools": 0.15},  # epoch 3: + tool use
    ]

    total_steps = 0
    for ep in range(args.epochs):
        weights = epoch_plans[min(ep, len(epoch_plans) - 1)]
        print(f"\n=== epoch {ep+1}/{args.epochs} | mix={weights} ===")
        loader = make_loader(parts, weights, args.batch_size, args.num_workers)
        steps_per_epoch = max(1, len(loader) // args.grad_accum)
        total_steps += steps_per_epoch
    warmup = max(50, int(args.warmup_frac * total_steps))
    print(f"[sft] total_steps≈{total_steps}  warmup={warmup}")

    model.train()
    t_start = time.time()
    step = 0
    running_loss = 0.0
    running_n = 0

    for ep in range(args.epochs):
        weights = epoch_plans[min(ep, len(epoch_plans) - 1)]
        loader = make_loader(parts, weights, args.batch_size, args.num_workers)
        data_iter = iter(loader)
        steps_per_epoch = max(1, len(loader) // args.grad_accum)

        for _ in range(steps_per_epoch):
            cur_lr = cosine_with_warmup(step, warmup, total_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _micro in range(args.grad_accum):
                try:
                    xs, ys, ms = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    xs, ys, ms = next(data_iter)
                xs = xs.to(args.device, non_blocking=True)
                ys = ys.to(args.device, non_blocking=True)
                ms = ms.to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                    _, loss = model(xs, targets=ys, loss_mask=ms)
                    loss = loss / args.grad_accum
                loss.backward()
                loss_accum += loss.item() * args.grad_accum

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            running_loss += loss_accum / args.grad_accum
            running_n += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                avg = running_loss / running_n
                print(f"[sft ep{ep+1} step {step:>5}/{total_steps}] loss={avg:.4f} "
                      f"lr={cur_lr:.2e} gnorm={gnorm:.2f} elapsed={elapsed/60:.1f}min")
                log_jsonl(log_path, {"epoch": ep + 1, "step": step, "loss": avg,
                                     "lr": cur_lr, "gnorm": float(gnorm)})
                running_loss = 0.0
                running_n = 0

            if step % args.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, optimizer,
                                {"step": step}, step,
                                extra={"epoch": ep + 1, "weights": weights})

        save_checkpoint(out_dir / f"epoch{ep+1}.pt", model, optimizer,
                        {"step": step}, step,
                        extra={"epoch": ep + 1, "weights": weights})
        print(f"[save] {out_dir}/epoch{ep+1}.pt")

    save_checkpoint(out_dir / "final.pt", model, optimizer, {"step": step}, step,
                    extra={"done": True})
    print(f"[done] SFT → {out_dir}/final.pt")


if __name__ == "__main__":
    main()
