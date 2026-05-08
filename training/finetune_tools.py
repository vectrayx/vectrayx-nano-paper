"""Tool-use focused SFT for VectraYX Nano/Base.

This is a simplified version of finetune_sft.py that trains ONLY on tool-use examples.
The goal is to test the hypothesis that B4=0.000 is due to diluted tool-call gradients
in the mixed SFT corpus, not a capacity gate.

Run example:
    python -m training_v2.train.finetune_tools \
        --config training_v2/configs/nano.json \
        --tokenizer models/vectrayx_bpe.model \
        --resume checkpoints/nano_final.pt \
        --tool-corpus /tmp/tool_sft_v1.jsonl \
        --out checkpoints/tool_sft_nano \
        --batch-size 16 --grad-accum 4 --epochs 2 --lr 1e-5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.data.sft_dataset import SFTDataset
from training_v2.model.transformer import VectraYXNano, ModelConfig
from training_v2.train.utils import (
    cosine_with_warmup, make_optimizer, save_checkpoint, load_checkpoint, log_jsonl,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--resume", required=True, help="checkpoint to fine-tune from")
    p.add_argument("--tool-corpus", required=True, help="tool-use JSONL corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-steps", type=int, default=None, help="for testing")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    cfg = ModelConfig.from_json(args.config)
    model = VectraYXNano(cfg).to(args.device)
    print(f"[model] {model.num_params()/1e6:.2f}M params")
    load_checkpoint(args.resume, model, optimizer=None, map_location=args.device)
    print(f"[resume] {args.resume}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0

    # Build tool-only dataset
    block_size = cfg.max_seq_len
    tool_corpus = Path(args.tool_corpus)
    if not tool_corpus.exists():
        raise FileNotFoundError(f"Tool corpus not found: {tool_corpus}")
    
    dataset = SFTDataset([tool_corpus], sp, block_size, pad_id=pad_id, seed=args.seed)
    print(f"[dataset] {len(dataset)} tool-use examples from {tool_corpus}")

    # Setup output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # Optimizer
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # AMP setup
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_amp = args.device == "cuda" and dtype != torch.float32

    # Training loop
    def collate(batch):
        xs = torch.stack([b[0] for b in batch], 0)
        ys = torch.stack([b[1] for b in batch], 0)
        ms = torch.stack([b[2] for b in batch], 0)
        return xs, ys, ms

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(50, int(args.warmup_frac * total_steps))
    print(f"[train] epochs={args.epochs} steps_per_epoch≈{steps_per_epoch} total_steps={total_steps} warmup={warmup}")

    model.train()
    t_start = time.time()
    step = 0
    running_loss = 0.0
    running_n = 0

    for ep in range(args.epochs):
        print(f"\n=== epoch {ep+1}/{args.epochs} (tool-only) ===")
        data_iter = iter(loader)

        for _ in range(steps_per_epoch):
            if args.max_steps and step >= args.max_steps:
                break

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
                print(f"[tool-sft ep{ep+1} step {step:>5}/{total_steps}] loss={avg:.4f} "
                      f"lr={cur_lr:.2e} gnorm={gnorm:.2f} elapsed={elapsed/60:.1f}min")
                log_jsonl(log_path, {"epoch": ep + 1, "step": step, "loss": avg,
                                     "lr": cur_lr, "gnorm": float(gnorm)})
                running_loss = 0.0
                running_n = 0

            if step % args.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, optimizer,
                                {"step": step}, step,
                                extra={"epoch": ep + 1, "tool_only": True})

        if args.max_steps and step >= args.max_steps:
            break

        save_checkpoint(out_dir / f"epoch{ep+1}.pt", model, optimizer,
                        {"step": step}, step,
                        extra={"epoch": ep + 1, "tool_only": True})
        print(f"[save] {out_dir}/epoch{ep+1}.pt")

    save_checkpoint(out_dir / "final.pt", model, optimizer, {"step": step}, step,
                    extra={"done": True, "tool_only": True})
    print(f"[done] {out_dir}/final.pt")


if __name__ == "__main__":
    main()
