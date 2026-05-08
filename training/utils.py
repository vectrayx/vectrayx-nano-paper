"""Training utilities: optimizer setup, LR schedule, checkpointing."""

import json
import math
import os
from pathlib import Path

import torch


def cosine_with_warmup(step, warmup, total, max_lr, min_lr_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return min_lr_ratio * max_lr + 0.5 * (max_lr - min_lr_ratio * max_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True):
    """AdamW with weight decay only on 2D weights (no decay on biases / norms / embeddings).

    Per Loshchilov & Hutter; same convention as nanoGPT.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "tok_emb" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    extra = {}
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)


def save_checkpoint(path, model, optimizer, scheduler_state, step, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler_state,
        "step": step,
        "config": {k: getattr(model.cfg, k) for k in model.cfg.__dataclass_fields__},
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # Si el checkpoint tiene tie_embeddings=True, usar strict=False
    # (lm_head comparte pesos con tok_emb y no se guarda por separado)
    strict = not payload.get("tie_embeddings", False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if missing:
        print(f"[load_checkpoint] missing keys (expected with tie_embeddings): {missing[:3]}")
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("step", 0), payload.get("extra", {})


def count_tokens(loader_output_iter, n_steps, block_size, batch_size):
    """Approximate; effective tokens consumed per step."""
    return n_steps * block_size * batch_size


def log_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
