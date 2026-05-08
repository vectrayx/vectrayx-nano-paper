"""LoRA tool-use SFT para VectraYX Nano.

Aplica LoRA sobre las proyecciones de atención (wq, wk, wv, wo) del modelo
custom VectraYXNano. Congela todos los pesos base y solo entrena los adaptadores.

Ventaja sobre full fine-tune:
- Solo ~0.5% de parámetros entrenables (~200K vs 42M)
- Menos riesgo de catastrofic forgetting en B1/B2/B5
- SmolLM2-135M logra B4=0.16 con LoRA — probamos si Nano puede hacer lo mismo

Run example:
    python -m training_v2.train.finetune_lora_tools \
        --config training_v2/configs/nano.json \
        --tokenizer models/vectrayx_bpe.model \
        --resume checkpoints/nano_sft_v5.pt \
        --tool-corpus corpus/tool_sft_v2_simple.jsonl \
        --out checkpoints/nano_lora_tools \
        --lora-rank 16 --lora-alpha 32 \
        --batch-size 16 --grad-accum 4 --epochs 5 --lr 2e-4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.data.sft_dataset import SFTDataset
from training_v2.model.transformer import VectraYXNano, ModelConfig
from training_v2.train.utils import (
    cosine_with_warmup, log_jsonl,
)


# ---------------------------------------------------------------------------
# LoRA implementation
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Reemplaza un nn.Linear con LoRA: W' = W + (B @ A) * scale."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear          # pesos base — CONGELADOS
        self.rank = rank
        self.scale = alpha / rank

        in_f = linear.in_features
        out_f = linear.out_features

        # A: inicialización kaiming, B: ceros (LoRA paper §4)
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Congelar pesos base
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base = self.linear(x)
        # Asegurar que lora_A y lora_B estén en el mismo device que x
        lora = (x @ self.lora_A.to(x.device).T) @ self.lora_B.to(x.device).T
        return base + lora * self.scale


def inject_lora(model: nn.Module, rank: int, alpha: float,
                target_modules=("wq", "wk", "wv", "wo")) -> int:
    """Inyecta LoRA en todas las capas de atención del modelo.
    
    Retorna el número de parámetros entrenables.
    """
    replaced = 0
    for name, module in model.named_modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                original = getattr(module, attr_name)
                if isinstance(original, nn.Linear):
                    setattr(module, attr_name, LoRALinear(original, rank, alpha))
                    replaced += 1

    # Congelar todo excepto LoRA
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] Inyectado en {replaced} módulos | "
          f"Entrenables: {trainable/1e3:.1f}K / {total/1e6:.2f}M "
          f"({trainable/total*100:.2f}%)")
    return trainable


def save_lora_checkpoint(path: Path, model: nn.Module, optimizer, step: int,
                         extra: dict = None):
    """Guarda solo los pesos LoRA (no el modelo base)."""
    lora_state = {k: v for k, v in model.state_dict().items()
                  if "lora_A" in k or "lora_B" in k}
    torch.save({
        "lora_state_dict": lora_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "step": step,
        **(extra or {}),
    }, path)
    print(f"[save] LoRA checkpoint → {path} ({path.stat().st_size/1e6:.1f}MB)")


def load_lora_checkpoint(path: Path, model: nn.Module, optimizer=None,
                         map_location="cpu"):
    """Carga pesos LoRA en el modelo."""
    ckpt = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    lora_keys = [k for k in ckpt["lora_state_dict"]]
    print(f"[load] LoRA: {len(lora_keys)} keys loaded, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--resume", required=True, help="checkpoint base a fine-tunear")
    p.add_argument("--tool-corpus", required=True, help="tool-use JSONL corpus")
    p.add_argument("--out", required=True)
    # LoRA
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank r (default 16)")
    p.add_argument("--lora-alpha", type=float, default=32.0,
                   help="LoRA alpha (default 32, scale=alpha/rank=2)")
    p.add_argument("--lora-targets", nargs="+",
                   default=["wq", "wk", "wv", "wo"],
                   help="Módulos de atención a inyectar LoRA")
    # Training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="LR más alto que full FT (LoRA converge más rápido)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Cargar modelo base
    cfg = ModelConfig.from_json(args.config)
    model = VectraYXNano(cfg).to(args.device)
    total_params = model.num_params()
    print(f"[model] {total_params/1e6:.2f}M params (base)")

    # Cargar checkpoint base (full weights) usando load_checkpoint de utils
    from training_v2.train.utils import load_checkpoint as _load_ckpt
    _load_ckpt(args.resume, model, optimizer=None, map_location=args.device)
    print(f"[resume] {args.resume}")

    # 2. Inyectar LoRA
    trainable = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                            target_modules=args.lora_targets)
    # Mover parámetros LoRA al mismo device que el modelo
    model = model.to(args.device)

    # 3. Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0

    # 4. Dataset
    block_size = cfg.max_seq_len
    tool_corpus = Path(args.tool_corpus)
    if not tool_corpus.exists():
        raise FileNotFoundError(f"Tool corpus not found: {tool_corpus}")

    dataset = SFTDataset([tool_corpus], sp, block_size, pad_id=pad_id, seed=args.seed)
    print(f"[dataset] {len(dataset)} ejemplos de {tool_corpus.name}")

    # 5. Output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # 6. Optimizer — solo parámetros LoRA
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))

    # 7. AMP
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    use_amp = args.device == "cuda" and dtype != torch.float32

    # 8. Training loop
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        ms = torch.stack([b[2] for b in batch])
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
    warmup = max(20, int(args.warmup_frac * total_steps))

    print(f"\n[train] LoRA rank={args.lora_rank} alpha={args.lora_alpha} "
          f"scale={args.lora_alpha/args.lora_rank:.1f}")
    print(f"[train] epochs={args.epochs} steps/epoch≈{steps_per_epoch} "
          f"total={total_steps} warmup={warmup}")
    print(f"[train] lr={args.lr} batch={args.batch_size} accum={args.grad_accum} "
          f"effective_batch={args.batch_size * args.grad_accum}")

    model.train()
    t_start = time.time()
    step = 0
    running_loss = 0.0
    running_n = 0

    for ep in range(args.epochs):
        print(f"\n=== epoch {ep+1}/{args.epochs} (LoRA tool-SFT) ===")
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

            gnorm = torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            step += 1
            running_loss += loss_accum / args.grad_accum
            running_n += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                avg = running_loss / running_n
                print(f"[lora ep{ep+1} step {step:>4}/{total_steps}] "
                      f"loss={avg:.4f} lr={cur_lr:.2e} "
                      f"gnorm={gnorm:.2f} {elapsed/60:.1f}min")
                log_jsonl(log_path, {"epoch": ep+1, "step": step, "loss": avg,
                                     "lr": cur_lr, "gnorm": float(gnorm)})
                running_loss = 0.0
                running_n = 0

            if step % args.save_every == 0:
                save_lora_checkpoint(out_dir / "last_lora.pt", model, optimizer,
                                     step, {"epoch": ep+1})

        if args.max_steps and step >= args.max_steps:
            break

        save_lora_checkpoint(out_dir / f"epoch{ep+1}_lora.pt", model, optimizer,
                             step, {"epoch": ep+1})
        print(f"[save] epoch{ep+1}_lora.pt")

    # Guardar checkpoint final con pesos COMPLETOS (base + LoRA merged)
    # Estrategia: construir state_dict manualmente fusionando LoRA
    print("\n[merge] Mergeando LoRA en pesos base...")

    # Primero recolectar todos los módulos LoRA con sus rutas
    lora_modules = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            lora_modules[mod_name] = mod

    # Construir state_dict fusionado
    merged_state = {}
    for param_name, param in model.named_parameters():
        # Detectar si este parámetro pertenece a un LoRALinear
        is_lora_internal = False
        for lora_path in lora_modules:
            if param_name.startswith(lora_path + ".lora_"):
                is_lora_internal = True  # saltar lora_A y lora_B
                break
            if param_name == lora_path + ".linear.weight":
                # Fusionar con LoRA
                lora_mod = lora_modules[lora_path]
                fused = param.data + (lora_mod.lora_B.data @ lora_mod.lora_A.data) * lora_mod.scale
                # Guardar con nombre limpio (sin .linear)
                clean = lora_path + ".weight"
                merged_state[clean] = fused
                is_lora_internal = True
                break
            if param_name == lora_path + ".linear.bias":
                clean = lora_path + ".bias"
                merged_state[clean] = param.data
                is_lora_internal = True
                break
        if not is_lora_internal:
            merged_state[param_name] = param.data

    print(f"[merge] {len(merged_state)} keys en merged state_dict")

    # Guardar solo LoRA ANTES de modificar el modelo
    save_lora_checkpoint(out_dir / "final_lora_only.pt", model, optimizer,
                         step, {"done": True, "lora_rank": args.lora_rank,
                                "lora_alpha": args.lora_alpha})

    # Guardar merged (full model) para benchmark — usar clave "model" que espera load_checkpoint
    # strict=False en benchmark porque lm_head comparte pesos con tok_emb (tie_embeddings)
    torch.save({"model": merged_state, "step": step,
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "merged": True, "tie_embeddings": True},
               out_dir / "final.pt")
    print(f"[done] final.pt (merged) → {out_dir}")
    print(f"[done] final_lora_only.pt (adapter only) → {out_dir}")


if __name__ == "__main__":
    main()
