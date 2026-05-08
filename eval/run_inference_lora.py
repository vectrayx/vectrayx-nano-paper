#!/usr/bin/env python3
"""
Inferencia con LoRA aplicado correctamente.

Estrategia:
1. Carga el modelo base (nano_sft_v5.pt) con pesos completos
2. Inyecta LoRA en wq/wk/wv/wo
3. Carga los pesos LoRA desde final_lora_only.pt
4. Corre inferencia

Esto evita el problema del merge con tie_embeddings=True.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.model.transformer import VectraYXNano, ModelConfig
from training_v2.train.utils import load_checkpoint


# ─── LoRA (misma implementación que finetune_lora_tools.py) ──────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A.to(x.device).T) @ self.lora_B.to(x.device).T
        return base + lora * self.scale


def inject_lora(model, rank, alpha, targets=("wq", "wk", "wv", "wo")):
    for name, module in model.named_modules():
        for attr in targets:
            if hasattr(module, attr):
                orig = getattr(module, attr)
                if isinstance(orig, nn.Linear):
                    setattr(module, attr, LoRALinear(orig, rank, alpha))
    # Congelar todo excepto LoRA
    for name, p in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] {trainable/1e3:.1f}K entrenables / {total/1e6:.2f}M total ({trainable/total*100:.2f}%)")


def load_lora_weights(path, model, device):
    """Carga solo los pesos lora_A y lora_B desde el checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    lora_state = ckpt.get("lora_state_dict", {})
    if not lora_state:
        # Intentar cargar directamente si el formato es diferente
        lora_state = {k: v for k, v in ckpt.items()
                      if "lora_A" in k or "lora_B" in k}
    if not lora_state:
        print(f"[warn] No se encontraron pesos LoRA en {path}")
        return
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    lora_loaded = len(lora_state)
    print(f"[lora] Cargados {lora_loaded} pesos LoRA | missing={len(missing)} unexpected={len(unexpected)}")


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres VectraYX, un asistente de ciberseguridad en español. "
    "Tienes 5 herramientas MCP disponibles:\n"
    "- nvd_get_cve(cve_id): obtener detalle de un CVE\n"
    "- nvd_search(query, limit): buscar CVEs por palabra clave\n"
    "- cisa_kev_check(cve_id): comprobar si un CVE está en el catálogo KEV\n"
    "- otx_check_ioc(ioc_type, value): reputación de IOC (ip, domain, hash)\n"
    "- bash_exec(cmd): ejecutar comando shell local\n"
    "Cuando la pregunta requiera datos externos o ejecutar algo, emite EXACTAMENTE:\n"
    '<|tool_call|>{"name":"<tool>","args":{...}}<|/tool_call|>\n'
    "Si la pregunta es conversacional o conceptual, responde en prosa SIN llamar herramientas."
)

TEST_PROMPTS = [
    ("qué hora es",                         "bash_exec"),
    ("dame la fecha de hoy",                "bash_exec"),
    ("quién soy",                           "bash_exec"),
    ("en qué directorio estoy",             "bash_exec"),
    ("cuánta memoria libre hay",            "bash_exec"),
    ("uso de disco",                        "bash_exec"),
    ("qué sistema operativo es",            "bash_exec"),
    ("cuál es mi IP",                       "bash_exec"),
    ("lista los archivos aquí",             "bash_exec"),
    ("qué puertos están escuchando",        "bash_exec"),
    ("dame detalles de CVE-2021-44228",     "nvd_get_cve"),
    ("busca CVEs de log4j",                 "nvd_search"),
    ("está CVE-2021-44228 en KEV",          "cisa_kev_check"),
    ("es maliciosa la IP 45.155.205.12",    "otx_check_ioc"),
    ("qué es un zero-day",                  None),
    ("hola, cómo estás",                    None),
]


def build_prompt(sp, question):
    text = (f"<|system|>{SYSTEM_PROMPT}<|end|>"
            f"<|user|>{question}<|end|>"
            f"<|assistant|>")
    return torch.tensor([sp.encode(text, out_type=int)], dtype=torch.long)


@torch.no_grad()
def generate(model, input_ids, sp, max_new=100, temperature=0.7,
             top_k=40, top_p=0.9, repeat_penalty=1.3):
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    end_id = sp.piece_to_id("<|end|>")
    generated = []

    for _ in range(max_new):
        logits, _ = model(ids)
        logits = logits[0, -1, :]

        # Repeat penalty
        if repeat_penalty != 1.0 and generated:
            for tok in set(generated[-50:]):
                logits[tok] /= repeat_penalty

        logits = logits / temperature

        # Top-k
        if top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < vals[-1]] = float('-inf')

        # Top-p
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum() + 1e-8
        next_tok = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

        generated.append(next_tok)
        ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

        if next_tok == end_id:
            break

    return sp.decode(generated)


def run_benchmark(model, sp, out_path):
    results = []
    correct = 0
    total_tool = 0

    print("\n" + "="*70)
    print("BENCHMARK — Nano LoRA v3")
    print("="*70)

    for question, expected_tool in TEST_PROMPTS:
        input_ids = build_prompt(sp, question)
        response = generate(model, input_ids, sp)

        # Detectar tool-call
        detected_tool = None
        detected_args = None
        if "<|tool_call|>" in response and "<|/tool_call|>" in response:
            try:
                s = response.index("<|tool_call|>") + len("<|tool_call|>")
                e = response.index("<|/tool_call|>")
                call = json.loads(response[s:e])
                detected_tool = call.get("name")
                detected_args = call.get("args", {})
            except Exception:
                pass

        if expected_tool is not None:
            total_tool += 1
            ok = detected_tool == expected_tool
            if ok:
                correct += 1
            status = "✅" if ok else "❌"
        else:
            ok = detected_tool is None
            status = "✅" if ok else "⚠️"

        print(f"\n[{status}] Q: {question}")
        print(f"     Expected: {expected_tool or 'ninguna'}")
        print(f"     Got:      {detected_tool or 'ninguna'}")
        if detected_args:
            print(f"     Args:     {detected_args}")
        # Mostrar respuesta limpia (solo ASCII+español)
        clean = ''.join(c if ord(c) < 0x4000 else '?' for c in response[:150])
        print(f"     Raw:      {clean.strip()}")

        results.append({
            "question": question,
            "expected_tool": expected_tool,
            "detected_tool": detected_tool,
            "detected_args": detected_args,
            "response": response,
            "correct": ok,
        })

    b4 = correct / total_tool if total_tool > 0 else 0
    print("\n" + "="*70)
    print(f"B4 SCORE: {correct}/{total_tool} = {b4:.3f}")
    print("="*70)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"b4_score": b4, "correct": correct,
                   "total_tool": total_tool, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
    return b4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True,
                   help="Checkpoint base del modelo (nano_sft_v5.pt)")
    p.add_argument("--lora-checkpoint", required=True,
                   help="Checkpoint LoRA (final_lora_only.pt)")
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="/tmp/nano_lora_bench_v2.json")
    args = p.parse_args()

    # 1. Cargar modelo base
    cfg = ModelConfig.from_json(args.config)
    model = VectraYXNano(cfg).to(args.device)
    print(f"[model] {model.num_params()/1e6:.2f}M params")

    load_checkpoint(args.base_checkpoint, model, map_location=args.device)
    print(f"[base] cargado: {args.base_checkpoint}")

    # 2. Inyectar LoRA
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(args.device)

    # 3. Cargar pesos LoRA
    load_lora_weights(args.lora_checkpoint, model, args.device)
    print(f"[lora] cargado: {args.lora_checkpoint}")

    model.eval()

    # 4. Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # 5. Benchmark
    run_benchmark(model, sp, args.out)


if __name__ == "__main__":
    main()
