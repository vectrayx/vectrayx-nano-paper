#!/usr/bin/env python3
"""
Inferencia interactiva del modelo Base 260M (o Nano 42M) con PyTorch.
Descarga el checkpoint desde S3, carga el modelo y permite hacer preguntas.

Uso:
    python run_inference_base.py --model base --checkpoint s3://...
    python run_inference_base.py --model nano  --checkpoint s3://...

Requiere:
    pip install torch sentencepiece boto3
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import sentencepiece as spm

# Añadir training_v2 al path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.model.transformer import VectraYXNano, ModelConfig

# ─── Prompts de prueba ────────────────────────────────────────────────────────

TEST_PROMPTS = [
    # Bash básico
    ("qué hora es",                         "bash_exec", "date"),
    ("dame la fecha de hoy",                "bash_exec", "date +%Y-%m-%d"),
    ("quién soy",                           "bash_exec", "whoami"),
    ("cuánta memoria libre hay",            "bash_exec", "free -h"),
    ("uso de disco",                        "bash_exec", "df -h"),
    ("qué sistema operativo es",            "bash_exec", "uname -a"),
    ("cuál es mi IP",                       "bash_exec", "hostname -I"),
    ("lista los archivos aquí",             "bash_exec", "ls -lh"),
    # Bash intermedio
    ("qué puertos están escuchando",        "bash_exec", "ss -tuln"),
    ("procesos que más CPU usan",           "bash_exec", "ps aux --sort=-%cpu"),
    # MCP
    ("dame detalles de CVE-2021-44228",     "nvd_get_cve", None),
    ("busca CVEs de log4j",                 "nvd_search", None),
    ("está CVE-2021-44228 en KEV",          "cisa_kev_check", None),
    ("es maliciosa la IP 45.155.205.12",    "otx_check_ioc", None),
    # Conversacional (no debe llamar tool)
    ("qué es un zero-day",                  None, None),
    ("hola, cómo estás",                    None, None),
]

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


def build_prompt(sp, question: str) -> torch.Tensor:
    text = (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>{question}<|end|>"
        f"<|assistant|>"
    )
    ids = sp.encode(text, out_type=int)
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def generate(model, input_ids, sp, max_new=120, temperature=0.7,
             top_k=40, top_p=0.9, repeat_penalty=1.3):
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    end_id = sp.piece_to_id("<|end|>")
    generated = []

    for _ in range(max_new):
        logits, _ = model(ids)
        logits = logits[0, -1, :]  # (vocab,)

        # Repeat penalty
        if repeat_penalty != 1.0 and generated:
            for tok in set(generated[-50:]):
                logits[tok] /= repeat_penalty

        logits = logits / temperature

        # Top-k
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[-1]] = float('-inf')

        # Top-p
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        next_tok = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

        generated.append(next_tok)
        ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

        if next_tok == end_id:
            break

    return sp.decode(generated)


def s3_download(s3_path: str, local_path: Path):
    print(f"[s3] Descargando {s3_path} ...", flush=True)
    r = subprocess.run(["aws", "s3", "cp", s3_path, str(local_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] {r.stderr}")
        sys.exit(1)
    print(f"[s3] ✓ {local_path} ({local_path.stat().st_size/1e6:.1f}MB)")


def load_model(checkpoint_path: Path, config_path: Path, device: str):
    cfg = ModelConfig.from_json(str(config_path))
    model = VectraYXNano(cfg).to(device)
    print(f"[model] {model.num_params()/1e6:.2f}M params")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")
    model.eval()
    return model


def run_benchmark(model, sp, results_path: Path):
    """Corre los prompts de prueba y guarda resultados."""
    results = []
    correct = 0
    total_tool = 0

    print("\n" + "="*70)
    print("BENCHMARK DE INFERENCIA")
    print("="*70)

    for question, expected_tool, expected_cmd in TEST_PROMPTS:
        input_ids = build_prompt(sp, question)
        response = generate(model, input_ids, sp)

        # Detectar tool-call
        detected_tool = None
        detected_args = None
        if "<|tool_call|>" in response:
            try:
                start = response.index("<|tool_call|>") + len("<|tool_call|>")
                end = response.index("<|/tool_call|>")
                call = json.loads(response[start:end])
                detected_tool = call.get("name")
                detected_args = call.get("args", {})
            except Exception:
                pass

        # Evaluar
        if expected_tool is not None:
            total_tool += 1
            ok = detected_tool == expected_tool
            if ok:
                correct += 1
            status = "✅" if ok else "❌"
        else:
            # Conversacional — no debe llamar tool
            ok = detected_tool is None
            status = "✅" if ok else "⚠️ (llamó tool innecesariamente)"

        print(f"\n[{status}] Q: {question}")
        print(f"     Expected: {expected_tool or 'ninguna'}")
        print(f"     Got tool: {detected_tool or 'ninguna'}")
        if detected_args:
            print(f"     Args: {detected_args}")
        print(f"     Response: {response[:120].strip()}")

        results.append({
            "question": question,
            "expected_tool": expected_tool,
            "expected_cmd": expected_cmd,
            "detected_tool": detected_tool,
            "detected_args": detected_args,
            "response": response,
            "correct": ok,
        })

    # Resumen
    b4_score = correct / total_tool if total_tool > 0 else 0
    print("\n" + "="*70)
    print(f"RESULTADO B4: {correct}/{total_tool} = {b4_score:.3f}")
    print("="*70)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "b4_score": b4_score,
            "correct": correct,
            "total_tool": total_tool,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {results_path}")
    return b4_score


def interactive_mode(model, sp):
    """Modo interactivo para hacer preguntas manualmente."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO — escribe 'exit' para salir")
    print("="*70)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        input_ids = build_prompt(sp, question)
        response = generate(model, input_ids, sp)
        print(f"\n{response}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["nano", "base"], default="base")
    p.add_argument("--checkpoint", help="Path local o s3:// al checkpoint")
    p.add_argument("--config", help="Path al JSON de config (opcional)")
    p.add_argument("--tokenizer", help="Path al tokenizer .model (opcional)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--benchmark", action="store_true", help="Correr benchmark automático")
    p.add_argument("--interactive", action="store_true", help="Modo interactivo")
    p.add_argument("--out", default="inference_results.json")
    args = p.parse_args()

    # Defaults por modelo
    model_defaults = {
        "nano": {
            "checkpoint": "s3://vectrayx-sagemaker-792811916323/checkpoints/nano_sft_v5.pt",
            "config": "training_v2/configs/nano.json",
        },
        "base": {
            "checkpoint": "s3://vectrayx-sagemaker-792811916323/checkpoints/vectrayx-base-20260506-1901/phase3_last.pt",
            "config": "training_v2/configs/base.json",
        },
    }

    checkpoint = args.checkpoint or model_defaults[args.model]["checkpoint"]
    config_path = Path(args.config or model_defaults[args.model]["config"])

    # Descargar checkpoint si es S3
    if checkpoint.startswith("s3://"):
        local_ckpt = Path(f"/tmp/vectrayx_{args.model}_ckpt.pt")
        if not local_ckpt.exists():
            s3_download(checkpoint, local_ckpt)
        else:
            print(f"[cache] Usando checkpoint en caché: {local_ckpt}")
        checkpoint = local_ckpt

    # Tokenizer
    tokenizer_path = Path(args.tokenizer or "/tmp/vectrayx_bpe.model")
    if not tokenizer_path.exists():
        s3_download(
            "s3://vectrayx-sagemaker-792811916323/tokenizers/vectrayx_bpe.model",
            tokenizer_path
        )

    # Cargar
    print(f"\n[init] Cargando modelo {args.model} en {args.device}...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    model = load_model(Path(checkpoint), config_path, args.device)

    # Ejecutar
    if args.benchmark or not args.interactive:
        run_benchmark(model, sp, Path(args.out))

    if args.interactive:
        interactive_mode(model, sp)


if __name__ == "__main__":
    main()
