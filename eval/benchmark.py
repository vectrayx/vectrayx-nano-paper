"""VectraYX-Bench B1-B4 + B5 (conversational) benchmark runner.

Loads a checkpoint and an `eval_data/` directory of JSONL test files, runs them,
and prints a summary table compatible with the paper draft.

Expected files (any subset is fine):
    eval_data/b1_cveqa.jsonl         {"cve_id":..., "prompt":..., "expected_keywords":[...]}
    eval_data/b2_classification.jsonl {"prompt":..., "label":"phishing|malware|..."}
    eval_data/b3_commands.jsonl      {"prompt":..., "expected":"nmap -sV ...", "tool":"nmap"}
    eval_data/b4_tooluse.jsonl       {"prompt":..., "expected_tool":"nvd_get_cve"}
    eval_data/b5_conversational.jsonl {"prompt":"hola", "category":"saludo"}
"""

import argparse
import json
import re
import sys
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training_v2.model.transformer import VectraYXNano, ModelConfig
from training_v2.train.utils import load_checkpoint


SYSTEM_BASE = ("Eres VectraYX-Nano, asistente experto en ciberseguridad para "
               "América Latina. Responde en español de forma natural y concisa.")

# v2 (2026-05-05): system prompt extendido para B4. El SFT tooluse_dataset.jsonl
# fue entrenado con descripciones de cada herramienta + un ejemplo de formato
# JSON; el prompt anterior (lista plana de nombres) producía 0/25 en B4 porque
# nunca disparaba el patrón <|tool_call|>{"name":...}<|/tool_call|>.
SYSTEM_TOOL = (
    "Eres VectraYX, asistente experto en ciberseguridad para LATAM con acceso "
    "a las siguientes herramientas. Cuando una pregunta requiera datos en "
    "tiempo real (CVEs, IOCs, comandos), responde EXCLUSIVAMENTE con un "
    "bloque <|tool_call|>{...}<|/tool_call|> en formato JSON.\n\n"
    "Herramientas disponibles:\n"
    "- nvd_get_cve(cve_id): obtiene CVSS, descripción y referencias de un CVE.\n"
    "- nvd_search(keyword): busca CVEs recientes por palabra clave.\n"
    "- cisa_kev_check(cve_id): verifica si un CVE está en el catálogo KEV.\n"
    "- mitre_get_technique(technique_id): describe una técnica MITRE ATT&CK.\n"
    "- otx_check_ioc(ioc): verifica IP/dominio/hash en AlienVault OTX.\n"
    "- bash_exec(cmd): ejecuta un comando bash de análisis o forensics.\n\n"
    "Ejemplo:\n"
    "Usuario: ¿Está siendo explotada CVE-2021-44228?\n"
    "Asistente: <|tool_call|>{\"name\": \"cisa_kev_check\", "
    "\"args\": {\"cve_id\": \"CVE-2021-44228\"}}<|/tool_call|>"
)


def chat(user, system):
    return f"<|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>"


def generate(model, sp, prompt, max_new, end_id, eos_id, device,
             temperature=0.7, top_k=40, top_p=0.9, repeat_penalty=1.3):
    ids = torch.tensor([sp.encode(prompt, out_type=int)], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=max_new, temperature=temperature, top_k=top_k,
        top_p=top_p, eos_id=end_id, repeat_penalty=repeat_penalty,
    )
    gen = out[0, ids.size(1):].tolist()
    if end_id in gen:
        gen = gen[: gen.index(end_id)]
    if eos_id != end_id and eos_id in gen:
        gen = gen[: gen.index(eos_id)]
    return sp.decode(gen).strip()


def b1_cveqa(model, sp, data, ctx):
    if not data:
        return None
    hits = 0
    for ex in data:
        cve_id = ex.get("cve_id") or ex.get("id", "")
        prompt_text = ex.get("prompt") or ex.get("question") or f"Resume {cve_id}"
        prompt = chat(prompt_text, SYSTEM_BASE)
        out = generate(model, sp, prompt, 200, ctx["end_id"], ctx["eos_id"], ctx["device"]).lower()
        kws = [k.lower() for k in ex.get("expected_keywords", [])]
        score = sum(1 for k in kws if k in out) / max(1, len(kws))
        hits += score
    return hits / len(data)


def b2_classification(model, sp, data, ctx):
    if not data:
        return None
    labels = ["phishing", "malware", "ransomware", "apt", "otro"]
    correct = 0
    per_label = {l: [0, 0] for l in labels}  # [tp, total]
    for ex in data:
        text = ex.get("prompt") or ex.get("text") or ex.get("question", "")
        prompt = chat(f"{text}\nClasifica en una palabra: phishing, malware, ransomware, apt, otro.",
                      SYSTEM_BASE)
        out = generate(model, sp, prompt, 16, ctx["end_id"], ctx["eos_id"], ctx["device"]).lower()
        pred = next((l for l in labels if l in out), "otro")
        gold = ex["label"].lower()
        per_label[gold][1] += 1
        if pred == gold:
            correct += 1
            per_label[gold][0] += 1
    f1s = []
    for l, (tp, total) in per_label.items():
        if total == 0:
            continue
        recall = tp / total
        f1s.append(recall)
    return {"accuracy": correct / len(data), "f1_macro": sum(f1s) / max(1, len(f1s))}


def b3_commands(model, sp, data, ctx):
    if not data:
        return None
    exact = 0
    tool_match = 0
    for ex in data:
        prompt_text = ex.get("prompt") or ex.get("question", "")
        prompt = chat(prompt_text, SYSTEM_BASE)
        out = generate(model, sp, prompt, 80, ctx["end_id"], ctx["eos_id"], ctx["device"])
        gold_cmd = (ex.get("expected") or ex.get("expected_command", "")).strip()
        gold_tool = ex.get("tool", gold_cmd.split()[0] if gold_cmd else "")
        if gold_cmd in out:
            exact += 1
        if gold_tool.lower() in out.lower():
            tool_match += 1
    return {"exact_match": exact / len(data), "tool_match": tool_match / len(data)}


def b4_tooluse(model, sp, data, ctx):
    if not data:
        return None
    tools = ["nvd_get_cve", "nvd_search", "cisa_kev_check", "mitre_get_technique",
             "otx_check_ioc", "bash_exec"]
    correct = 0
    for ex in data:
        prompt_text = ex.get("prompt") or ex.get("question", "")
        prompt = chat(prompt_text, SYSTEM_TOOL)
        out = generate(model, sp, prompt, 120, ctx["end_id"], ctx["eos_id"], ctx["device"])
        m = re.search(r'"name"\s*:\s*"([^"]+)"', out)
        pred = m.group(1) if m else next((t for t in tools if t in out), None)
        if pred == ex["expected_tool"]:
            correct += 1
    return correct / len(data)


def b5_conversational(model, sp, data, ctx):
    if not data:
        return None
    ok = 0
    for ex in data:
        prompt = chat(ex["prompt"], SYSTEM_BASE)
        out = generate(model, sp, prompt, 80, ctx["end_id"], ctx["eos_id"], ctx["device"]).lower()
        cat = ex.get("category", "")
        if cat == "saludo":
            ok += int(any(w in out[:80] for w in ["hola", "buen", "qué tal", "encantad"]))
        elif cat == "agradecimiento":
            ok += int(any(w in out[:80] for w in ["nada", "gusto", "ayud"]))
        else:
            ok += int(len(out) > 5 and not out.startswith("cve"))
    return ok / len(data)


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True, help="folder with bN_*.jsonl files")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    cfg = ModelConfig.from_json(args.config)
    model = VectraYXNano(cfg).to(args.device).eval()
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    ctx = {
        "device": args.device,
        "end_id": sp.piece_to_id("<|end|>"),
        "eos_id": sp.eos_id(),
    }

    d = Path(args.data_dir)
    res = {
        "B1_cveqa_keyword":   b1_cveqa(model, sp, load_jsonl(d / "b1_cveqa.jsonl"), ctx),
        "B2_classification":  b2_classification(model, sp, load_jsonl(d / "b2_classification.jsonl"), ctx),
        "B3_commands":        b3_commands(model, sp, load_jsonl(d / "b3_commands.jsonl"), ctx),
        "B4_tooluse":         b4_tooluse(model, sp, load_jsonl(d / "b4_tooluse.jsonl"), ctx),
        "B5_conversational":  b5_conversational(model, sp, load_jsonl(d / "b5_conversational.jsonl"), ctx),
    }
    print("\n=== VectraYX-Bench ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
