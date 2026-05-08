#!/usr/bin/env python3
"""SageMaker entrypoint: LoRA tool-use SFT para VectraYX Nano - S3 ONLY.

Hyperparameters via env:
    CORPUS_NAME    = "v3_bash" (default)
    EPOCHS         = "5"
    LR             = "2e-4"
    LORA_RANK      = "16"
    LORA_ALPHA     = "32"
    SEED           = "42"
"""
import os, sys, json, subprocess, shutil
from pathlib import Path

S3_BUCKET = "s3://vectrayx-sagemaker-792811916323"
SM_OUTPUT = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
WD = Path("/opt/ml/code/work")
ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

# Nano config — checkpoint post-SFT mixto
NANO_CKPT = f"{S3_BUCKET}/checkpoints/nano_sft_v5.pt"
NANO_CFG  = "nano.json"
NANO_BATCH = 16
NANO_ACCUM = 4   # effective batch = 64


def die(m): print(f"\n[FATAL] {m}", flush=True); sys.exit(1)


def s3_download(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["aws", "s3", "cp", src, str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        die(f"s3 download failed: {src}\n{r.stderr}")
    print(f"[s3] ✓ {src} ({dst.stat().st_size/1e6:.1f}MB)", flush=True)


def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, env={**os.environ, **ENV},
                       cwd=str(cwd or WD))
    if r.returncode != 0:
        die(f"Failed: {cmd}")


def main():
    corpus_name = os.environ.get("CORPUS_NAME", "v3_bash")
    epochs      = int(os.environ.get("EPOCHS", "5"))
    lr          = float(os.environ.get("LR", "2e-4"))
    lora_rank   = int(os.environ.get("LORA_RANK", "16"))
    lora_alpha  = float(os.environ.get("LORA_ALPHA", "32"))
    seed        = int(os.environ.get("SEED", "42"))

    WD.mkdir(parents=True, exist_ok=True)
    SM_OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"[config] corpus={corpus_name} epochs={epochs} lr={lr} "
          f"lora_rank={lora_rank} lora_alpha={lora_alpha} seed={seed}", flush=True)

    # 1. Deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "sentencepiece", "tokenizers"], check=True)

    # 2. Código training_v2
    print("[code] Downloading training_v2 from S3...", flush=True)
    subprocess.run(["aws", "s3", "cp",
                    f"{S3_BUCKET}/code/training_v2.tar.gz",
                    "/tmp/tv2.tar.gz"], check=True)
    sh("tar xzf /tmp/tv2.tar.gz", cwd=WD)
    print(f"[code] ✓ training_v2 extracted", flush=True)

    # 3. Tokenizer
    s3_download(f"{S3_BUCKET}/tokenizers/vectrayx_bpe.model", WD/"tokenizer.model")

    # 4. Checkpoint base Nano (post-SFT mixto)
    s3_download(NANO_CKPT, WD/"resume.pt")

    # 5. Corpus tool-use
    s3_download(f"{S3_BUCKET}/training-data/tool_sft_{corpus_name}.jsonl",
                WD/"tool_sft.jsonl")

    # 6. Eval data — b4_tooluse_v2 tiene 50 preguntas con bash básico
    eval_dir = WD / "eval_data"
    for b in ["b1_cveqa", "b2_classification", "b3_commands",
              "b5_conversational"]:
        try:
            s3_download(f"{S3_BUCKET}/eval-data/{b}.jsonl",
                        eval_dir / f"{b}.jsonl")
        except Exception:
            print(f"[s3] skip (optional) {b}.jsonl", flush=True)
    # B4 v2 — benchmark ampliado con bash básico (60%) + MCP (40%)
    s3_download(f"{S3_BUCKET}/eval-data/b4_tooluse_v2.jsonl",
                eval_dir / "b4_tooluse.jsonl")  # mismo nombre para que benchmark.py lo encuentre

    # 7. LoRA fine-tune
    out_dir = WD / "checkpoints/lora_tool_sft"
    sh(f"{sys.executable} -m training_v2.train.finetune_lora_tools "
       f"--config {WD}/training_v2/configs/{NANO_CFG} "
       f"--tokenizer {WD}/tokenizer.model "
       f"--resume {WD}/resume.pt "
       f"--tool-corpus {WD}/tool_sft.jsonl "
       f"--out {out_dir} "
       f"--lora-rank {lora_rank} "
       f"--lora-alpha {lora_alpha} "
       f"--batch-size {NANO_BATCH} "
       f"--grad-accum {NANO_ACCUM} "
       f"--epochs {epochs} "
       f"--lr {lr} "
       f"--seed {seed}")

    # 8. Copiar artefactos al output
    shutil.copy(out_dir / "final.pt",          SM_OUTPUT / "final.pt")
    shutil.copy(out_dir / "final_lora_only.pt", SM_OUTPUT / "final_lora_only.pt")
    shutil.copy(WD / f"training_v2/configs/{NANO_CFG}", SM_OUTPUT / "model_config.json")

    # 9. Benchmark B1–B5 (usa final.pt merged)
    sh(f"{sys.executable} -m training_v2.eval.benchmark "
       f"--checkpoint {out_dir}/final.pt "
       f"--config {WD}/training_v2/configs/{NANO_CFG} "
       f"--tokenizer {WD}/tokenizer.model "
       f"--data-dir {eval_dir} "
       f"--out {SM_OUTPUT}/bench_lora_tools.json")

    # 10. Manifest
    manifest = {
        "model": "nano",
        "method": "lora",
        "corpus": corpus_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "resume_from": NANO_CKPT,
        "effective_batch": NANO_BATCH * NANO_ACCUM,
    }
    (SM_OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] LoRA tool-SFT Nano → {SM_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
