#!/usr/bin/env python3
"""SageMaker entrypoint: tool-use mini-SFT focalizado (Nano o Base) - S3 ONLY.

Hyperparameters via env:
    MODEL          = "nano" | "base"
    CORPUS_NAME    = "v1" | "v2"
    EPOCHS         = "2"
    LR             = "1e-5"
    SEED           = "42"
"""
import os, sys, json, subprocess, shutil
from pathlib import Path

S3_BUCKET = "s3://vectrayx-sagemaker-792811916323"
SM_OUTPUT = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
WD = Path("/opt/ml/code/work")
ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

MODEL_CFG = {
    "nano": {
        "config":   "nano.json",
        "ckpt_src": f"{S3_BUCKET}/checkpoints/nano_sft_v5.pt",
        "batch":    16,
        "accum":    4,
    },
    "base": {
        "config":   "base.json",
        "ckpt_src": f"{S3_BUCKET}/checkpoints/vectrayx-base-20260506-1901/phase3_last.pt",
        "batch":    8,
        "accum":    8,
    },
}


def die(m): print(f"\n[FATAL] {m}", flush=True); sys.exit(1)


def s3_download(src, dst):
    """Download from S3 using AWS CLI."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["aws", "s3", "cp", src, str(dst)], 
                      capture_output=True, text=True)
    if r.returncode != 0:
        die(f"s3 download failed: {src}\n{r.stderr}")
    print(f"[s3] ✓ {src} ({dst.stat().st_size/1e6:.1f}MB)", flush=True)


def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, env={**os.environ, **ENV}, cwd=str(cwd or WD))
    if r.returncode != 0: die(f"Failed: {cmd}")


def main():
    model_name = os.environ.get("MODEL", "nano")
    corpus_name = os.environ.get("CORPUS_NAME", "v1")
    epochs = int(os.environ.get("EPOCHS", "2"))
    lr = float(os.environ.get("LR", "1e-5"))
    seed = int(os.environ.get("SEED", "42"))

    if model_name not in MODEL_CFG: die(f"Unknown MODEL={model_name}")
    cfg = MODEL_CFG[model_name]

    WD.mkdir(parents=True, exist_ok=True)
    SM_OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. Deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "sentencepiece", "tokenizers"], check=True)

    # 2. Download and extract training_v2 code
    print("[code] Downloading training_v2 from S3...", flush=True)
    subprocess.run(["aws", "s3", "cp", 
                   "s3://vectrayx-sagemaker-792811916323/code/training_v2.tar.gz",
                   "/tmp/tv2.tar.gz"], check=True)
    sh("tar xzf /tmp/tv2.tar.gz", cwd=WD)
    print(f"[code] ✓ training_v2 extracted to {WD}", flush=True)

    # 3. Tokenizer
    s3_download(f"{S3_BUCKET}/tokenizers/vectrayx_bpe.model", WD/"tokenizer.model")

    # 4. Checkpoint inicial
    s3_download(cfg["ckpt_src"], WD/"resume.pt")

    # 5. Tool SFT corpus
    s3_download(f"{S3_BUCKET}/training-data/tool_sft_{corpus_name}.jsonl",
                WD/"tool_sft.jsonl")

    # 6. Eval data
    eval_dir = WD / "eval_data"
    for b in ["b1_cveqa", "b2_classification", "b3_commands",
              "b4_tooluse", "b5_conversational"]:
        try:
            s3_download(f"{S3_BUCKET}/eval-data/{b}.jsonl",
                       eval_dir/f"{b}.jsonl")
        except:
            print(f"[s3] skip (optional) {b}.jsonl", flush=True)

    # 7. Mini-SFT focalizado
    out_dir = WD / "checkpoints/tool_sft"
    sh(f"{sys.executable} -m training_v2.train.finetune_tools "
       f"--config {WD}/training_v2/configs/{cfg['config']} "
       f"--tokenizer {WD}/tokenizer.model "
       f"--resume {WD}/resume.pt "
       f"--tool-corpus {WD}/tool_sft.jsonl "
       f"--out {out_dir} "
       f"--batch-size {cfg['batch']} --grad-accum {cfg['accum']} "
       f"--epochs {epochs} --lr {lr} --seed {seed}")

    # 8. Copiar checkpoint final
    shutil.copy(out_dir/"final.pt", SM_OUTPUT/"final.pt")
    shutil.copy(WD/f"training_v2/configs/{cfg['config']}",
                SM_OUTPUT/"model_config.json")

    # 9. Bench B1–B5
    sh(f"{sys.executable} -m training_v2.eval.benchmark "
       f"--checkpoint {out_dir}/final.pt "
       f"--config {WD}/training_v2/configs/{cfg['config']} "
       f"--tokenizer {WD}/tokenizer.model "
       f"--data-dir {eval_dir} "
       f"--out {SM_OUTPUT}/bench_tool_sft.json")

    # 10. Manifest
    manifest = {
        "model": model_name,
        "corpus": corpus_name,
        "epochs": epochs, "lr": lr, "seed": seed,
        "resume_from": cfg["ckpt_src"],
    }
    (SM_OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] tool-SFT {model_name}/{corpus_name}/seed={seed} → {SM_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
