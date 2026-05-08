# VectraYX — Reproducibility Release

**Paper:** *VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model with Curriculum Learning and Native Tool Use*

This repository contains the code, datasets, and pre-computed results needed to reproduce the key experiments from the paper.

---

## Repository Structure

```
release/
├── Makefile                           ← make repro / make bench-nano / make lora-nano
├── requirements.txt                   ← exact package versions
├── configs/
│   ├── nano.json                      ← Nano 42M architecture (GQA 8q/2kv, d_model=512)
│   └── base.json                      ← Base 260M architecture (GQA 16q/4kv, d_model=1024)
├── training/
│   ├── transformer.py                 ← VectraYXNano model (GQA + QK-Norm + Z-loss + RoPE)
│   ├── pretrain.py                    ← 3-phase curriculum pre-training driver
│   ├── finetune_sft.py                ← SFT with assistant-only loss masking + mini-curriculum
│   ├── finetune_lora_tools.py         ← LoRA adapter injection + merge (key experiment)
│   ├── finetune_tools.py              ← Full fine-tune (baseline comparison)
│   ├── sft_dataset.py                 ← JSONL → tokenized dataset with loss masking
│   ├── utils.py                       ← AdamW, cosine LR, checkpoint save/load
│   ├── aws_lora_nano_tools_s3.py      ← SageMaker launcher: Nano LoRA (S3-only)
│   └── aws_lora_base_tools_s3.py      ← SageMaker launcher: Base LoRA (S3-only)
├── eval/
│   ├── benchmark.py                   ← VectraYX-Bench B1–B5 harness
│   ├── run_inference_lora.py          ← Inference with LoRA adapter loaded
│   ├── run_inference_base.py          ← Inference with base checkpoint
│   └── red_team_eval.py               ← Adversarial probe script
├── eval_data/
│   ├── b1_cveqa.jsonl                 ← 500 CVE Q&A prompts + expected keywords
│   ├── b2_classification.jsonl        ← 200 threat classification examples
│   ├── b3_commands.jsonl              ← 35 command-line completion prompts
│   ├── b4_tooluse.jsonl               ← 25 tool-selection prompts (v2: 50 prompts)
│   └── b5_conversational.jsonl        ← 10 conversational gate prompts
├── corpus/
│   ├── tool_sft_mini_v1.jsonl         ← 2,801 tool-use examples (ratio 1:21) ← KEY
│   ├── tool_sft_v3_bash.jsonl         ← 296 bash-focused examples
│   ├── tool_sft_v2_simple.jsonl       ← 115 simple bash examples
│   ├── b4_tooluse_v2.jsonl            ← B4 benchmark v2 (50 questions, 60% bash)
│   ├── build_mini_tool_corpus.py      ← Regenerate tool_sft_mini_v1 from scratch
│   ├── build_tool_sft_corpus.py       ← Full tool-use corpus generator
│   └── build_v3_and_bench.py          ← v3 corpus + benchmark builder
├── results/
│   ├── bench_nano_baseline_multiseed.json  ← Nano baseline N=4 seeds (paper Table 2)
│   ├── bench_nano_lora_multiseed.json      ← Nano LoRA N=4 seeds (paper Table 3)
│   └── bench_base_lora_s42.json            ← Base LoRA seed=42 (paper Table 3)
└── paper/
    └── main.pdf                       ← Paper PDF
```

---

## Key Finding: Tool-Use Corpus Density

The B4=0.000 floor in mixed SFT is a **corpus-density artifact**, not a capacity gate.

| Model | Corpus | Ratio | B4 |
|---|---|---|---|
| Nano 42M (mixed SFT, N=4 seeds) | 62K examples | 1:211 | **0.000** |
| **Nano 42M + LoRA (N=4 seeds)** | **2,801 examples** | **1:21** | **0.145 ± 0.046** |
| Base 260M (mixed SFT) | 62K examples | 1:211 | **0.000** |
| **Base 260M + LoRA** | **2,801 examples** | **1:21** | **0.580** |
| Pro 3B + LoRA-64 | 62K examples | ~1:10 | 0.600 |
| Pro 7B + QLoRA-32 | 62K examples | ~1:10 | 0.880 |

### Nano LoRA Multi-Seed Results (N=4, Table 3 in paper)

| Seed | B1 KW | B2 F1 | B3 TM | **B4** | B5 |
|------|-------|-------|-------|--------|-----|
| 42   | 0.008 | 0.200 | 0.029 | **0.220** | 0.500 |
| 7    | 0.017 | 0.200 | 0.029 | **0.140** | 0.600 |
| 13   | 0.006 | 0.200 | 0.000 | **0.120** | 0.600 |
| 23   | 0.014 | 0.205 | 0.029 | **0.100** | 0.600 |
| **Mean ± std** | **0.011 ± 0.004** | **0.201 ± 0.002** | **0.021 ± 0.012** | **0.145 ± 0.046** | **0.575 ± 0.043** |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download checkpoints

```bash
mkdir -p checkpoints
# From HuggingFace (links TBD — see paper for GCS paths)
# Nano 42M post-SFT (503 MB)
# wget https://huggingface.co/vectrayx/nano-sft-v5/resolve/main/nano_sft_v5.pt \
#      -O checkpoints/nano_sft_v5.pt
# Base 260M post-Phase3 (3.1 GB)
# wget https://huggingface.co/vectrayx/base-phase3/resolve/main/base_phase3_last.pt \
#      -O checkpoints/base_phase3_last.pt
# Tokenizer (474 KB)
# wget https://huggingface.co/vectrayx/tokenizer/resolve/main/vectrayx_bpe.model \
#      -O checkpoints/vectrayx_bpe.model
```

### 3. Run the full reproducibility suite

```bash
make repro
```

This runs:
1. `make bench-nano` — B1–B5 on Nano baseline (expected B4=0.000)
2. `make bench-base` — B1–B5 on Base baseline (expected B4=0.000)
3. `make lora-nano` — LoRA fine-tune Nano + eval (expected B4≈0.220 for seed=42)
4. `make lora-base` — LoRA fine-tune Base + eval (expected B4≈0.580 for seed=42)

### 4. Run individual experiments

```bash
# Benchmark only (no training)
make bench-nano
make bench-base

# LoRA fine-tune + benchmark
make lora-nano   # ~30 min on A10G
make lora-base   # ~45 min on A10G

# Regenerate corpus
make corpus
```

---

## Reproducing the Pre-Training Pipeline

The full from-scratch pre-training pipeline (Phases 1–3 + SFT) is described in `training_v2/README.md` in the main repository. The key entry points are:

```bash
# 1. Train tokenizer (BPE-16384, 50/50 conv/tech balance)
python -m training.tokenizer.train_spm_bpe \
    --config configs/nano.json \
    --corpus-root /path/to/corpus \
    --out-dir checkpoints/tokenizer

# 2. Tokenize corpus → binary shards
python -m training.data.prepare_corpus \
    --tokenizer checkpoints/tokenizer/vectrayx_bpe.model \
    --corpus-root /path/to/corpus \
    --out-root data/bins

# 3. Pre-train (3 phases with replay buffer)
python training/pretrain.py --config configs/nano.json \
    --bins data/bins --out checkpoints --phase 1 \
    --batch-size 16 --grad-accum 8 --epochs 2
python training/pretrain.py --config configs/nano.json \
    --bins data/bins --out checkpoints --phase 2 \
    --resume checkpoints/phase1/last.pt
python training/pretrain.py --config configs/nano.json \
    --bins data/bins --out checkpoints --phase 3 \
    --resume checkpoints/phase2/last.pt

# 4. SFT with mini-curriculum
python training/finetune_sft.py \
    --config configs/nano.json \
    --tokenizer checkpoints/tokenizer/vectrayx_bpe.model \
    --resume checkpoints/phase3/last.pt \
    --out checkpoints/sft_v5 \
    --batch-size 16 --grad-accum 4 --epochs 3 --lr 2e-5

# 5. Benchmark
python eval/benchmark.py \
    --config configs/nano.json \
    --tokenizer checkpoints/tokenizer/vectrayx_bpe.model \
    --checkpoint checkpoints/sft_v5/final.pt \
    --data-dir eval_data \
    --out results/bench_nano_baseline.json
```

**Estimated cost:** ~$12 USD on GCP L4 for 3 full runs (v2/v4/v6 ablations).

---

## SageMaker Experiments (LoRA)

The LoRA experiments were run on AWS SageMaker `ml.g5.xlarge` (NVIDIA A10G 24GB).

```bash
# Prerequisites: AWS CLI configured, S3 bucket with assets
# See training/aws_lora_nano_tools_s3.py for full setup

# Upload assets to S3
aws s3 cp checkpoints/nano_sft_v5.pt s3://YOUR_BUCKET/checkpoints/
aws s3 cp checkpoints/vectrayx_bpe.model s3://YOUR_BUCKET/tokenizers/
aws s3 cp corpus/tool_sft_mini_v1.jsonl s3://YOUR_BUCKET/training-data/

# Launch Nano LoRA (seed=42)
bash corpus/launch_nano_lora_mini_ondemand.sh

# Launch Base LoRA (seed=42)
bash corpus/launch_base_lora_mini_ondemand.sh
```

**Estimated cost per run:** ~$1.50 USD (ml.g5.xlarge on-demand, ~45 min).

---

## Model Checkpoints

| Checkpoint | Size | Description | Link |
|---|---|---|---|
| `nano_sft_v5.pt` | 503 MB | Nano 42M post-SFT (base for LoRA) | HuggingFace (TBD) |
| `nano_lora_mini_s42.pt` | ~5 MB | Nano LoRA adapter (seed=42) | HuggingFace (TBD) |
| `base_phase3_last.pt` | 3.1 GB | Base 260M post-Phase3 (base for LoRA) | HuggingFace (TBD) |
| `base_lora_mini_s42.pt` | ~20 MB | Base LoRA adapter (seed=42) | HuggingFace (TBD) |
| `vectrayx_bpe.model` | 474 KB | BPE-16384 tokenizer | HuggingFace (TBD) |

---

## Environment

Experiments were run with:

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.11.0 |
| sentencepiece | 0.2.1 |
| numpy | 2.4.2 |
| CUDA | 12.1 |
| boto3 | 1.42.93 |
| sagemaker | 3.10.0 |

Hardware:
- Pre-training: GCP `g2-standard-4` (NVIDIA L4 24GB), `us-west1-a`
- LoRA experiments: AWS SageMaker `ml.g5.xlarge` (NVIDIA A10G 24GB), `us-east-1`
- Multi-seed runs: AWS EC2 `g4dn.xlarge` (NVIDIA T4 16GB)

---

## Citation

```bibtex
@inproceedings{santillana2026vectrayx,
  title     = {VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model
               with Curriculum Learning and Native Tool Use},
  author    = {Santillana, Juan S.},
  booktitle = {Preprint},
  year      = {2026}
}
```

---

## License

| Component | License |
|---|---|
| Training code | MIT |
| Evaluation datasets (B1–B5) | CC-BY-4.0 |
| Model weights | Apache 2.0 |
| Paper | CC-BY-4.0 |
