#!/usr/bin/env python3
"""
Upload VectraYX assets to HuggingFace Hub.

Creates 3 repositories:
  1. vectrayx/vectrayx-nano          — model weights + tokenizer + configs
  2. vectrayx/vectrayx-bench         — eval datasets (B1-B5 JSONL)
  3. vectrayx/vectrayx-paper-code    — reproducibility code (mirrors release/)

Usage:
    # First login:
    python3 -c "from huggingface_hub import login; login()"

    # Then upload everything:
    python3 upload_to_hf.py --org vectrayx --all

    # Or upload specific parts:
    python3 upload_to_hf.py --org vectrayx --models
    python3 upload_to_hf.py --org vectrayx --datasets
    python3 upload_to_hf.py --org vectrayx --code
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
except ImportError:
    print("Install: pip install huggingface_hub")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

RELEASE_DIR = Path(__file__).parent

# What to upload to each repo
REPOS = {
    "models": {
        "repo_id": "{org}/vectrayx-nano",
        "repo_type": "model",
        "description": "VectraYX-Nano: 42M-parameter Spanish cybersecurity LLM with curriculum learning and native MCP tool use.",
        "files": [
            # (local_path, repo_path)
            ("checkpoints/nano_sft_v5.pt",          "nano_sft_v5.pt"),
            ("checkpoints/vectrayx_bpe.model",       "tokenizer/vectrayx_bpe.model"),
            ("configs/nano.json",                    "configs/nano.json"),
            ("configs/base.json",                    "configs/base.json"),
            # LoRA adapters (Nano, 4 seeds)
            ("checkpoints/lora/nano_s42_lora_only.pt", "lora/nano_lora_mini_s42.pt"),
            ("checkpoints/lora/nano_s7_lora_only.pt",  "lora/nano_lora_mini_s7.pt"),
            ("checkpoints/lora/nano_s13_lora_only.pt", "lora/nano_lora_mini_s13.pt"),
            ("checkpoints/lora/nano_s23_lora_only.pt", "lora/nano_lora_mini_s23.pt"),
            # Base 260M
            ("checkpoints/base_phase3_last.pt",      "base_phase3_last.pt"),
            ("checkpoints/lora/base_s42_lora_only.pt", "lora/base_lora_mini_s42.pt"),
        ],
        "readme": "model_card.md",
    },
    "datasets": {
        "repo_id": "{org}/vectrayx-bench",
        "repo_type": "dataset",
        "description": "VectraYX-Bench: B1-B5 evaluation suite for Spanish cybersecurity LLMs.",
        "folders": [
            ("eval_data/", "data/"),
            ("corpus/",    "tool_sft_corpus/"),
            ("results/",   "paper_results/"),
        ],
        "readme": "dataset_card.md",
    },
    "code": {
        "repo_id": "{org}/vectrayx-paper-code",
        "repo_type": "model",  # use model type for code repos on HF
        "description": "Reproducibility code for VectraYX paper.",
        "folders": [
            ("training/",  "training/"),
            ("eval/",      "eval/"),
            ("configs/",   "configs/"),
        ],
        "files": [
            ("Makefile",          "Makefile"),
            ("requirements.txt",  "requirements.txt"),
            ("README.md",         "README.md"),
            ("paper/main.pdf",    "paper/main.pdf"),
        ],
        "readme": "README.md",
    },
}


# ── Model Card ─────────────────────────────────────────────────────────────────

MODEL_CARD = """---
language:
- es
license: apache-2.0
tags:
- cybersecurity
- spanish
- tool-use
- mcp
- curriculum-learning
- from-scratch
datasets:
- vectrayx/vectrayx-bench
metrics:
- accuracy
- f1
---

# VectraYX-Nano

A 42M-parameter Spanish cybersecurity language model trained from scratch with
curriculum learning and native MCP tool use.

## Key Results (VectraYX-Bench)

| Model | Params | B1 KW | B2 F1 | B3 TM | B4 Tool | B5 |
|---|---|---|---|---|---|---|
| VectraYX-Nano v2 (N=4 seeds) | 42M | 0.228 ± 0.079 | 0.196 ± 0.005 | 0.029 ± 0.040 | 0.000 | 0.775 ± 0.050 |
| **Nano + LoRA mini (N=4 seeds)** | 42M | 0.011 ± 0.004 | 0.201 ± 0.002 | 0.021 ± 0.012 | **0.145 ± 0.046** | 0.575 ± 0.043 |
| VectraYX-Base 260M | 260M | 0.325 | 0.220 | 0.114 | 0.000 | 0.800 |
| **Base + LoRA mini** | 260M | 0.025 | 0.200 | 0.000 | **0.580** | 0.600 |
| VectraYX-Pro 3B | 3.2B | 0.341 | 0.695 | 0.686 | 0.600 | 0.800 |
| VectraYX-Pro 7B | 7B | 0.335 | 0.815 | 0.686 | 0.880 | 0.800 |

## Key Finding

The B4=0.000 floor in mixed SFT is a **corpus-density artifact**, not a capacity gate.
At ratio 1:21 (2,801 tool-use examples), Nano 42M achieves B4=0.145 ± 0.046 and
Base 260M achieves B4=0.580.

## Usage

```python
# Load with custom inference script
# See: https://huggingface.co/vectrayx/vectrayx-paper-code

from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
ckpt_path = hf_hub_download("vectrayx/vectrayx-nano", "nano_sft_v5.pt")
tokenizer_path = hf_hub_download("vectrayx/vectrayx-nano", "tokenizer/vectrayx_bpe.model")
config_path = hf_hub_download("vectrayx/vectrayx-nano", "configs/nano.json")
```

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
"""

DATASET_CARD = """---
language:
- es
license: cc-by-4.0
tags:
- cybersecurity
- spanish
- evaluation
- benchmark
task_categories:
- question-answering
- text-classification
- token-classification
---

# VectraYX-Bench

Five-task evaluation suite for Spanish cybersecurity language models.

## Tasks

| Task | File | Size | Metric |
|---|---|---|---|
| B1 CVE Q&A | `data/b1_cveqa.jsonl` | 500 prompts | Keyword recall |
| B2 Threat classification | `data/b2_classification.jsonl` | 200 examples | Accuracy + macro F1 |
| B3 Command completion | `data/b3_commands.jsonl` | 35 prompts | Exact match + tool match |
| B4 Tool selection | `data/b4_tooluse.jsonl` | 25 prompts | Tool accuracy |
| B5 Conversational gate | `data/b5_conversational.jsonl` | 10 prompts | Pass/fail |

## Tool-Use Corpus

The `tool_sft_corpus/` directory contains the tool-use training corpora used in the paper:

| File | Size | Description |
|---|---|---|
| `tool_sft_mini_v1.jsonl` | 2,801 examples | Ratio 1:21 — the winning density |
| `tool_sft_v3_bash.jsonl` | 296 examples | Bash-focused |
| `tool_sft_v2_simple.jsonl` | 115 examples | Simple bash |

## Paper Results

See `paper_results/` for pre-computed B1-B5 results from the paper.
"""


# ── Upload logic ───────────────────────────────────────────────────────────────

def create_and_upload(api, org, repo_key, dry_run=False):
    cfg = REPOS[repo_key]
    repo_id = cfg["repo_id"].format(org=org)
    repo_type = cfg["repo_type"]

    print(f"\n{'='*60}")
    print(f"Repo: {repo_id} ({repo_type})")
    print(f"{'='*60}")

    if not dry_run:
        try:
            create_repo(repo_id=repo_id, repo_type=repo_type,
                       exist_ok=True, private=False)
            print(f"  ✓ Repo created/exists: {repo_id}")
        except Exception as e:
            print(f"  ✗ Failed to create repo: {e}")
            return

    # Upload README / model card
    if "readme" in cfg:
        readme_path = RELEASE_DIR / cfg["readme"]
        if readme_path.exists():
            content = readme_path.read_text()
        elif repo_key == "models":
            content = MODEL_CARD
        elif repo_key == "datasets":
            content = DATASET_CARD
        else:
            content = None

        if content:
            if dry_run:
                print(f"  [DRY] Would upload README.md")
            else:
                try:
                    api.upload_file(
                        path_or_fileobj=content.encode(),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type=repo_type,
                    )
                    print(f"  ✓ README.md")
                except Exception as e:
                    print(f"  ✗ README: {e}")

    # Upload individual files
    for local_rel, repo_path in cfg.get("files", []):
        local = RELEASE_DIR / local_rel
        if not local.exists():
            print(f"  ⚠ SKIP (not found): {local_rel}")
            continue
        size_mb = local.stat().st_size / 1e6
        if dry_run:
            print(f"  [DRY] Would upload {local_rel} ({size_mb:.1f}MB) → {repo_path}")
        else:
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"  ✓ {repo_path} ({size_mb:.1f}MB)")
            except Exception as e:
                print(f"  ✗ {repo_path}: {e}")

    # Upload folders
    for local_rel, repo_path in cfg.get("folders", []):
        local = RELEASE_DIR / local_rel
        if not local.exists():
            print(f"  ⚠ SKIP (not found): {local_rel}")
            continue
        if dry_run:
            files = list(local.rglob("*"))
            print(f"  [DRY] Would upload folder {local_rel}/ ({len(files)} files) → {repo_path}")
        else:
            try:
                api.upload_folder(
                    folder_path=str(local),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"  ✓ {repo_path}/ (folder)")
            except Exception as e:
                print(f"  ✗ {repo_path}/: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--org", default="vectrayx", help="HuggingFace org/username")
    p.add_argument("--all", action="store_true", help="Upload everything")
    p.add_argument("--models", action="store_true", help="Upload model weights")
    p.add_argument("--datasets", action="store_true", help="Upload eval datasets")
    p.add_argument("--code", action="store_true", help="Upload reproducibility code")
    p.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = p.parse_args()

    if not any([args.all, args.models, args.datasets, args.code]):
        p.print_help()
        return

    # Accept token from env var or HUGGING_FACE_HUB_TOKEN
    import os
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if not args.dry_run:
        try:
            user = api.whoami()
            print(f"Logged in as: {user['name']}")
        except Exception as e:
            print(f"ERROR: Not logged in — {e}")
            print("Set token: export HUGGING_FACE_HUB_TOKEN=hf_...")
            print("Or run: python3 -c \"from huggingface_hub import login; login()\"")
            sys.exit(1)

    to_upload = []
    if args.all or args.models:   to_upload.append("models")
    if args.all or args.datasets: to_upload.append("datasets")
    if args.all or args.code:     to_upload.append("code")

    for repo_key in to_upload:
        create_and_upload(api, args.org, repo_key, dry_run=args.dry_run)

    print("\n✅ Done!")
    if not args.dry_run:
        print(f"\nRepos available at:")
        for repo_key in to_upload:
            repo_id = REPOS[repo_key]["repo_id"].format(org=args.org)
            print(f"  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
