# VectraYX Reproducibility Makefile
# Reproduces the key experiments from the paper on a single NVIDIA L4 / A10G GPU.
#
# Prerequisites:
#   - Python 3.10+
#   - CUDA 12.1+
#   - pip install -r requirements.txt
#   - AWS CLI configured (for SageMaker experiments)
#   - Nano checkpoint: nano_sft_v5.pt  (download from HuggingFace, see README)
#   - Base checkpoint: base_phase3_last.pt (download from HuggingFace, see README)
#
# Usage:
#   make install          # install dependencies
#   make bench-nano       # run B1-B5 on Nano 42M (requires nano_sft_v5.pt)
#   make bench-base       # run B1-B5 on Base 260M (requires base_phase3_last.pt)
#   make lora-nano        # LoRA tool-use fine-tune on Nano (local GPU)
#   make lora-base        # LoRA tool-use fine-tune on Base (local GPU)
#   make repro            # full reproducibility run (bench + lora + bench again)
#   make corpus           # regenerate tool_sft_mini_v1.jsonl from scratch

PYTHON     := python3
NANO_CKPT  := checkpoints/nano_sft_v5.pt
BASE_CKPT  := checkpoints/base_phase3_last.pt
TOKENIZER  := checkpoints/vectrayx_bpe.model
NANO_CFG   := configs/nano.json
BASE_CFG   := configs/base.json
EVAL_DIR   := eval_data
CORPUS     := corpus/tool_sft_mini_v1.jsonl
LORA_OUT   := checkpoints/lora_out

.PHONY: install bench-nano bench-base lora-nano lora-base repro corpus clean help

help:
	@echo "VectraYX Reproducibility Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install Python dependencies"
	@echo "  bench-nano   Run B1-B5 benchmark on Nano 42M"
	@echo "  bench-base   Run B1-B5 benchmark on Base 260M"
	@echo "  lora-nano    LoRA fine-tune Nano 42M on tool-use corpus"
	@echo "  lora-base    LoRA fine-tune Base 260M on tool-use corpus"
	@echo "  repro        Full reproducibility run"
	@echo "  corpus       Regenerate tool_sft_mini_v1.jsonl"
	@echo "  clean        Remove generated checkpoints and results"

install:
	pip install -r requirements.txt

# ── Benchmark ──────────────────────────────────────────────────────────────────

bench-nano: $(NANO_CKPT) $(TOKENIZER)
	$(PYTHON) eval/benchmark.py \
		--config $(NANO_CFG) \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(NANO_CKPT) \
		--data-dir $(EVAL_DIR) \
		--out results/bench_nano_baseline.json
	@echo "Results: results/bench_nano_baseline.json"

bench-base: $(BASE_CKPT) $(TOKENIZER)
	$(PYTHON) eval/benchmark.py \
		--config $(BASE_CFG) \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(BASE_CKPT) \
		--data-dir $(EVAL_DIR) \
		--out results/bench_base_baseline.json
	@echo "Results: results/bench_base_baseline.json"

# ── LoRA fine-tune ─────────────────────────────────────────────────────────────

lora-nano: $(NANO_CKPT) $(TOKENIZER) $(CORPUS)
	mkdir -p $(LORA_OUT)/nano
	$(PYTHON) training/finetune_lora_tools.py \
		--config $(NANO_CFG) \
		--tokenizer $(TOKENIZER) \
		--resume $(NANO_CKPT) \
		--tool-corpus $(CORPUS) \
		--out $(LORA_OUT)/nano \
		--lora-rank 16 --lora-alpha 32 \
		--batch-size 16 --grad-accum 4 \
		--epochs 5 --lr 2e-4 --seed 42
	$(PYTHON) eval/run_inference_lora.py \
		--base-checkpoint $(NANO_CKPT) \
		--lora-checkpoint $(LORA_OUT)/nano/final_lora_only.pt \
		--config $(NANO_CFG) \
		--tokenizer $(TOKENIZER) \
		--data-dir $(EVAL_DIR) \
		--out results/bench_nano_lora_s42.json
	@echo "Results: results/bench_nano_lora_s42.json"

lora-base: $(BASE_CKPT) $(TOKENIZER) $(CORPUS)
	mkdir -p $(LORA_OUT)/base
	$(PYTHON) training/finetune_lora_tools.py \
		--config $(BASE_CFG) \
		--tokenizer $(TOKENIZER) \
		--resume $(BASE_CKPT) \
		--tool-corpus $(CORPUS) \
		--out $(LORA_OUT)/base \
		--lora-rank 16 --lora-alpha 32 \
		--batch-size 8 --grad-accum 8 \
		--epochs 5 --lr 2e-4 --seed 42
	$(PYTHON) eval/run_inference_lora.py \
		--base-checkpoint $(BASE_CKPT) \
		--lora-checkpoint $(LORA_OUT)/base/final_lora_only.pt \
		--config $(BASE_CFG) \
		--tokenizer $(TOKENIZER) \
		--data-dir $(EVAL_DIR) \
		--out results/bench_base_lora_s42.json
	@echo "Results: results/bench_base_lora_s42.json"

# ── Full reproducibility run ───────────────────────────────────────────────────

repro: install bench-nano bench-base lora-nano lora-base
	@echo ""
	@echo "=== Reproducibility Run Complete ==="
	@echo "Expected results (from paper, Table 3):"
	@echo "  Nano baseline  B4=0.000"
	@echo "  Base baseline  B4=0.000"
	@echo "  Nano + LoRA    B4=0.145 ± 0.046 (seed 42: 0.220)"
	@echo "  Base + LoRA    B4=0.580"
	@echo ""
	@echo "Your results:"
	@$(PYTHON) -c "import json; \
		r = {k: json.load(open(f'results/{k}.json')) for k in \
		['bench_nano_baseline','bench_base_baseline','bench_nano_lora_s42','bench_base_lora_s42'] \
		if __import__('pathlib').Path(f'results/{k}.json').exists()}; \
		[print(f'  {k}: B4={v.get(\"B4_tooluse\",\"N/A\")}') for k,v in r.items()]"

# ── Corpus generation ──────────────────────────────────────────────────────────

corpus:
	$(PYTHON) corpus/build_mini_tool_corpus.py \
		--size 2801 \
		--out corpus/tool_sft_mini_v1_repro.jsonl
	@echo "Generated: corpus/tool_sft_mini_v1_repro.jsonl"
	@echo "Note: compare with corpus/tool_sft_mini_v1.jsonl (released version)"

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -rf checkpoints/lora_out results/
	@echo "Cleaned generated files. Checkpoints preserved."
