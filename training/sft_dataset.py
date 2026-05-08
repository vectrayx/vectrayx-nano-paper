"""SFT dataset with proper assistant-only loss masking and safe packing.

Each example is a chat-formatted string with `<|system|> <|user|> <|assistant|> <|end|>`
turn delimiters.  We tokenize on the fly (corpus is small, ~25M tokens) and build a
mask=1 only on tokens that are part of an assistant response (everything between
`<|assistant|>` and the next `<|end|>`).

For pre-training-style packing without cross-example contamination we group multiple
short examples into a fixed-length window using `cu_seqlens`-style document boundaries
implemented via per-document attention reset.  Here we keep it simple: pad/truncate
each example to `block_size`. Throughput is still high (>40k tok/s on L4) for this
volume.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text") or ""
            if t:
                out.append({"text": t, "source": obj.get("source", Path(path).stem)})
    return out


def build_assistant_mask(token_ids, assistant_id, end_id):
    """mask[i] = 1 iff token_ids[i] is inside an `<|assistant|> ... <|end|>` span.

    We mark from the token AFTER `<|assistant|>` up to and including `<|end|>` so the
    model learns to emit the closing delimiter.
    """
    mask = np.zeros(len(token_ids), dtype=np.int64)
    inside = False
    for i, t in enumerate(token_ids):
        if t == assistant_id and not inside:
            inside = True
            continue  # don't include the assistant tag itself
        if inside:
            mask[i] = 1
            if t == end_id:
                inside = False
    return mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_paths, sp, block_size, assistant_token="<|assistant|>",
                 end_token="<|end|>", pad_id=0, seed=42, mix_weights=None):
        self.sp = sp
        self.block_size = block_size
        self.pad_id = pad_id
        self.assistant_id = sp.piece_to_id(assistant_token)
        self.end_id = sp.piece_to_id(end_token)
        if self.assistant_id < 0 or self.end_id < 0:
            raise ValueError(f"missing special tokens in tokenizer: "
                             f"{assistant_token}={self.assistant_id} {end_token}={self.end_id}")

        self.examples = []
        rng = random.Random(seed)
        for p in jsonl_paths:
            recs = _read_jsonl(p)
            w = (mix_weights or {}).get(Path(p).name, 1.0)
            if w != 1.0:
                k = int(len(recs) * w)
                recs = rng.sample(recs, min(k, len(recs)))
            self.examples.extend(recs)
            print(f"  [sft] {p}: {len(recs):,} ex (w={w})")
        rng.shuffle(self.examples)
        print(f"[sft] total: {len(self.examples):,} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]["text"]
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: self.block_size + 1]
        mask = build_assistant_mask(ids, self.assistant_id, self.end_id)

        if len(ids) < self.block_size + 1:
            need = self.block_size + 1 - len(ids)
            ids = ids + [self.pad_id] * need
            mask = np.concatenate([mask, np.zeros(need, dtype=np.int64)])

        ids = np.asarray(ids, dtype=np.int64)
        x = torch.from_numpy(ids[:-1])
        y = torch.from_numpy(ids[1:].copy())
        m = torch.from_numpy(mask[1:].copy())  # mask aligned with targets
        # zero out padded targets
        y[m == 0] = -100
        return x, y, m
