"""VectraYX-Nano transformer (decoder-only, ~42M params).

Modern small-LLM stack:
  RMSNorm (pre-norm)  ·  SwiGLU FFN  ·  RoPE  ·  GQA (8q/2kv)
  QK-Norm  ·  no biases  ·  tied embeddings  ·  z-loss
"""

import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_model: int = 512
    d_ffn: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    init_std: float = 0.02
    dropout: float = 0.0
    tie_embeddings: bool = True
    qk_norm: bool = True
    z_loss_coef: float = 1e-4

    @classmethod
    def from_json(cls, path):
        cfg = json.loads(open(path).read())["model"]
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(self.weight.dtype) * self.weight


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: (B, H, T, D) with D even.  cos/sin: (T, D/2)
    T, D = x.shape[-2], x.shape[-1]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2:]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.repeat = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_eps)

        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w_down = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VectraYXNano(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, m):
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def num_params(self, exclude_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"seq {T} > max {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        cos = self.rope_cos
        sin = self.rope_sin
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        # cross-entropy + z-loss for stability
        flat_logits = logits.view(-1, logits.size(-1))
        flat_tgt = targets.view(-1)
        ce = F.cross_entropy(flat_logits, flat_tgt, reduction="none", ignore_index=-100)
        if loss_mask is not None:
            mask = loss_mask.view(-1).float()
            denom = mask.sum().clamp_min(1.0)
            ce_loss = (ce * mask).sum() / denom
        else:
            valid = (flat_tgt != -100).float()
            denom = valid.sum().clamp_min(1.0)
            ce_loss = (ce * valid).sum() / denom

        if self.cfg.z_loss_coef > 0:
            lse = torch.logsumexp(flat_logits.float(), dim=-1)
            if loss_mask is not None:
                z = ((lse ** 2) * loss_mask.view(-1).float()).sum() / denom
            else:
                z = ((lse ** 2) * (flat_tgt != -100).float()).sum() / denom
            loss = ce_loss + self.cfg.z_loss_coef * z
        else:
            loss = ce_loss
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40, top_p=0.9,
                 eos_id=None, repeat_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(cond)
            logits = logits[:, -1, :].float()

            if repeat_penalty != 1.0:
                for token in set(idx[0].tolist()):
                    logits[0, token] = logits[0, token] / repeat_penalty if logits[0, token] > 0 else logits[0, token] * repeat_penalty

            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = probs.cumsum(-1)
                    drop = cumprobs > top_p
                    drop[..., 1:] = drop[..., :-1].clone()
                    drop[..., 0] = False
                    sorted_logits[drop] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx
