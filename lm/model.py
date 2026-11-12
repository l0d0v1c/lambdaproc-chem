"""Tiny decoder-only transformer in MLX (RoPE, RMSNorm, SwiGLU, tied embeddings)."""

import json
import os
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    vocab: int
    d: int = 256
    layers: int = 6
    heads: int = 8
    ff_mult: float = 8 / 3
    max_len: int = 512

    @property
    def ff(self) -> int:
        return int(self.d * self.ff_mult + 63) // 64 * 64


PRESETS = {
    "nano":  dict(d=128, layers=4, heads=4),    # ~0.8 M params
    "micro": dict(d=192, layers=6, heads=6),    # ~2.7 M
    "small": dict(d=256, layers=8, heads=8),    # ~6.3 M
    "base":  dict(d=384, layers=10, heads=8),   # ~17.7 M
}


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.heads
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.out = nn.Linear(cfg.d, cfg.d, bias=False)
        self.rope = nn.RoPE(cfg.d // cfg.heads)
        self.scale = (cfg.d // cfg.heads) ** -0.5

    def __call__(self, x, mask, cache=None):
        B, L, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, L, self.h, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
        off = cache[0].shape[2] if cache is not None else 0
        q, k = self.rope(q, offset=off), self.rope(k, offset=off)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, L, D)), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(cfg.d), nn.RMSNorm(cfg.d)
        self.attn = Attention(cfg)
        self.w1 = nn.Linear(cfg.d, cfg.ff, bias=False)
        self.w3 = nn.Linear(cfg.d, cfg.ff, bias=False)
        self.w2 = nn.Linear(cfg.ff, cfg.d, bias=False)

    def __call__(self, x, mask, cache=None):
        a, cache = self.attn(self.n1(x), mask, cache)
        x = x + a
        h = self.n2(x)
        return x + self.w2(nn.silu(self.w1(h)) * self.w3(h)), cache


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.norm = nn.RMSNorm(cfg.d)

    def __call__(self, x, mask="causal", cache=None):
        h = self.emb(x)
        new_cache = []
        for i, b in enumerate(self.blocks):
            h, c = b(h, mask, None if cache is None else cache[i])
            new_cache.append(c)
        return self.emb.as_linear(self.norm(h)), new_cache

    def n_params(self) -> int:
        from mlx.utils import tree_flatten
        return sum(v.size for _, v in tree_flatten(self.parameters()))


def save(model: GPT, path: str, extra: dict = None):
    os.makedirs(path, exist_ok=True)
    model.save_weights(os.path.join(path, "weights.safetensors"))
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump({**asdict(model.cfg), **(extra or {})}, fh, indent=2)


def load(path: str) -> GPT:
    with open(os.path.join(path, "config.json")) as fh:
        raw = json.load(fh)
    cfg = Config(**{k: raw[k] for k in Config.__dataclass_fields__})
    model = GPT(cfg)
    model.load_weights(os.path.join(path, "weights.safetensors"))
    model.eval()
    return model
