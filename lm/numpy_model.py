"""Pure-NumPy forward pass for the trained checkpoints.

Why this exists: it is the reference for running the model anywhere MLX is not
available --- in particular in a browser under Pyodide, where `lambdaproc`
(standard library only) and `lm.tokenizer` already run unchanged. Keeping the
inference in NumPy means the web demo runs the *same* tokeniser, the *same*
validator and the *same* canonicaliser as the published results, with no second
implementation to keep in sync.

    from lm.numpy_model import NumpyGPT
    m = NumpyGPT.load("models/micro18/best")
    ids = m.generate(prompt_ids, max_new=480)          # greedy
    ids = m.generate(prompt_ids, temperature=0.8, top_k=20, rng=np.random.default_rng(0))

Verified to reproduce the MLX implementation token for token
(`python -m lm.numpy_model --check models/micro18/best`).
"""

import argparse
import json
import os

import numpy as np

EPS = 1e-5  # mlx.nn.RMSNorm default


def _read_safetensors(path: str) -> dict:
    """Minimal safetensors reader: no dependency beyond NumPy."""
    dtypes = {"F32": np.float32, "F16": np.float16, "BF16": None}
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
        blob = fh.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = blob[start:end]
        dt = dtypes.get(meta["dtype"])
        if dt is None:  # bfloat16 -> float32 by zero-extending the mantissa
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=dt)
        out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def rms_norm(x, w):
    return x * w / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class NumpyGPT:
    """Decoder-only transformer: RoPE, RMSNorm, SwiGLU, tied embeddings."""

    def __init__(self, cfg: dict, w: dict):
        self.d = cfg["d"]
        self.layers = cfg["layers"]
        self.heads = cfg["heads"]
        self.hd = self.d // self.heads
        self.vocab = cfg["vocab"]
        self.w = w
        # RoPE frequencies, non-traditional (MLX default): the head dimension
        # is split in half and the halves rotate into each other.
        half = self.hd // 2
        self.inv_freq = 10000.0 ** (-np.arange(half, dtype=np.float32) / half)

    @classmethod
    def load(cls, path: str) -> "NumpyGPT":
        with open(os.path.join(path, "config.json")) as fh:
            cfg = json.load(fh)
        return cls(cfg, _read_safetensors(os.path.join(path, "weights.safetensors")))

    # -------------------------------------------------------------- pieces
    def _rope(self, x, offset: int):
        """x: (B, H, L, hd). Rotates the two halves of the head dimension."""
        B, H, L, hd = x.shape
        pos = np.arange(offset, offset + L, dtype=np.float32)[:, None]
        ang = pos * self.inv_freq[None, :]            # (L, hd/2)
        cos, sin = np.cos(ang)[None, None], np.sin(ang)[None, None]
        a, b = x[..., : hd // 2], x[..., hd // 2:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    def _block(self, i, x, cache, mask):
        w, B, L, _ = self.w, *x.shape
        p = f"blocks.{i}."
        h = rms_norm(x, w[p + "n1.weight"])
        qkv = h @ w[p + "attn.qkv.weight"].T
        q, k, v = np.split(qkv, 3, axis=-1)
        shape = (B, L, self.heads, self.hd)
        q, k, v = (t.reshape(shape).transpose(0, 2, 1, 3) for t in (q, k, v))
        off = 0 if cache is None else cache[0].shape[2]
        q, k = self._rope(q, off), self._rope(k, off)
        if cache is not None:
            k = np.concatenate([cache[0], k], axis=2)
            v = np.concatenate([cache[1], v], axis=2)
        att = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5)
        if mask is not None:
            # A large finite penalty, not -inf: left-padded query rows are
            # fully masked, and -inf would make their softmax NaN and poison
            # the residual stream.
            att = np.where(mask, att, -1e9)
        o = softmax(att) @ v
        x = x + (o.transpose(0, 2, 1, 3).reshape(B, L, self.d)
                 @ w[p + "attn.out.weight"].T)
        h = rms_norm(x, w[p + "n2.weight"])
        ff = silu(h @ w[p + "w1.weight"].T) * (h @ w[p + "w3.weight"].T)
        return x + ff @ w[p + "w2.weight"].T, (k, v)

    def __call__(self, ids, cache=None, mask=None):
        """ids: (B, L) int. Returns (logits of the last position, cache)."""
        h = self.w["emb.weight"][ids]
        new = []
        for i in range(self.layers):
            h, c = self._block(i, h, None if cache is None else cache[i], mask)
            new.append(c)
        h = rms_norm(h, self.w["norm.weight"])
        return h @ self.w["emb.weight"].T, new      # tied embeddings

    # ------------------------------------------------------------ sampling
    def generate(self, prompts, max_new=480, temperature=0.0, top_k=0,
                 rng=None, eos=3, pad=0, return_logprobs=False):
        """prompts: list of id lists (left-padded internally). Returns lists.

        With `return_logprobs`, also returns the mean model logprob per
        generated token, computed exactly as `lm.generate` does, since that
        score ranks the candidates of the sampling filter.
        """
        if prompts and isinstance(prompts[0], int):
            prompts = [prompts]
        B = len(prompts)
        P = max(len(p) for p in prompts)
        x = np.full((B, P), pad, dtype=np.int64)
        valid = np.zeros((B, P), dtype=bool)
        for i, p in enumerate(prompts):
            x[i, P - len(p):] = p
            valid[i, P - len(p):] = True
        causal = np.tril(np.ones((P, P), dtype=bool))
        mask = causal[None, None] & valid[:, None, None, :]
        logits, cache = self(x, mask=mask)
        logits = logits[:, -1]
        keymask = valid
        out = [[] for _ in range(B)]
        lp_sum = np.zeros(B, dtype=np.float64)
        done = np.zeros(B, dtype=bool)
        rng = rng or np.random.default_rng(0)
        for _ in range(max_new):
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                lg = logits / temperature
                if top_k:
                    cut = np.partition(lg, -top_k, axis=-1)[:, -top_k][:, None]
                    lg = np.where(lg < cut, -np.inf, lg)
                probs = softmax(lg)
                nxt = np.array([rng.choice(self.vocab, p=p) for p in probs])
            if return_logprobs:
                m = logits.max(axis=-1, keepdims=True)
                lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
                step_lp = logits[np.arange(B), nxt] - lse
            for i in range(B):
                if not done[i]:
                    out[i].append(int(nxt[i]))
                    if return_logprobs:
                        lp_sum[i] += float(step_lp[i])
                    if nxt[i] == eos:
                        done[i] = True
            if done.all():
                break
            # Keep masking the pad keys as the cache grows, exactly as the
            # MLX implementation does.
            keymask = np.concatenate(
                [keymask, np.ones((B, 1), dtype=bool)], axis=1)
            logits, cache = self(nxt[:, None], cache=cache,
                                 mask=keymask[:, None, None, :])
            logits = logits[:, -1]
        if return_logprobs:
            return out, [lp_sum[i] / max(1, len(out[i])) for i in range(B)]
        return out


def _check(path, n_tokens=120):
    """Compare greedy decoding against the MLX implementation, token by token."""
    import time

    import mlx.core as mx

    from lm.model import load as mlx_load
    from lm.tokenizer import Tokenizer
    from lm.generate import generate as mlx_generate

    tok = Tokenizer.load(os.path.join(path, "vocab.json"))
    targets = ["CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O",
               "Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
    prompts = [tok.encode_prompt(t) for t in targets]

    mx.set_memory_limit(4 * 2**30)
    ref = mlx_generate(mlx_load(path), prompts, max_new=n_tokens)

    m = NumpyGPT.load(path)
    t0 = time.time()
    got = m.generate([list(p) for p in prompts], max_new=n_tokens)
    dt = time.time() - t0

    ok = all(list(a) == list(b) for a, b in zip(ref, got))
    ntok = sum(len(g) for g in got)
    print(f"identical to MLX: {ok}   ({len(targets)} prompts, {ntok} tokens)")
    for a, b in zip(ref, got):
        if list(a) != list(b):
            j = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            print(f"  first divergence at token {j}: mlx={a[j]} numpy={b[j]}")
    print(f"numpy: {dt:.2f}s for {ntok} tokens "
          f"({ntok / dt:.0f} tok/s, batch of {len(targets)})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--check", action="store_true",
                    help="verify against the MLX implementation")
    ap.add_argument("--tokens", type=int, default=120)
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if _check(args.model, args.tokens) else 1)
    m = NumpyGPT.load(args.model)
    print(f"loaded {args.model}: {m.layers} layers, d={m.d}, vocab={m.vocab}")


if __name__ == "__main__":
    main()
