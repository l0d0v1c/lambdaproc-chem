#!/usr/bin/env python3
"""Generate LambdaProc procedures for target molecules.

    python -m lm.generate models/small/best "CC(=O)Nc1ccc(O)cc1" -n 5 -t 0.8

Each sample is decoded, its (use ...) block rebuilt, then parsed and validated
by lambdaproc; invalid samples are reported as such (no silent filtering).
"""

import argparse
import os

import mlx.core as mx
import numpy as np

from lm.chem import elements_ok, smiles_ok
from lm.lmview import check_view
from lm.model import load
from lm.tokenizer import EOS, PAD, Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def canon_target(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse target SMILES {smiles!r}")
    return Chem.MolToSmiles(mol)


def generate(model, prompts, max_new=480, temperature=0.0, top_k=0, seed=0,
             return_logprobs=False):
    """Batched generation with KV cache. prompts: list of id lists.

    Prompts are left-padded; pad keys are masked out. RoPE is relative, so
    left padding does not change attention between real tokens.
    Returns token lists, or (tokens, mean model logprob per token) when
    return_logprobs is set (the score used to rank candidates).
    """
    mx.random.seed(seed)
    B = len(prompts)
    P = max(len(p) for p in prompts)
    x = np.full((B, P), PAD, dtype=np.int32)
    valid = np.zeros((B, P), dtype=bool)
    for i, p in enumerate(prompts):
        x[i, P - len(p):] = p
        valid[i, P - len(p):] = True
    keymask = mx.array(valid)                                    # (B, Lk)
    causal = mx.tril(mx.ones((P, P), dtype=mx.bool_))
    mask = causal[None, None] & keymask[:, None, None, :]
    logits, cache = model(mx.array(x), mask=mask)
    logits = logits[:, -1]
    out = [[] for _ in range(B)]
    lp_sum = np.zeros(B, dtype=np.float64)
    done = np.zeros(B, dtype=bool)
    for _ in range(max_new):
        if temperature <= 0:
            nxt = mx.argmax(logits, axis=-1)
        else:
            lg = logits / temperature
            if top_k:
                kth = mx.sort(lg, axis=-1)[:, -top_k][:, None]
                lg = mx.where(lg < kth, -mx.inf, lg)
            nxt = mx.random.categorical(lg)
        nxt_np = np.array(nxt)
        if return_logprobs:
            step_lp = np.array(mx.take_along_axis(
                logits - mx.logsumexp(logits, axis=-1, keepdims=True),
                nxt[:, None], axis=-1))[:, 0]
        for i in range(B):
            if not done[i]:
                out[i].append(int(nxt_np[i]))
                if return_logprobs:
                    lp_sum[i] += float(step_lp[i])
                if nxt_np[i] == EOS:
                    done[i] = True
        if done.all():
            break
        keymask = mx.concatenate([keymask, mx.ones((B, 1), dtype=mx.bool_)], axis=1)
        logits, cache = model(nxt[:, None], mask=keymask[:, None, None, :], cache=cache)
        logits = logits[:, -1]
    if return_logprobs:
        return out, [lp_sum[i] / max(1, len(out[i])) for i in range(B)]
    return out


def propose(model, tok, target, n=10, temperature=0.9, top_k=20, seed=0,
            inventory=None, max_new=480, include_greedy=True):
    """Sample n procedures for one target, keep the chemically sane ones.

    The lambdaproc validator and RDKit act as the filter (this is what the
    language buys us); surviving candidates are ranked by the model's own
    mean logprob. Returns the full list, best first, each annotated.
    """
    prompt = tok.encode_prompt(target, inventory)
    outs, scores = generate(model, [prompt] * n, max_new=max_new,
                            temperature=temperature, top_k=top_k, seed=seed,
                            return_logprobs=True)
    greedy_flags = [False] * len(outs)
    if include_greedy:
        # Ranking by model logprob alone is a poor selector (measured: it is
        # worse than greedy decoding), so the greedy procedure always competes
        # and wins ties; sampling then only adds alternatives.
        g_out, g_score = generate(model, [prompt], max_new=max_new,
                                  temperature=0.0, return_logprobs=True)
        outs, scores = g_out + outs, g_score + scores
        greedy_flags = [True] + greedy_flags
    cands = []
    for ids, score, is_greedy in zip(outs, scores, greedy_flags):
        view = tok.decode_view(ids)
        ok, res = check_view(view)
        c = {"score": round(score, 4), "greedy": is_greedy, "valid_lproc": ok,
             "canonical": res if ok else None,
             "reason": None if ok else res, "text": view}
        if ok:
            c["valid_smiles"] = smiles_ok(res)
            c["elements_ok"] = elements_ok(target, res)
            if not c["valid_smiles"]:
                c["reason"] = "invalid material SMILES"
            elif not c["elements_ok"]:
                c["reason"] = "target elements missing from materials"
        else:
            c["valid_smiles"] = c["elements_ok"] = False
        c["kept"] = bool(ok and c["valid_smiles"] and c["elements_ok"])
        cands.append(c)
    # dedupe identical procedures, keep the best-scoring copy
    seen, uniq = {}, []
    for c in sorted(cands, key=lambda c: (-c["kept"], -c["greedy"], -c["score"])):
        key = c["canonical"] or c["text"]
        if key in seen:
            seen[key]["n_samples"] += 1
            seen[key]["greedy"] = seen[key]["greedy"] or c["greedy"]
            continue
        c["n_samples"] = 1
        seen[key] = c
        uniq.append(c)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("targets", nargs="+", help="target SMILES")
    ap.add_argument("--vocab", default=os.path.join(ROOT, "data/lm/vocab.json"))
    ap.add_argument("-n", type=int, default=10, help="samples per target")
    ap.add_argument("-t", "--temperature", type=float, default=0.9)
    ap.add_argument("-k", "--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--have", nargs="*", default=None,
                    help="inventory SMILES (requires a model trained with one)")
    ap.add_argument("--show", type=int, default=3, help="procedures to print")
    ap.add_argument("--all", action="store_true", help="also print rejected samples")
    args = ap.parse_args()

    model = load(args.model)
    vocab_path = os.path.join(args.model, "vocab.json")
    tok = Tokenizer.load(vocab_path if os.path.exists(vocab_path) else args.vocab)
    inventory = [canon_target(s) for s in args.have] if args.have else None
    for target in args.targets:
        t = canon_target(target)
        cands = propose(model, tok, t, n=args.n, temperature=args.temperature,
                        top_k=args.top_k, seed=args.seed, inventory=inventory)
        kept = [c for c in cands if c["kept"]]
        print(f"; ===== target {t}")
        print(f"; {len(kept)}/{len(cands)} distinct proposals passed validation "
              f"({args.n} samples)")
        for j, c in enumerate(kept[: args.show]):
            print(f"; --- proposal {j + 1}  score {c['score']}  "
                  f"sampled {c['n_samples']}x")
            print(c["canonical"])
        if args.all:
            for c in (c for c in cands if not c["kept"]):
                print(f"; --- rejected: {c['reason']}")
                print(c["canonical"] or c["text"])


if __name__ == "__main__":
    main()
