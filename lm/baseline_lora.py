#!/usr/bin/env python3
"""Pretrained-LLM baseline: export the corpus for mlx-lm LoRA, then evaluate.

Answers one question: does fine-tuning a small pretrained LLM beat our
from-scratch model on the same held-out targets and the same metrics?

    # 1. export (uses the same product-hash split as lm.prepare)
    python -m lm.baseline_lora export --n-train 100000
    # 2. fine-tune (mlx-lm CLI, one GPU job at a time)
    python -m mlx_lm.lora --model LiquidAI/LFM2-350M --train \
        --data data/lfm2 --batch-size 8 --iters 3000 --adapter-path models/lfm2_lora
    # 3. evaluate with lm.evaluate's metrics
    python -m lm.baseline_lora eval --adapter models/lfm2_lora --n 300

mlx-lm's CompletionsDataset wraps every prompt/completion pair in the model's
**chat template** (user/assistant turns), so evaluation must do the same: a
plain-text prompt is off-distribution and the adapter degenerates visibly the
longer it is trained (see lm/README.md, "Pretrained baseline"). The completion
is the same LM view the from-scratch model produces, so both are judged by the
same validator and the same chemistry checks.
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "target: {}\nprocedure:\n"


def export(args):
    from lm.lmview import to_view
    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for split, name, limit in (("train", "train", args.n_train),
                               ("val", "valid", args.n_valid)):
        src = os.path.join(args.data, f"{split}.jsonl")
        if split == "train":
            # train.jsonl is not written by lm.prepare (tokens only): rebuild
            # the split from the corpus with the same hash rule.
            from lm.prepare import split_of
            rows = []
            with open(args.corpus, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if split_of(r["product"], 10, 10) != "train":
                        continue
                    rows.append(r)
                    if limit and len(rows) >= limit:
                        break
        else:
            rows = [json.loads(l) for l in open(src, encoding="utf-8")][:limit]
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                view = to_view(r["canonical"])
                fh.write(json.dumps({"prompt": PROMPT.format(r["product"]),
                                     "completion": view}, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    print(json.dumps(counts, indent=2))
    print(f"wrote {args.out}/train.jsonl and valid.jsonl")


def chat_prompt(tok, target) -> list:
    """Token ids for one target, wrapped exactly as training wrapped them."""
    return tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(target)}],
        add_generation_prompt=True)


def evaluate(args):
    import mlx.core as mx
    from mlx_lm import batch_generate, load

    from lm.chem import reactants_of
    from lm.evaluate import KEYS, pick_targets, score
    from lm.lmview import check_view

    mx.set_memory_limit(int(args.mem_gb * 2**30))
    mx.set_cache_limit(int(args.mem_gb * 2**30 / 4))
    refs = {}
    with open(os.path.join(args.data, f"{args.split}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            refs.setdefault(r["product"], []).append(r["canonical"])
    targets = pick_targets(sorted(refs), args.n, args.sample)

    model, tok = load(args.model, adapter_path=args.adapter)
    keys = list(KEYS)
    tot = {k: 0.0 for k in keys}
    rows = []
    for b in range(0, len(targets), args.bs):
        chunk = targets[b : b + args.bs]
        prompts = [chat_prompt(tok, t) for t in chunk]
        res = batch_generate(model, tok, prompts, max_tokens=args.max_tokens,
                             verbose=False)
        for t, text in zip(chunk, res.texts):
            view = text.split("<|im_end|>")[0].split("target:")[0].strip()
            ok, canon = check_view(view)
            row = {"target": t, "generated": canon if ok else view,
                   **score(t, canon if ok else None, refs[t])}
            for k in keys:
                tot[k] += row[k]
            rows.append(row)
        print(f"{len(rows)}/{len(targets)} " + " ".join(
            f"{k}={tot[k] / len(rows):.3f}" for k in keys), flush=True)
    summary = {"model": args.model, "adapter": args.adapter, "split": args.split,
               "targets": len(rows), "mode": "greedy", "sample": bool(args.sample),
               **{k: round(tot[k] / len(rows), 4) for k in keys}}
    print("SUMMARY", json.dumps(summary))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--corpus", default=os.path.join(ROOT, "data/uspto/corpus_smiles_full_dedup.jsonl"))
    e.add_argument("--data", default=os.path.join(ROOT, "data/lm"))
    e.add_argument("--out", default=os.path.join(ROOT, "data/lfm2"))
    e.add_argument("--n-train", type=int, default=100000)
    e.add_argument("--n-valid", type=int, default=500)
    e.set_defaults(func=export)

    v = sub.add_parser("eval")
    v.add_argument("--model", default="LiquidAI/LFM2-350M")
    v.add_argument("--adapter", default=None)
    v.add_argument("--data", default=os.path.join(ROOT, "data/lm"))
    v.add_argument("--split", default="test")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--bs", type=int, default=32)
    v.add_argument("--max-tokens", type=int, default=400)
    v.add_argument("--mem-gb", type=float, default=48)
    v.add_argument("--sample", action="store_true",
                   help="seeded random target slice (see lm.evaluate)")
    v.add_argument("--out", default=None)
    v.set_defaults(func=evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
