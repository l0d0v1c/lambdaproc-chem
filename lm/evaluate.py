#!/usr/bin/env python3
"""Evaluate a target -> procedure model on held-out (unseen) target molecules.

    python -m lm.evaluate models/micro/best --n 1000                  # greedy
    python -m lm.evaluate models/micro/best --n 500 --best-of 10       # sampling
    python -m lm.evaluate models/cond/best --n 500 --data data/lm_cond # inventory

One procedure per distinct test target; references are all the corpus
procedures for that target. Metrics (fractions of targets):
  valid_lproc     decoded text parses, (use) rebuilds, lambdaproc validates
  valid_smiles    every material SMILES parses with RDKit
  elements_ok     every heavy element of the target occurs in some material
  kept            passes all three (what a user would be shown)
  reactant_recall recall of the best reference's non-solvent materials
  bb_recall       the same over building blocks only (ubiquitous reagents
                  dropped as well): the metric that separates a model from a
                  nearest-neighbour lookup, see lm.baseline_retrieval
  ops_exact       operation skeleton (indices stripped) equals a reference
  exact           canonical document equals a reference
With --best-of N: metrics are those of the top-ranked surviving candidate,
plus recall_oracle (best of the N) and n_kept (distinct surviving proposals).
With an inventory dataset: inv_precision (share of used non-solvent materials
that were on the shelf) and inv_recall (share of needed shelf items used).
"""

import argparse
import json
import os
from collections import defaultdict

from lm.chem import (building_blocks_of, elements_ok, reactants_of,
                     skeleton, smiles_ok)
from lm.generate import generate, propose
from lm.lmview import check_view
from lm.model import load
from lm.tokenizer import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["valid_lproc", "valid_smiles", "elements_ok", "kept",
        "reactant_recall", "bb_recall", "ops_exact", "exact"]


# Fixed, independent of --seed: every model must see the same target slice.
SAMPLE_SEED = 12345


def pick_targets(all_targets, n, sample=False) -> list:
    """The n targets to evaluate.

    Default (historical) order is alphabetical, which concentrates on one
    region of chemical space -- fine for comparing models against each other,
    misleading as an absolute number. --sample draws a seeded random slice.
    """
    ts = list(all_targets)
    if sample:
        import random
        random.Random(SAMPLE_SEED).shuffle(ts)
    return ts[:n]


def score(target, canon, refs, shelf=None) -> dict:
    """Quality of one generated procedure against all references of a target."""
    row = {k: 0.0 for k in KEYS}
    if canon is None:
        return row
    row["valid_lproc"] = 1.0
    row["valid_smiles"] = float(smiles_ok(canon))
    row["elements_ok"] = float(elements_ok(target, canon))
    row["kept"] = float(row["valid_smiles"] and row["elements_ok"])
    mats = reactants_of(canon)
    best = 0.0
    for ref in refs:
        key = reactants_of(ref)
        if key:
            best = max(best, len(key & mats) / len(key))
    row["reactant_recall"] = best
    # Same recall over building blocks only: no solvents, no ubiquitous
    # reagents. reactant_recall flatters any baseline that copies a plausible
    # work-up -- a constant answer scores 0.133 on it and 0.000 here.
    blocks = building_blocks_of(canon)
    best_bb = 0.0
    for ref in refs:
        key = building_blocks_of(ref)
        if key:
            best_bb = max(best_bb, len(key & blocks) / len(key))
    row["bb_recall"] = best_bb
    sk = skeleton(canon)
    row["ops_exact"] = float(any(sk == skeleton(r) for r in refs))
    row["exact"] = float(canon in refs)
    if shelf is not None:
        on_shelf = set(shelf)
        row["inv_precision"] = len(mats & on_shelf) / len(mats) if mats else 0.0
        needed = set().union(*(reactants_of(r) for r in refs)) & on_shelf
        row["inv_recall"] = len(needed & mats) / len(needed) if needed else 1.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lm"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000, help="distinct targets")
    ap.add_argument("--bs", type=int, default=250, help="targets per batch (greedy)")
    ap.add_argument("--best-of", type=int, default=0,
                    help="sample N candidates per target and keep the valid ones")
    ap.add_argument("-t", "--temperature", type=float, default=0.9)
    ap.add_argument("-k", "--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true",
                    help="seeded random target sample instead of the "
                         "alphabetically first n (which over-represents one "
                         "region of chemical space)")
    ap.add_argument("--out", default=None, help="per-target results JSONL")
    # MLX keeps freed buffers in a reuse cache that is unbounded by default:
    # at --bs 250 it grew to 196 GB of retained allocations for 2 GB of live
    # ones, which is a memory-pressure alert on a 256 GB machine. Cap both,
    # as lm.train does.
    ap.add_argument("--mem-gb", type=float, default=24)
    args = ap.parse_args()

    import mlx.core as mx
    mx.set_memory_limit(int(args.mem_gb * 2**30))
    mx.set_cache_limit(int(args.mem_gb * 2**30 / 4))

    refs, shelves = defaultdict(list), {}
    with open(os.path.join(args.data, f"{args.split}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            refs[r["product"]].append(r["canonical"])
            if "inventory" in r:
                shelves.setdefault(r["product"], r["inventory"])
    targets = pick_targets(sorted(refs), args.n, args.sample)
    use_inv = bool(shelves)

    model = load(args.model)
    tok = Tokenizer.load(os.path.join(args.data, "vocab.json"))
    keys = list(KEYS)
    if args.best_of:
        keys += ["recall_oracle", "n_kept"]
    if use_inv:
        keys += ["inv_precision", "inv_recall"]
    tot = {k: 0.0 for k in keys}
    rows = []

    def account(row):
        for k in keys:
            tot[k] += row.get(k, 0.0)
        rows.append(row)
        print(f"{len(rows)}/{len(targets)} " + " ".join(
            f"{k}={tot[k] / len(rows):.3f}" for k in keys), flush=True)

    if args.best_of:
        for t in targets:
            shelf = shelves.get(t) if use_inv else None
            cands = propose(model, tok, t, n=args.best_of, temperature=args.temperature,
                            top_k=args.top_k, seed=args.seed, inventory=shelf)
            kept = [c for c in cands if c["kept"]]
            top = kept[0]["canonical"] if kept else None
            row = {"target": t, "generated": top, **score(t, top, refs[t], shelf)}
            row["n_kept"] = len(kept)
            # Score every survivor, not only the first: the oracle and any
            # reranking study need the whole shelf of candidates. Storing all
            # of them costs a few MB and is what `lm.rerank` reads.
            per = [score(t, c["canonical"], refs[t], shelf) for c in kept]
            row["recall_oracle"] = max(
                (p["reactant_recall"] for p in per), default=0.0)
            row["bb_oracle"] = max((p["bb_recall"] for p in per), default=0.0)
            row["candidates"] = [
                {"canonical": c["canonical"], "score": c.get("score"),
                 "greedy": c.get("greedy"), "bb_recall": p["bb_recall"],
                 "reactant_recall": p["reactant_recall"]}
                for c, p in zip(kept, per)]
            row["alternatives"] = [c["canonical"] for c in kept[1:4]]
            account(row)
    else:
        for b in range(0, len(targets), args.bs):
            chunk = targets[b : b + args.bs]
            prompts = [tok.encode_prompt(t, shelves.get(t) if use_inv else None)
                       for t in chunk]
            outs = generate(model, prompts)
            for t, ids in zip(chunk, outs):
                view = tok.decode_view(ids)
                ok, res = check_view(view)
                canon = res if ok else None
                shelf = shelves.get(t) if use_inv else None
                row = {"target": t, "generated": canon or view,
                       **score(t, canon, refs[t], shelf)}
                account(row)

    summary = {"model": args.model, "split": args.split, "targets": len(rows),
               "sample": bool(args.sample),
               "mode": f"best-of-{args.best_of} t={args.temperature}" if args.best_of
                       else "greedy",
               "inventory": use_inv,
               **{k: round(tot[k] / len(rows), 4) for k in keys}}
    print("SUMMARY", json.dumps(summary))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
