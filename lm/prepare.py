#!/usr/bin/env python3
"""Build the token datasets for the target -> procedure LM.

Input : data/uspto/corpus_smiles_full_dedup.jsonl (scripts/run_full_uspto.py --smiles)
Output: data/lm/{train,val,test}.npz, data/lm/vocab.json, data/lm/test.jsonl,
        data/lm/meta.json

Split by product (sha1 of the target SMILES): every procedure of a given
target lands in the same split, so test targets are unseen molecules.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lm.flatview import project  # noqa: E402
from lm.tokenizer import Tokenizer  # noqa: E402


def split_of(product: str, val_pm: int, test_pm: int) -> str:
    h = int.from_bytes(hashlib.sha1(product.encode()).digest()[:4], "big") % 1000
    return "test" if h < test_pm else "val" if h < test_pm + val_pm else "train"



def _hard_neighbours(products, pool, top=12):
    """For each requested product, the pool materials most similar to it.

    ~3.3 ms per product, so restricting this to the products drawn into the
    `hard` regime turns half an hour into a few minutes.
    """
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smiles):
        m = Chem.MolFromSmiles(smiles)
        return gen.GetFingerprint(m) if m else None

    pool_fp, pool_ok = [], []
    for s in pool:
        f = fp(s)
        if f is not None:
            pool_fp.append(f)
            pool_ok.append(s)
    products = sorted(products)
    print(f"hard neighbours for {len(products)} products", flush=True)
    out = {}
    for i, p in enumerate(products):
        f = fp(p)
        if f is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, pool_fp))
        out[p] = [pool_ok[j] for j in np.argsort(-sims)[:top] if pool_ok[j] != p]
        if (i + 1) % 25000 == 0:
            print(f"  hard neighbours {i + 1}/{len(products)}", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data/uspto/corpus_smiles_full_dedup.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/lm"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-permille", type=int, default=10)
    ap.add_argument("--test-permille", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=3,
                    help="tokens rarer than this in train map to <unk>")
    ap.add_argument("--inventory", action="store_true",
                    help="condition on an available-materials shelf (lm.inventory)")
    ap.add_argument("--shelf-mix", default="trained",
                    choices=("trained", "robust"),
                    help="trained: every shelf holds ~80 %% of the true "
                         "reactants. robust: a mixture that also includes "
                         "shelves with no true reactant and shelves whose "
                         "distractors are near-misses (lm.inventory.REGIMES)")
    ap.add_argument("--pool-size", type=int, default=20000,
                    help="most frequent train reactants used as distractors")
    ap.add_argument("--view", default="nested", choices=("nested", "flat", "mats"),
                    help="output representation (lm.flatview): the ablation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from lm.tokenizer import lproc_tokens, smiles_tokens
    recs = {"train": [], "val": [], "test": []}
    bad = 0
    with open(args.src, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            try:
                view = project(r["canonical"], args.view)
            except Exception:
                bad += 1
                continue
            try:
                smiles_tokens(r["product"]), lproc_tokens(view)
            except ValueError:
                bad += 1
                continue
            recs[split_of(r["product"], args.val_permille, args.test_permille)].append(
                (r["id"], r["product"], view, r["canonical"]))
    print({k: len(v) for k, v in recs.items()}, "view:", args.view,
          "unusable:", bad)

    rng = np.random.default_rng(args.seed)
    pool, invs = None, {}
    if args.inventory:
        from collections import Counter
        from lm import inventory as inv_mod
        from lm.chem import reactants_of
        counts = Counter()
        for _, _, _, canon in recs["train"]:
            counts.update(reactants_of(canon))
        pool = [s for s, _ in counts.most_common(args.pool_size)]
        inv_mod.save_pool(os.path.join(args.out, "pool.json"), pool)
        print("distractor pool", len(pool))
        # Draw the regimes first, so the Tanimoto sweep runs only for the
        # products that actually need near-miss distractors (~15 %).
        regimes = ({split: [inv_mod.draw_regime(rng) for _ in rows]
                    for split, rows in recs.items()}
                   if args.shelf_mix == "robust" else {})
        need_hard = {p for split, rows in recs.items()
                     for (_, p, _, _), w in zip(rows, regimes.get(split, []))
                     if w == "hard"}
        hard_of = (_hard_neighbours(need_hard, pool) if need_hard else {})
        regime_counts = Counter()
        for split, rows in recs.items():
            # never put the target itself on the shelf (it is a material in
            # ~1 % of references, via purification actions)
            if args.shelf_mix == "trained":
                invs[split] = [inv_mod.build(reactants_of(c) - {p}, pool, rng)
                               for _, p, _, c in rows]
            else:
                out = []
                for (_, p, _, c), w in zip(rows, regimes[split]):
                    shelf, which = inv_mod.build_mixed(
                        reactants_of(c) - {p}, pool, rng, regime=w,
                        hard_pool=hard_of.get(p))
                    out.append(shelf)
                    regime_counts[f"{split}/{which}"] += 1
                invs[split] = out
        if regime_counts:
            print("shelf regimes", dict(sorted(regime_counts.items())))

    def pairs_for(split):
        for i, (_, p, v, _) in enumerate(recs[split]):
            yield (p, invs[split][i], v) if args.inventory else (p, v)

    tok = Tokenizer.build(pairs_for("train"), args.min_count)
    tok.save(os.path.join(args.out, "vocab.json"))
    print("vocab", len(tok))

    meta = {"view": args.view,"vocab": len(tok), "max_len": args.max_len, "src": args.src,
            "untokenizable": bad, "inventory": args.inventory, "splits": {}}
    for split, rows in recs.items():
        flat, offsets, plens, kept, n_unk, too_long = [], [0], [], [], 0, 0
        for k, (rid, product, view, canon) in enumerate(rows):
            shelf = invs[split][k] if args.inventory else None
            ids, plen = tok.encode_pair(product, view, shelf)
            if len(ids) > args.max_len:
                too_long += 1
                continue
            n_unk += ids.count(4)
            flat.extend(ids)
            offsets.append(len(flat))
            plens.append(plen)
            kept.append((rid, product, canon, shelf))
        np.savez(os.path.join(args.out, f"{split}.npz"),
                 tokens=np.array(flat, dtype=np.uint16),
                 offsets=np.array(offsets, dtype=np.int64),
                 plens=np.array(plens, dtype=np.int32))
        if split != "train":
            with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
                for rid, product, canon, shelf in kept:
                    rec = {"id": rid, "product": product, "canonical": canon}
                    if shelf is not None:
                        rec["inventory"] = shelf
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta["splits"][split] = {"examples": len(kept), "tokens": len(flat),
                                 "dropped_too_long": too_long, "unk_tokens": n_unk,
                                 "products": len({p for _, p, _, _ in kept})}
        print(split, meta["splits"][split])
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
