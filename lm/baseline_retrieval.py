#!/usr/bin/env python3
"""Trivial baselines: is the model doing anything a lookup could not?

Two questions a reviewer will ask before reading any model result.

1. **Nearest-neighbour retrieval.** Take the most similar *training* molecule
   (Morgan fingerprint, Tanimoto) and copy its procedure verbatim. If that
   matches the model, the model is pointless.
2. **Most frequent reactants.** Always propose the k commonest non-solvent
   materials of the corpus, ignoring the target entirely. This is the floor.

Both are scored with exactly the metrics of `lm.evaluate`, on exactly the same
test targets, so the numbers drop straight into the same table.

    python -m lm.baseline_retrieval --n 1000 --out models/eval/retrieval.jsonl
    python -m lm.baseline_retrieval --n 1000 --topk 10     # retrieval oracle
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def morgan_generator(bits: int, radius: int):
    from rdkit.Chem import rdFingerprintGenerator
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def fingerprints(smiles_list, bits, radius, desc=""):
    """Pack Morgan fingerprints of a list of SMILES into a uint8 bit matrix."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    gen = morgan_generator(bits, radius)
    out = np.zeros((len(smiles_list), bits // 8), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[i] = np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
        ok[i] = True
        if desc and i and i % 100000 == 0:
            print(f"  {desc} {i}/{len(smiles_list)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out, ok


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def tanimoto(query_row, matrix, pop_matrix):
    """Tanimoto of one packed fingerprint against a packed matrix."""
    inter = POPCOUNT[np.bitwise_and(matrix, query_row)].sum(axis=1)
    pq = int(POPCOUNT[query_row].sum())
    union = pop_matrix + pq - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def build_index(corpus, bits, radius, cache):
    """product -> one canonical procedure, plus a packed fingerprint matrix."""
    from lm.prepare import split_of
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"index loaded from {cache}: {z['fps'].shape[0]} molecules")
        return list(z["prods"]), list(z["procs"]), z["fps"]
    first = {}
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if split_of(r["product"], 10, 10) != "train":
                continue
            first.setdefault(r["product"], r["canonical"])
    prods = list(first)
    procs = [first[p] for p in prods]
    print(f"train molecules: {len(prods)}", flush=True)
    fps, ok = fingerprints(prods, bits, radius, desc="fingerprinting")
    prods = [p for p, k in zip(prods, ok) if k]
    procs = [c for c, k in zip(procs, ok) if k]
    fps = fps[ok]
    print(f"usable: {len(prods)}", flush=True)
    if cache:
        np.savez_compressed(cache, prods=np.array(prods, dtype=object),
                            procs=np.array(procs, dtype=object), fps=fps)
    return prods, procs, fps


def main():
    from lm.chem import SOLVENTS, mats_of, reactants_of
    from lm.evaluate import KEYS, pick_targets, score
    from lm.lmview import from_view, to_view

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(
        ROOT, "data/uspto/corpus_smiles_full_dedup.jsonl"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lm"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=1,
                    help=">1 reports the oracle over the k nearest neighbours")
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--freq-k", type=int, default=4,
                    help="materials proposed by the frequency baseline")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--cache", default=os.path.join(ROOT, "data/lm/retrieval_index.npz"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    refs = defaultdict(list)
    with open(os.path.join(args.data, f"{args.split}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            refs[r["product"]].append(r["canonical"])
    targets = pick_targets(sorted(refs), args.n, args.sample)

    prods, procs, fps = build_index(args.corpus, args.bits, args.radius, args.cache)
    pop = POPCOUNT[fps].sum(axis=1).astype(np.int32)

    # ---- frequency floor: the k commonest non-solvent corpus materials ----
    counts = Counter()
    for c in procs[:200000]:
        counts.update(reactants_of(c))
    common = [s for s, _ in counts.most_common(args.freq_k)]
    print(f"frequency baseline proposes: {common}", flush=True)

    qfps, qok = fingerprints(targets, args.bits, args.radius)

    keys = list(KEYS)
    tot_r = {k: 0.0 for k in keys}
    tot_f = {k: 0.0 for k in keys}
    oracle = 0.0
    sims = []
    rows = []
    t0 = time.time()
    for i, t in enumerate(targets):
        if not qok[i]:
            continue
        sim = tanimoto(qfps[i], fps, pop)
        order = np.argpartition(-sim, args.topk)[: args.topk]
        order = order[np.argsort(-sim[order])]
        j = int(order[0])
        sims.append(float(sim[j]))
        row = {"target": t, "neighbour": prods[j], "similarity": round(float(sim[j]), 4),
               "generated": procs[j], **score(t, procs[j], refs[t])}
        if args.topk > 1:
            row["recall_oracle"] = max(
                score(t, procs[int(k)], refs[t])["reactant_recall"] for k in order)
            oracle += row["recall_oracle"]
        for k in keys:
            tot_r[k] += row[k]
        # frequency floor: a fixed material list, no operations of substance
        freq_doc = from_view(
            "(mat " + " ".join(f'({chr(97 + z)} "{s}")' for z, s in enumerate(common))
            + ")\n(def procede (ADD " + " ".join(chr(97 + z) for z in range(len(common)))
            + "))")
        fr = score(t, freq_doc, refs[t])
        for k in keys:
            tot_f[k] += fr[k]
        rows.append(row)
        if len(rows) % 200 == 0:
            print(f"{len(rows)}/{len(targets)} "
                  f"retrieval recall={tot_r['reactant_recall'] / len(rows):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    n = len(rows)
    summary = {
        "baseline": "nearest-neighbour retrieval",
        "split": args.split, "targets": n, "topk": args.topk,
        "fingerprint": f"Morgan r={args.radius} {args.bits} bits",
        "mean_similarity": round(float(np.mean(sims)), 4),
        **{k: round(tot_r[k] / n, 4) for k in keys},
    }
    if args.topk > 1:
        summary["recall_oracle"] = round(oracle / n, 4)
    freq = {"baseline": f"{args.freq_k} most frequent materials", "targets": n,
            **{k: round(tot_f[k] / n, 4) for k in keys}}
    print("SUMMARY", json.dumps(summary))
    print("FLOOR  ", json.dumps(freq))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
            json.dump({"retrieval": summary, "frequency_floor": freq}, fh, indent=2)


if __name__ == "__main__":
    main()
