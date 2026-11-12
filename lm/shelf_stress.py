#!/usr/bin/env python3
"""Stress-test the inventory-conditioned model on shelves it was not trained on.

The training shelf (`lm.inventory.build`) holds ~80 % of the true reactants and
a few distractors drawn at random from the corpus. Random distractors are
chemically unrelated to the target, so rejecting them is easy, and copying the
whole shelf already scores 0.760 bb recall --- more than the model. This module
asks the harder questions a real inventory poses, at evaluation time only:

* ``hard``         the same shelf, distractors replaced by the pool materials
                   most similar to the target (Morgan r=2 Tanimoto);
* ``absent``       true reactants removed, shelf size kept, random fill;
* ``absent_hard``  true reactants removed, hard distractors only;
* ``large``        the training shelf padded to 10 items (16 overflows the 512-token training context) with random ones.

    python -m lm.shelf_stress build                     # data/lm_cond_<v>/
    python -m lm.evaluate models/cond/best --data data/lm_cond_hard \
        --out models/eval/cond_hard.jsonl                # one per variant
    python -m lm.shelf_stress score                     # table + baselines

Caveat for ``hard``: the most similar pool material is sometimes a legitimate
alternative reactant (another ester of the same acid). Scored against the
reference it counts as wrong, so hard-shelf numbers are lower bounds.
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ["trained", "hard", "absent", "absent_hard", "large"]
LARGE = 10


def _fps(smiles_list, gen):
    from rdkit import Chem
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprint(m) if m else None)
    return out


def build(args):
    import numpy as np
    from rdkit import DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    from lm.chem import COMMON_REAGENTS, SOLVENTS, reactants_of

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    src = os.path.join(ROOT, "data/lm_cond")
    pool = json.load(open(os.path.join(src, "pool.json"), encoding="utf-8"))
    pool = [p for p in pool if p not in SOLVENTS and p not in COMMON_REAGENTS]
    pool_fp = _fps(pool, gen)
    keep = [i for i, f in enumerate(pool_fp) if f is not None]
    pool = [pool[i] for i in keep]
    pool_fp = [pool_fp[i] for i in keep]
    rng = np.random.default_rng(args.seed)

    rows = [json.loads(l) for l in open(os.path.join(src, "test.jsonl"),
                                        encoding="utf-8")]
    true_by_target = defaultdict(set)
    for r in rows:
        true_by_target[r["product"]] |= reactants_of(r["canonical"])

    cache = {}

    def variants_for(target, shelf):
        if target in cache:
            return cache[target]
        true = true_by_target[target]
        kept = [s for s in shelf if s in true]
        distract = [s for s in shelf if s not in true]
        tfp = _fps([target], gen)[0]
        sims = (np.array(DataStructs.BulkTanimotoSimilarity(tfp, pool_fp))
                if tfp is not None else np.zeros(len(pool)))
        hard = []
        for i in np.argsort(-sims):
            c = pool[i]
            if c not in true and c != target and c not in hard:
                hard.append(c)
            if len(hard) >= len(shelf):
                break

        def rand_fill(base, n):
            out, tries = list(base), 0
            while len(out) < n and tries < 1000:
                tries += 1
                c = pool[rng.integers(len(pool))]
                if c not in true and c not in out and c != target:
                    out.append(c)
            return out

        v = {
            "trained": sorted(shelf),
            "hard": sorted(set(kept) | set(hard[:len(distract)])),
            "absent": sorted(rand_fill(distract, len(shelf))),
            "absent_hard": sorted(hard[:len(shelf)]),
            "large": sorted(rand_fill(shelf, LARGE)),
        }
        cache[target] = v
        return v

    outs = {v: open_variant(v) for v in VARIANTS}
    for r in rows:
        vs = variants_for(r["product"], r["inventory"])
        for v, fh in outs.items():
            fh.write(json.dumps({**r, "inventory": vs[v]}, ensure_ascii=False) + "\n")
    for fh in outs.values():
        fh.close()
    print("built", ", ".join(f"data/lm_cond_{v}" for v in VARIANTS))


def open_variant(v):
    d = os.path.join(ROOT, f"data/lm_cond_{v}")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "data/lm_cond/vocab.json"), d)
    return open(os.path.join(d, "test.jsonl"), "w", encoding="utf-8")


def score(args):
    import numpy as np
    from rdkit import DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    from lm.chem import (COMMON_REAGENTS, SOLVENTS, building_blocks_of,
                         elements, reactants_of)
    from lm.flatview import materials_of_text

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def metrics(target, mats, refs, shelf):
        mats = set(mats)
        blocks = mats - SOLVENTS - COMMON_REAGENTS
        ns = mats - SOLVENTS
        bb = rr = prec = 0.0
        for ref in refs:
            kb, kr = building_blocks_of(ref), reactants_of(ref)
            if kb:
                bb = max(bb, len(kb & blocks) / len(kb))
                if blocks:
                    prec = max(prec, len(kb & blocks) / len(blocks))
            if kr:
                rr = max(rr, len(kr & ns) / len(kr))
        have = set().union(*(elements(s) for s in mats)) if mats else set()
        from_shelf = len(blocks & set(shelf)) / len(blocks) if blocks else 0.0
        return {"bb_recall": bb, "bb_precision": prec, "reactant_recall": rr,
                "elements_ok": float(elements(target) <= have),
                "from_shelf": from_shelf, "n_blocks": float(len(blocks))}

    table = {}
    for v in VARIANTS:
        data = os.path.join(ROOT, f"data/lm_cond_{v}/test.jsonl")
        ev = os.path.join(ROOT, f"models/eval/cond_{v}.jsonl")
        if not os.path.exists(ev):
            print(f"skip {v}: no {ev}")
            continue
        refs, shelves = defaultdict(list), {}
        for l in open(data, encoding="utf-8"):
            r = json.loads(l)
            refs[r["product"]].append(r["canonical"])
            shelves.setdefault(r["product"], r["inventory"])
        acc = defaultdict(lambda: defaultdict(list))
        for l in open(ev, encoding="utf-8"):
            r = json.loads(l)
            t, shelf = r["target"], shelves[r["target"]]
            # Same convention as lm.evaluate: a document the validator
            # rejects proposes nothing and scores zero.
            mm = (materials_of_text(r["generated"] or "")
                  if float(r.get("valid_lproc", 0)) else set())
            for k, x in metrics(t, mm, refs[t], shelf).items():
                acc["model"][k].append(x)
            for k, x in metrics(t, shelf, refs[t], shelf).items():
                acc["copy shelf"][k].append(x)
            tf = _fps([t], gen)[0]
            cand = [s for s in shelf if s not in SOLVENTS and s not in COMMON_REAGENTS]
            cf = _fps(cand, gen)
            order = sorted(range(len(cand)), key=lambda i: -(
                DataStructs.TanimotoSimilarity(tf, cf[i]) if tf and cf[i] else 0))
            m = max(1, int(acc["model"]["n_blocks"][-1]))
            for k, x in metrics(t, [cand[i] for i in order[:m]], refs[t], shelf).items():
                acc["top-m Tanimoto"][k].append(x)
            kb = max((building_blocks_of(x) for x in refs[t]), key=len, default=set())
            if kb:
                acc["ceiling"]["on_shelf"].append(len(kb & set(shelf)) / len(kb))
        table[v] = {name: {k: round(float(np.mean(x)), 4) for k, x in d.items()}
                    for name, d in acc.items()}
        print(f"\n=== {v}  (true building blocks on shelf: "
              f"{table[v]['ceiling']['on_shelf']:.3f})")
        for name in ["model", "copy shelf", "top-m Tanimoto"]:
            print(f"  {name:15s} " + "  ".join(
                f"{k}={x:.3f}" for k, x in table[v][name].items()))
    with open(os.path.join(ROOT, "models/eval/shelf_stress.json"), "w") as fh:
        json.dump(table, fh, indent=2)


def elements(args):
    """Why the conditioned model's elements_ok is below the free model's.

    `elements_ok` asks whether every element of the target appears in some
    proposed material. A model restricted to a shelf inherits a ceiling: when
    the shelf (plus solvents, which are always assumed available) does not
    between them contain every element of the target, no shelf-obedient
    procedure can satisfy the metric. Since each true reactant is kept with
    probability 0.8, that case is built into the training distribution.

    Reports the ceiling, the measured value, and how the shortfall below the
    ceiling divides between skipping a true reactant and skipping a
    distractor that merely happened to carry the element.
    """
    import numpy as np

    from lm.chem import SOLVENTS, elements as elems, reactants_of

    shelves, refs = {}, defaultdict(list)
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            shelves.setdefault(r["product"], r["inventory"])
            refs[r["product"]].append(r["canonical"])
    rows = {}
    for path, name in ((args.cond, "cond"), (args.free, "free")):
        with open(path, encoding="utf-8") as fh:
            rows[name] = {r["target"]: r for r in map(json.loads, fh)}

    sol = set().union(*(elems(s) for s in SOLVENTS))
    ts = sorted(rows["cond"])
    ceiling = np.mean([float(elems(t) <= (
        set().union(*(elems(s) for s in shelves[t])) | sol)) for t in ts])

    skipped_true = skipped_distractor = shelf_short = 0
    for t in ts:
        r = rows["cond"][t]
        if r["elements_ok"] >= 1:
            continue
        mats = reactants_of(r["generated"]) if r["valid_lproc"] else set()
        have = set().union(*(elems(s) for s in mats)) if mats else set()
        miss = elems(t) - have
        shelf_el = set().union(*(elems(s) for s in shelves[t])) | sol
        if not miss <= shelf_el:
            shelf_short += 1
            continue
        true_r = set().union(*(reactants_of(x) for x in refs[t]))
        avail = [s for s in shelves[t] if (elems(s) & miss) and s not in mats]
        if any(s in true_r for s in avail):
            skipped_true += 1
        else:
            skipped_distractor += 1

    shared = sorted(set(rows["cond"]) & set(rows["free"]))
    out = {
        "targets": len(ts),
        "free_elements_ok": round(float(np.mean(
            [rows["free"][t]["elements_ok"] for t in shared])), 4),
        "cond_elements_ok": round(float(np.mean(
            [rows["cond"][t]["elements_ok"] for t in ts])), 4),
        "ceiling_shelf_plus_solvents": round(float(ceiling), 4),
        "failures_shelf_could_not_cover": shelf_short,
        "failures_skipped_true_reactant": skipped_true,
        "failures_skipped_distractor_only": skipped_distractor,
        "materials_used_cond": round(float(np.mean(
            [len(reactants_of(rows["cond"][t]["generated"]))
             if rows["cond"][t]["valid_lproc"] else 0 for t in shared])), 3),
        "materials_used_free": round(float(np.mean(
            [len(reactants_of(rows["free"][t]["generated"]))
             if rows["free"][t]["valid_lproc"] else 0 for t in shared])), 3),
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    with open(os.path.join(ROOT, "models/eval/elements_audit.json"), "w") as fh:
        json.dump(out, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "score", "elements"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lm_cond/test.jsonl"))
    ap.add_argument("--cond", default=os.path.join(ROOT, "models/eval/cond.jsonl"))
    ap.add_argument("--free", default=os.path.join(ROOT, "models/eval/micro18.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"build": build, "score": score, "elements": elements}[args.cmd](args)


if __name__ == "__main__":
    main()
