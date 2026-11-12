#!/usr/bin/env python3
"""Is there a rerankable signal among the candidates the validator accepts?

`lm.evaluate --best-of N` leaves a gap between what greedy decoding picks and
what the best survivor would have scored (0.260 against 0.422 bb recall for
`micro18` at N=10). Closing it needs a selector that ranks candidates
*without* seeing the reference. This module measures whether any such signal
exists before anyone trains a model to exploit it.

For every candidate it computes reference-free features --- the model's own
mean logprob, and chemistry features based on Tanimoto similarity between the
proposed building blocks and the target, which is what retrosynthetic locality
would predict --- then reports, per feature:

* the score obtained by selecting the argmax candidate, against the greedy
  baseline, a random survivor and the oracle;
* the mean *within-target* Spearman correlation with true quality, with a 95 %
  interval over targets. This is the quantity that matters: a feature can
  correlate across targets merely by tracking target difficulty while being
  useless for choosing between two candidates for the same target.

    python -m lm.rerank models/eval/micro18_bo10_full.jsonl

Requires an evaluation run with `--best-of`, which stores every scored
survivor under `candidates`.
"""

import argparse
import json
import sys

FEATURES = ["score", "max_sim", "mean_sim", "sum_sim", "n_blocks", "n_mats"]


def features(target_fp, canonical, gen):
    """Reference-free description of one candidate."""
    from rdkit import DataStructs

    from lm.chem import COMMON_REAGENTS, SOLVENTS
    from lm.flatview import materials_of_text

    mats = materials_of_text(canonical or "")
    blocks = mats - SOLVENTS - COMMON_REAGENTS
    fps = [f for f in (fingerprint(s, gen) for s in blocks) if f is not None]
    sims = [DataStructs.TanimotoSimilarity(target_fp, f) for f in fps] or [0.0]
    return {"max_sim": max(sims), "mean_sim": sum(sims) / len(sims),
            "sum_sim": float(sum(sims)), "n_blocks": float(len(blocks)),
            "n_mats": float(len(mats))}


def fingerprint(smiles, gen):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    return gen.GetFingerprint(m) if m else None


def main():
    import numpy as np
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("evaluation", help="JSONL from `lm.evaluate --best-of N`")
    ap.add_argument("--key", default="bb_recall")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--bits", type=int, default=2048)
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius,
                                                    fpSize=args.bits)
    rows = []
    with open(args.evaluation, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cands = r.get("candidates") or []
            if len(cands) < 2:
                continue
            tfp = fingerprint(r["target"], gen)
            if tfp is None:
                continue
            rows.append([
                (c[args.key],
                 {"score": c.get("score") or 0.0,
                  **features(tfp, c["canonical"], gen)},
                 bool(c.get("greedy")))
                for c in cands])
    if not rows:
        raise SystemExit(f"{args.evaluation}: no row carries >=2 `candidates`; "
                         "re-run lm.evaluate --best-of N")

    q = [[x[0] for x in s] for s in rows]
    greedy = float(np.mean([next((x[0] for x in s if x[2]), s[0][0])
                            for s in rows]))
    oracle = float(np.mean([max(x) for x in q]))
    rng = np.random.default_rng(0)
    rand = float(np.mean([[qq[rng.integers(len(qq))] for qq in q]
                          for _ in range(500)]))

    out = {"evaluation": args.evaluation, "key": args.key, "targets": len(rows),
           "mean_candidates": round(float(np.mean([len(s) for s in rows])), 2),
           "mean_distinct_quality":
               round(float(np.mean([len(set(x)) for x in q])), 2),
           "greedy": round(greedy, 4), "random_survivor": round(rand, 4),
           "oracle": round(oracle, 4), "rules": {}, "within_target_rho": {}}

    for name in FEATURES:
        for sign, tag in ((1, "+"), (-1, "-")):
            sel = float(np.mean([max(s, key=lambda x: sign * x[1][name])[0]
                                 for s in rows]))
            out["rules"][tag + name] = round(sel, 4)
        cors = [spearmanr([x[1][name] for x in s], [x[0] for x in s]).statistic
                for s in rows
                if len({x[0] for x in s}) > 1 and len({x[1][name] for x in s}) > 1]
        cors = [c for c in cors if not np.isnan(c)]
        if cors:
            m = float(np.mean(cors))
            ci = 1.96 * float(np.std(cors)) / len(cors) ** 0.5
            out["within_target_rho"][name] = [round(m, 4), round(ci, 4)]

    best = max(out["rules"].items(), key=lambda kv: kv[1])
    out["best_rule"], out["best_rule_score"] = best
    out["headroom_unclaimed"] = round(oracle - best[1], 4)
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
