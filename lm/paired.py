#!/usr/bin/env python3
"""Paired comparison of two saved evaluations on a common set of targets.

Every evaluation in `models/eval/*.jsonl` stores one row per target with the
same metric keys, so two systems can be compared target by target rather than
by their means. That matters here: the differences we care about are a few
points on a metric whose per-target standard deviation is an order of magnitude
larger, and an unpaired comparison of means cannot tell a real effect from
the sampling of the 1000 targets.

Reports, for metric `--key`:

* the paired mean difference A - B with a 95 % confidence interval from
  `--iters` bootstrap resamples of the *targets* (not of the rows, so both
  systems are resampled together);
* a two-sided Wilcoxon signed-rank p-value on the non-zero differences,
  which makes no normality assumption and is insensitive to the heavy
  right tail of per-target recall.

    python -m lm.paired models/eval/abl_nested.jsonl models/eval/abl_mats.jsonl
"""

import argparse
import json


def load(path, key):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if key not in r:
                raise SystemExit(f"{path}: no key {key!r}; has {sorted(r)}")
            out[r["target"]] = float(r[key])
    return out


def main():
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", default="bb_recall")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, B = load(args.a, args.key), load(args.b, args.key)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("no targets in common")
    a = np.array([A[t] for t in shared])
    b = np.array([B[t] for t in shared])
    d = a - b

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(d), size=(args.iters, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    from scipy.stats import wilcoxon
    nz = d[d != 0]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0

    print(json.dumps({
        "key": args.key, "a": args.a, "b": args.b,
        "n_shared": len(shared), "n_differing": int(len(nz)),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "diff": round(float(d.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "wilcoxon_p": float(f"{p:.3g}"),
        "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum()),
    }))


if __name__ == "__main__":
    main()
