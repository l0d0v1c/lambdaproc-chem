#!/usr/bin/env python3
"""Representation ablation: does writing the procedure as a composition help?

Trains are done by `lm.train` on datasets built with `lm.prepare --view`.
This module evaluates them on a common footing:

* the **targets** and the **reference documents** come from `data/lm/test.jsonl`
  for every view, so the three models are asked the same questions and scored
  against the same ground truth;
* the metrics are material-level (`bb_recall`, `reactant_recall`,
  `valid_smiles`, `elements_ok`), which no view privileges --- all three keep
  the `(mat ...)` form verbatim;
* structural validity is reported only for the nested view, because it is the
  only one a parser can check. That asymmetry is itself a result.

    python -m lm.ablation models/micro/best       --view nested
    python -m lm.ablation models/micro_flat/best  --view flat  --data data/lm_flat
    python -m lm.ablation models/micro_mats/best  --view mats  --data data/lm_mats
"""

import argparse
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["valid_smiles", "elements_ok", "reactant_recall", "bb_recall",
        "valid_lproc"]


def score_materials(target, mats, refs) -> dict:
    """Material-level scoring, independent of the output representation."""
    from lm.chem import COMMON_REAGENTS, SOLVENTS, elements, reactants_of
    from lm.chem import building_blocks_of
    from rdkit import Chem

    row = {k: 0.0 for k in KEYS}
    if not mats:
        return row
    row["valid_smiles"] = float(all(Chem.MolFromSmiles(s) is not None for s in mats))
    have = set().union(*(elements(s) for s in mats)) if mats else set()
    row["elements_ok"] = float(elements(target) <= have)
    non_solvent = mats - SOLVENTS
    blocks = non_solvent - COMMON_REAGENTS
    best = best_bb = 0.0
    for ref in refs:
        key = reactants_of(ref)
        if key:
            best = max(best, len(key & non_solvent) / len(key))
        kb = building_blocks_of(ref)
        if kb:
            best_bb = max(best_bb, len(kb & blocks) / len(kb))
    row["reactant_recall"] = best
    row["bb_recall"] = best_bb
    return row


def main():
    import mlx.core as mx

    from lm.evaluate import pick_targets
    from lm.flatview import materials_of_text
    from lm.generate import generate
    from lm.lmview import check_view
    from lm.model import load
    from lm.tokenizer import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--view", required=True, choices=("nested", "flat", "mats"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lm"),
                    help="dataset of the view (for its vocabulary)")
    ap.add_argument("--refs", default=os.path.join(ROOT, "data/lm/test.jsonl"),
                    help="common references: always the nested test split")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--mem-gb", type=float, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mx.set_memory_limit(int(args.mem_gb * 2**30))
    # The reuse cache is unbounded by default and dwarfs live memory during
    # batched generation; see the note in lm.evaluate.
    mx.set_cache_limit(int(args.mem_gb * 2**30 / 4))
    refs = defaultdict(list)
    with open(args.refs, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            refs[r["product"]].append(r["canonical"])
    targets = pick_targets(sorted(refs), args.n)

    model = load(args.model)
    vocab = os.path.join(args.model, "vocab.json")
    tok = Tokenizer.load(vocab if os.path.exists(vocab)
                         else os.path.join(args.data, "vocab.json"))

    tot = {k: 0.0 for k in KEYS}
    rows = []
    for b in range(0, len(targets), args.bs):
        chunk = targets[b: b + args.bs]
        outs = generate(model, [tok.encode_prompt(t) for t in chunk])
        for t, ids in zip(chunk, outs):
            text = tok.decode_view(ids)
            row = {"target": t, "generated": text,
                   **score_materials(t, materials_of_text(text), refs[t])}
            if args.view == "nested":
                ok, res = check_view(text)
                row["valid_lproc"] = float(ok)
                if ok:
                    row["generated"] = res
            for k in KEYS:
                tot[k] += row[k]
            rows.append(row)
        print(f"{len(rows)}/{len(targets)} " + " ".join(
            f"{k}={tot[k] / len(rows):.3f}" for k in KEYS), flush=True)

    summary = {"model": args.model, "view": args.view, "targets": len(rows),
               "parser_checkable": args.view == "nested",
               **{k: round(tot[k] / len(rows), 4) for k in KEYS}}
    print("SUMMARY", json.dumps(summary))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
