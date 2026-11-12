"""Alternative output representations, for the representation ablation.

The paper claims that writing a procedure as a *composition* does part of the
work for the model. Testing that claim means training the same model, on the
same corpus and the same budget, to emit weaker representations:

``nested``  the LambdaProc view (the baseline): materials, then the procedure
            as a nested composition --- nesting says which stream each
            operation consumes.
``flat``    materials, then a linear action sequence in the style of an
            action-extraction pipeline: each operation with the materials it
            introduces, the stream left implicit. A projection of ``nested``
            that discards the stream structure.
``mats``    materials only, no operations at all. Isolates how much of the
            material-level score comes from the procedure at all.

``flat`` is lossy, and `lossy_flat` says exactly when: an operation taking two
or more *computed* arguments (a join in the DAG, e.g. combining two prepared
streams) cannot be linearised, because a single implicit stream cannot stand
for two. Measuring that fraction is part of the ablation.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lambdaproc import App, Def, Document, Lambda, Ref, parse  # noqa: E402

VIEWS = ("nested", "flat", "mats")
MAT_LINE_RE = re.compile(r"^\(mat .*\)$", re.M)


def _body(doc: Document):
    for form in doc.forms:
        if isinstance(form, Def) and not isinstance(form.body, Lambda):
            return form.body
    raise ValueError("no (def ...) body found")


def _mat_line(canon: str) -> str:
    m = MAT_LINE_RE.search(canon)
    if not m:
        raise ValueError("no (mat ...) form found")
    return m.group(0)


def linearise(expr, out=None, lossy=None):
    """Post-order linearisation. Returns a list of 'OP arg arg' strings."""
    out = [] if out is None else out
    lossy = [False] if lossy is None else lossy
    if isinstance(expr, Ref):
        return out, lossy
    apps = [a for a in expr.args if isinstance(a, App)]
    refs = [a for a in expr.args if isinstance(a, Ref)]
    if len(apps) > 1:
        lossy[0] = True
    for a in apps:
        linearise(a, out, lossy)
    out.append(" ".join([str(expr.head)] + [str(r) for r in refs]))
    return out, lossy


def to_flat(canon: str) -> str:
    """Materials, then a linear action sequence (one line, ' | ' separated)."""
    steps, _ = linearise(_body(parse(canon)))
    return _mat_line(canon) + "\n" + " | ".join(steps)


def to_mats(canon: str) -> str:
    return _mat_line(canon)


def lossy_flat(canon: str) -> bool:
    """True when linearisation loses a stream join."""
    _, lossy = linearise(_body(parse(canon)))
    return lossy[0]


def project(canon: str, view: str) -> str:
    if view == "flat":
        return to_flat(canon)
    if view == "mats":
        return to_mats(canon)
    from lm.lmview import to_view
    return to_view(canon)


# --------------------------------------------------------------- scoring --
def materials_of_text(text: str) -> set:
    """Materials of a generated text, whatever the view.

    Every view keeps the `(mat (sym "SMILES") ...)` form, so the three can be
    scored on materials with one code path -- which is the point: the ablation
    compares representations on a metric none of them privileges.
    """
    return set(re.findall(r'\(\s*[a-z]+[0-9]*\s+"([^"]*)"\s*\)', text))


def main():
    import argparse
    import json
    from collections import Counter

    ap = argparse.ArgumentParser(description="corpus statistics of the views")
    ap.add_argument("--corpus", default=os.path.join(
        ROOT, "data/uspto/corpus_smiles_full_dedup.jsonl"))
    ap.add_argument("--n", type=int, default=200000)
    args = ap.parse_args()

    n = lossy = 0
    lens = Counter()
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            if n >= args.n:
                break
            canon = json.loads(line)["canonical"]
            try:
                steps, lo = linearise(_body(parse(canon)))
            except Exception:
                continue
            n += 1
            lossy += bool(lo[0])
            lens[len(steps)] += 1
    print(json.dumps({
        "documents": n,
        "lossy_under_flattening": lossy,
        "lossy_fraction": round(lossy / n, 4),
        "mean_steps": round(sum(k * v for k, v in lens.items()) / n, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
