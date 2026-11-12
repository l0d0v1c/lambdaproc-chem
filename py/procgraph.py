"""Turn a validated λproc document into a drawable process graph.

The page shows the canonical text, which is exact but reads badly: a chemist
sees `(REM (FIL (DRY (WAS (EXT (ADD ...` and has to unwind the nesting by eye.
This module hands the JS side a structure it can draw as a flowsheet.

Everything here reads the AST produced by `lambdaproc.parse` and identifies
nodes by `canon_expr`, the canonicaliser's own key. The drawing is therefore a
rendering of the parsed document, not of the string: there is no second parser
to drift from the first, which is the property the whole demo rests on.

Two consequences of using the canonical key are deliberate:

* Two textually distinct mentions of the same sub-process are **one** node, so
  the picture shows the DAG the language denotes, not the tree the text spells
  out. The second mention is emitted as a `ref` so the drawing can point back
  instead of duplicating a whole branch.
* Arguments of a commutative operation are keyed in canonical order, so a
  document and its commutation give the same picture.
"""

import json
import sys

from lambdaproc import App, Def, Lambda, MatDecl, Ref, UseDecl, parse
from lambdaproc.core import build_env, canon_expr


def _expand(expr, env):
    """Inline 0-ary macro references, as `lambdaproc.to_dag` does.

    A macro used once is a naming convenience, not a separate process; leaving
    the reference in place would draw a box with no contents.
    """
    if isinstance(expr, Ref) and expr.base in env.defs:
        d = env.defs[expr.base]
        if not isinstance(d.body, Lambda):
            return _expand(d.body, env)
    if isinstance(expr, App):
        return App(expr.head, tuple(_expand(a, env) for a in expr.args))
    return expr


def build(canonical_doc: str) -> dict:
    doc = parse(canonical_doc)
    env = build_env(doc)

    mats, ops = {}, {}
    for form in doc.forms:
        if isinstance(form, MatDecl):
            for b in form.bindings:
                mats[b.sym] = {"smiles": b.label, "minor": bool(b.minor)}
        elif isinstance(form, UseDecl):
            for b in form.bindings:
                ops[b.sym] = {"label": b.ref, "machine": bool(b.is_machine),
                              "outputs": list(b.outputs), "comm": bool(b.comm)}

    root, name = None, None
    for form in doc.forms:
        if isinstance(form, Def) and not isinstance(form.body, Lambda):
            root, name = form.body, form.name
        elif isinstance(form, (Ref, App)):
            root, name = form, None
    if root is None:
        return {"error": "no process form in this document"}
    root = _expand(root, env)

    # How often does each sub-expression occur? A node seen twice is drawn
    # once and referenced, which is what makes the picture a DAG.
    seen = {}

    def count(e):
        seen[canon_expr(e, env)] = seen.get(canon_expr(e, env), 0) + 1
        if isinstance(e, App):
            for a in e.args:
                count(a)

    count(root)
    emitted = set()

    def node(e):
        key = canon_expr(e, env)
        if key in emitted and seen[key] > 1:
            return {"kind": "ref", "id": key}
        emitted.add(key)
        if isinstance(e, Ref):
            base, idx = e.base, e.index
            if base in mats:
                return {"kind": "mat", "id": key, "sym": base, "index": idx,
                        "smiles": mats[base]["smiles"],
                        "minor": mats[base]["minor"]}
            # A bare symbol that is not a material: a parameter or an unbound
            # reference. Show it rather than dropping it silently.
            return {"kind": "sym", "id": key, "sym": base, "index": idx}
        info = ops.get(e.head.base, {})
        return {"kind": "op", "id": key, "sym": e.head.base,
                "index": e.head.index, "label": info.get("label", ""),
                "machine": info.get("machine", False),
                "shared": seen[key] > 1,
                "args": [node(a) for a in e.args]}

    tree = node(root)
    return {"name": name, "root": tree,
            "n_nodes": len(seen), "n_mentions": sum(seen.values())}


def graph_json(canonical_doc: str) -> str:
    """JSON for the page; errors are returned, never raised, so a drawing
    failure cannot take down a result that parsed and validated."""
    try:
        return json.dumps(build(canonical_doc))
    except Exception as e:  # noqa: BLE001 - the page must keep working
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(graph_json(sys.stdin.read()))
