"""LM view of a LambdaProc document: canonical form without the `(use ...)` block.

The use block of a USPTO-derived document is a pure function of the operation
symbols appearing in the body (fixed table ACTION_OPS), so the model never has
to learn it: `to_view` drops it, `from_view` rebuilds it, and the result is
parsed and validated by lambdaproc like any hand-written document.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lambdaproc import canonical, parse, validate  # noqa: E402
from uspto_to_lproc import ACTION_OPS  # noqa: E402

OP_LABELS = {sym: label for sym, label in ACTION_OPS.values()}
_OP_RE = re.compile(r"\(([A-Z]{3})\d*[\s)]")


def to_view(canon: str) -> str:
    lines = canon.strip().split("\n")
    if not lines[0].startswith("(use "):
        raise ValueError("expected a leading (use ...) form")
    return "\n".join(lines[1:])


def from_view(view: str) -> str:
    """Rebuild a full LambdaProc document (raises on unknown op symbols)."""
    ops = sorted(set(_OP_RE.findall(re.sub(r'"[^"]*"', '""', view))))
    unknown = [o for o in ops if o not in OP_LABELS]
    if unknown:
        raise ValueError(f"unknown operation symbols: {unknown}")
    use = "(use " + " ".join(f'({o} "{OP_LABELS[o]}")' for o in ops) + ")"
    return use + "\n" + view.strip() + "\n"


def check_view(view: str):
    """Return (ok, canonical_or_error) for a generated view."""
    try:
        doc = parse(from_view(view))
        v = validate(doc)
        if v.errors:
            return False, "; ".join(v.errors[:3])
        return True, canonical(doc)
    except Exception as e:  # parse errors, unknown ops...
        return False, f"{type(e).__name__}: {e}"
