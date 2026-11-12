"""Glue between the page and the published Python code.

Everything here is orchestration. The tokeniser, the model, the validator and
the canonicaliser are the modules copied verbatim from the repository, so a
procedure accepted in the browser is accepted by exactly the code that
produced the paper's numbers.

The JS side calls `propose(target, n, temperature, top_k, seed, inventory)`
and gets back JSON.
"""

import json

import numpy as np

from procgraph import graph_json
from lm.lmview import check_view
from lm.numpy_model import NumpyGPT
from lm.tokenizer import Tokenizer

_models = {}


def load(name, cfg_json, vocab_json, weights_bytes):
    """Called once per model from JS, which has already fetched the files."""
    import io
    import os
    import tempfile

    # NumpyGPT.load reads from a directory; we are handed bytes. Write them to
    # Pyodide's in-memory filesystem rather than duplicating the reader.
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as fh:
        fh.write(cfg_json)
    with open(os.path.join(d, "vocab.json"), "w") as fh:
        fh.write(vocab_json)
    with open(os.path.join(d, "weights.safetensors"), "wb") as fh:
        fh.write(bytes(weights_bytes))
    model = NumpyGPT.load(d)
    # float16 weights: cast once, here, rather than in every matmul.
    model.w = {k: v.astype(np.float32) for k, v in model.w.items()}
    _models[name] = (model, Tokenizer.load(os.path.join(d, "vocab.json")))
    return json.dumps({"ok": True, "vocab": len(_models[name][1])})


def propose(name, target, n=10, temperature=0.8, top_k=20, seed=0,
            inventory=None, max_new=480):
    """Sample n procedures, validate each, return them best-first.

    Mirrors `lm.generate.propose`: the greedy candidate always competes, since
    ranking by likelihood alone was measured to be worse than greedy decoding
    (paper, sec:bestofn).
    """
    model, tok = _models[name]
    inv = list(inventory) if inventory else None
    prompt = tok.encode_prompt(target, inv)

    outs, scores = [], []
    g, gs = model.generate([prompt], max_new=max_new, temperature=0.0,
                           return_logprobs=True)
    outs, scores, greedy = list(g), list(gs), [True]
    if n > 1:
        rng = np.random.default_rng(seed)
        s, ss = model.generate([prompt] * (n - 1), max_new=max_new,
                               temperature=temperature, top_k=top_k, rng=rng,
                               return_logprobs=True)
        outs += list(s)
        scores += list(ss)
        greedy += [False] * (n - 1)

    seen, cands = {}, []
    for ids, score, is_greedy in zip(outs, scores, greedy):
        view = tok.decode_view(ids)
        ok, res = check_view(view)
        key = res if ok else view
        if key in seen:
            seen[key]["n_samples"] += 1
            seen[key]["greedy"] = seen[key]["greedy"] or is_greedy
            continue
        c = {"canonical": res if ok else None, "valid": bool(ok),
             "reason": None if ok else res, "text": view,
             "score": round(float(score), 4), "greedy": bool(is_greedy),
             "n_samples": 1}
        seen[key] = c
        cands.append(c)
    # greedy first, then by likelihood
    cands.sort(key=lambda c: (not c["greedy"], -c["score"]))
    return json.dumps(cands)


def graph(canonical_doc):
    """Drawable flowsheet for a validated document (see procgraph)."""
    return graph_json(canonical_doc)


def materials(canonical_doc):
    """Non-solvent-agnostic material list: every (mat ...) binding.

    The browser has no RDKit on the Python side, so solvent and reagent
    classification -- which needs canonical SMILES -- is left to the JS side
    via RDKit MinimalLib.
    """
    import re
    out = []
    for m in re.finditer(r'\(([a-z][\w-]*)\s+"([^"]*)"\)', canonical_doc):
        out.append({"symbol": m.group(1), "smiles": m.group(2)})
    return json.dumps(out)
