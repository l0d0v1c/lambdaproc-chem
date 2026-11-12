#!/usr/bin/env python3
"""λproc explorer — an interactive demo of the language and the model.

    pip install -r requirements.txt
    streamlit run app.py

Two independent halves:

* **Synthesise** needs MLX and RDKit. It asks a small from-scratch model for
  procedures that make a target molecule, and shows what the validator did
  with each sample — kept, or rejected with a reason.
* **Language** needs nothing but the standard library. It parses, validates,
  canonicalises and hash-conses any λproc document you paste.
"""

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from lambdaproc import canonical, load_machines, parse, to_dag, validate  # noqa: E402

MODELS = {
    "micro-18 — target only (2.73 M params)": ("models/micro18/best", False),
    "cond — target + inventory (2.74 M params)": ("models/cond/best", True),
}

# The conditioned model was trained on shelves holding the true reactants
# plus at least as many distractors: a two-item shelf is out of distribution.
DEFAULT_SHELF = "\n".join([
    "Nc1ccc(O)cc1",      # 4-aminophenol   (a real precursor of paracetamol)
    "CC(=O)OC(C)=O",     # acetic anhydride (the other one)
    "CC(=O)Cl",          # acetyl chloride  (distractor)
    "CCN(CC)CC",         # triethylamine    (distractor)
    "c1ccncc1",          # pyridine         (distractor)
    "CO",                # methanol         (distractor)
])

EXAMPLE_TARGETS = {
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "benzocaine": "CCOC(=O)c1ccc(N)cc1",
}

st.set_page_config(page_title="λproc explorer", page_icon="⚗️",
                   layout="wide")


# --------------------------------------------------------------- utilities
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load an MLX checkpoint and its vocabulary (cached across reruns)."""
    import mlx.core as mx

    from lm.model import load
    from lm.tokenizer import Tokenizer

    mx.set_memory_limit(8 * 2**30)
    model = load(os.path.join(ROOT, path))
    tok = Tokenizer.load(os.path.join(ROOT, path, "vocab.json"))
    return model, tok


def cross_check(target: str, conditioned_doc: str, shelf: list) -> None:
    """Compare the conditioned proposal with the unconditioned model's.

    Measured on 874 test targets (paper, inventory stress tests): when the two
    share a building block, bb recall is 0.839, against 0.680 when they do
    not. But disagreement also occurs on 62 % of *adequate* shelves, so it is a
    confidence hint, never a verdict -- and it is presented as one.
    """
    from lm.chem import building_blocks_of
    from lm.generate import generate
    from lm.lmview import check_view

    model, tok = load_model(MODELS["micro-18 — target only (2.73 M params)"][0])
    out = generate(model, [tok.encode_prompt(target)])[0]
    ok, res = check_view(tok.decode_view(out))
    free = building_blocks_of(res) if ok else set()
    cond = building_blocks_of(conditioned_doc)
    shared = cond & free
    if shared:
        st.success(
            f"Cross-check: the target-only model, which never saw your shelf, "
            f"also proposes {len(shared)} of these building blocks. On test "
            "data, agreement goes with higher accuracy (0.84 vs 0.68 building-"
            "block recall).")
    else:
        off = sorted(free - set(shelf))
        st.info(
            "Cross-check: the target-only model proposes different building "
            "blocks. This is common even with a good shelf (62 % of test cases) "
            "and is not proof of a problem. But if your shelf might lack the "
            "key building block, know that this model does not detect it: it "
            "substitutes the closest thing on the shelf, in a procedure that "
            "still validates."
            + (f" Without your shelf it would have used: "
               + ", ".join(f"`{x}`" for x in off[:4]) if off else ""))


@st.cache_resource(show_spinner=False)
def machines():
    try:
        return load_machines(os.path.join(ROOT, "machines_procedes.json"))
    except Exception:
        return None


def mol_svg(smiles: str, w: int = 230, h: int = 150):
    """Render one SMILES as an SVG string, or None if RDKit cannot parse it."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def dag_stats(doc):
    nodes, tree = to_dag(doc)
    return len(nodes), tree


def doc_report(text: str):
    """Parse / validate / canonicalise a document. Returns a dict."""
    out = {"ok": False, "errors": [], "canonical": None, "nodes": None,
           "tree": None}
    try:
        doc = parse(text)
    except Exception as e:
        out["errors"] = [f"{type(e).__name__}: {e}"]
        return out
    v = validate(doc, machines())
    out["errors"] = list(v.errors)
    out["ok"] = not v.errors
    try:
        out["canonical"] = canonical(doc)
        out["nodes"], out["tree"] = dag_stats(doc)
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["ok"] = False
    return out


# ------------------------------------------------------------------ layout
st.title("⚗️ λproc explorer")
st.caption(
    "A process is a composition of functions over streams. The graph is not "
    "drawn — it emerges from referential transparency."
)

tab_gen, tab_lang, tab_about = st.tabs(
    ["Synthesise a molecule", "Language playground", "About"])


# ============================================================ synthesise ===
with tab_gen:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.subheader("Target")
        choice = st.selectbox("Model", list(MODELS))
        model_path, wants_inventory = MODELS[choice]

        preset = st.selectbox("Load an example", ["(type my own)"]
                              + list(EXAMPLE_TARGETS))
        default = EXAMPLE_TARGETS.get(preset, "CC(=O)Nc1ccc(O)cc1")
        target = st.text_input("Target molecule (SMILES)", value=default)

        inventory_text = ""
        if wants_inventory:
            st.markdown("**What you have on the shelf**")
            st.caption(
                "One SMILES per line. Solvents are assumed always available "
                "and need not be listed. This model was trained to **pick** "
                "from a shelf it cannot assume complete, so give it a real "
                "shelf — several candidates, not only the right ones. With "
                "fewer than four entries it is out of its training "
                "distribution and tends to drop reactants."
            )
            inventory_text = st.text_area(
                "Available materials", height=150,
                value=DEFAULT_SHELF, label_visibility="collapsed")

        n = st.slider("Samples", 1, 30, 10)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
        top_k = st.slider("Top-k", 0, 100, 20)
        seed = st.number_input("Seed", value=0, step=1)
        show_rejected = st.checkbox("Show rejected samples", value=False)
        go = st.button("Generate", type="primary", use_container_width=True)

        svg = mol_svg(target, 300, 200)
        if svg:
            st.image(svg)

    with right:
        if not go:
            st.info(
                "Pick a target and press **Generate**. Every sample is decoded, "
                "its `(use …)` preamble rebuilt, then parsed and validated by "
                "λproc before you see it — nothing is filtered silently."
            )
        else:
            try:
                from lm.chem import reactants_of
                from lm.generate import canon_target, propose
            except ImportError as e:
                st.error(f"MLX or RDKit missing — generation unavailable ({e}). "
                         "The **Language playground** tab works regardless.")
                st.stop()

            try:
                t_canon = canon_target(target)
            except Exception as e:
                st.error(f"RDKit cannot parse that target: {e}")
                st.stop()

            inv = None
            if wants_inventory:
                inv = []
                for line in inventory_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        inv.append(canon_target(line))
                    except Exception:
                        st.warning(f"Ignored unparseable inventory SMILES: {line}")
                inv = inv or None
                if inv and len(inv) < 4:
                    st.warning(
                        f"Only {len(inv)} material(s) on the shelf. This model "
                        "was trained on shelves of four or more and tends to "
                        "drop reactants below that — add a few distractors."
                    )

            with st.spinner(f"Sampling {n} procedures…"):
                model, tok = load_model(model_path)
                cands = propose(model, tok, t_canon, n=int(n),
                                temperature=float(temperature),
                                top_k=int(top_k), seed=int(seed),
                                inventory=inv)

            kept = [c for c in cands if c["kept"]]
            if inv and kept:
                cross_check(t_canon, kept[0]["canonical"], inv)
            c1, c2, c3 = st.columns(3)
            c1.metric("Distinct proposals", len(cands))
            c2.metric("Passed validation", len(kept))
            c3.metric("Canonical target", "ok")
            st.caption(f"Target as RDKit canonical SMILES: `{t_canon}`")

            if not kept:
                st.warning("No sample survived validation. Try a higher "
                           "temperature, more samples, or another target.")

            for i, c in enumerate(kept):
                label = (f"Proposal {i + 1} — score {c['score']}"
                         + ("  ·  greedy" if c.get("greedy") else "")
                         + (f"  ·  sampled {c['n_samples']}×"
                            if c["n_samples"] > 1 else ""))
                with st.expander(label, expanded=(i == 0)):
                    st.code(c["canonical"], language="lisp")
                    rep = doc_report(c["canonical"])
                    a, b, d = st.columns(3)
                    a.metric("DAG nodes", rep["nodes"] if rep["nodes"] else "—")
                    b.metric("Tree size", rep["tree"] if rep["tree"] else "—")
                    d.metric("Sharing",
                             f"{rep['tree'] / rep['nodes']:.2f}×"
                             if rep["nodes"] else "—")
                    mats = sorted(reactants_of(c["canonical"]))
                    if mats:
                        st.markdown("**Non-solvent materials**")
                        cols = st.columns(min(4, len(mats)))
                        for j, smi in enumerate(mats):
                            with cols[j % len(cols)]:
                                s = mol_svg(smi)
                                if s:
                                    st.image(s)
                                st.caption(f"`{smi}`")
                                if inv and smi in inv:
                                    st.caption("✅ on your shelf")

            if show_rejected:
                for c in (c for c in cands if not c["kept"]):
                    with st.expander(f"❌ rejected — {c['reason']}"):
                        st.code(c["canonical"] or c["text"], language="lisp")


# =============================================================== language ===
with tab_lang:
    st.subheader("Parse, validate, canonicalise, hash-cons")
    st.caption(
        "Pure standard library — no model involved. The DAG is obtained by "
        "interning every sub-expression: two textual mentions of the same "
        "expression are one physical node."
    )

    ex_dir = os.path.join(ROOT, "examples")
    files = sorted(f for f in os.listdir(ex_dir) if f.endswith(".lproc")) \
        if os.path.isdir(ex_dir) else []
    pick = st.selectbox("Load an example", ["(empty)"] + files)
    initial = ""
    if pick != "(empty)":
        with open(os.path.join(ex_dir, pick), encoding="utf-8") as fh:
            initial = fh.read()

    text = st.text_area("λproc document", value=initial, height=320,
                        key=f"src_{pick}")

    if text.strip():
        rep = doc_report(text)
        if rep["ok"]:
            st.success("Valid λproc document.")
        else:
            st.error("Invalid:")
            for e in rep["errors"][:10]:
                st.markdown(f"- `{e}`")

        if rep["canonical"]:
            a, b, c = st.columns(3)
            a.metric("DAG nodes", rep["nodes"])
            b.metric("Tree size (occurrences)", rep["tree"])
            c.metric("Sharing factor",
                     f"{rep['tree'] / rep['nodes']:.2f}×" if rep["nodes"] else "—")
            st.markdown("**Canonical form** — one process, one string")
            st.code(rep["canonical"], language="lisp")
            st.caption(
                "Idempotent: canon(parse(canon(d))) = canon(d). Commutative "
                "operations have their arguments sorted, so `(MIX a b)` and "
                "`(MIX b a)` are the same process."
            )


# ================================================================== about ===
with tab_about:
    st.markdown("""
### λproc

A process — a recipe, a formulation, an industrial flowsheet, a synthesis
procedure — is written as a **composition of functions over streams**:

* a **unit operation** is a function from streams to streams, bound either to
  an entry of a 444-machine database or to a free-text definition;
* a **sub-process** is a named lambda, the unit in which knowledge accumulates;
* the **process graph** is not drawn and not declared: two textual mentions of
  the same expression denote the same physical node.

Three properties follow, and they are what makes the language useful to a
machine-learning pipeline:

| property | what it buys |
|---|---|
| canonical form | one process, one string — exact deduplication, deterministic learning target |
| decidable validity | a filter independent of the generator, usable inside the sampling loop |
| derivable preamble | the `(use …)` block is removed from the learning problem entirely |

### The models in this demo

Decoder-only transformers trained **from scratch** on 685 516 procedures
derived from US patents, split by target molecule so that test molecules are
never seen in training.

| model | params | weights | valid documents | reactant recall |
|---|---|---|---|---|
| micro-18 | 2.73 M | 11 MB | 99.8 % | 0.253 |
| cond (inventory) | 2.74 M | 11 MB | 99.9 % | **0.764** |

For comparison, a pretrained 350 M-parameter model fine-tuned on the same
corpus in five configurations reaches at best 0.091.

### What this is not

The model agrees with what a chemist did in a patent; it does **not** prove a
procedure works. No procedure was run and no feasibility was computed.
Validity here means *well-formed*, not *feasible*.

---
MIT licence. Corpus derived from D. M. Lowe, *Chemical reactions from US
patents (1976–Sep2016)*, figshare, CC0.
""")
