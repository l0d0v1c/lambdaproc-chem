# λproc

A process description language — recipes, formulations, industrial flowsheets,
chemical synthesis procedures — written as S-expressions, where **unit
operations are functions over streams**, **sub-processes are named lambdas**,
and the **process graph emerges from referential transparency**: two textual
mentions of the same expression denote the same physical node.

**Try** [WebGPU model](https://l0d0v1c.github.io/lambdaproc-chem)

```lisp
(use
  (K  13.01 (crystals liquor))   ; forced-circulation crystalliser
  (FN 4.12  (cake filtrate))     ; Nutsche filter
  (SE 7.02  (dried air)))        ; fluid-bed dryer

(mat (brine "saturated brine"))

(def pure-salt
  (dried (SE60 (cake (FN (crystals (K brine)))))))
```

This repository accompanies the paper *λproc: a compositional language for
process description, with a case study in chemical synthesis* (`../main.tex`).
It contains the language implementation, the training and evaluation code, two
trained checkpoints, and an interactive application.

---

## Quick start

### The language alone — no dependencies

```bash
PYTHONPATH=src python -m lambdaproc check examples/*.lproc --db machines_procedes.json
PYTHONPATH=src python -m lambdaproc canon examples/distillation.lproc
PYTHONPATH=src python -m lambdaproc stats examples/*.lproc
PYTHONPATH=src python -m pytest tests -q
```

### The interactive app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Two tabs. **Synthesise a molecule** asks a 2.7 M-parameter model for procedures
that make a target, and shows what the validator did with every sample — kept,
or rejected with a reason. **Language playground** parses, validates,
canonicalises and hash-conses any document you paste, and needs no model.

### Generation from the command line

```bash
python -m lm.generate models/micro18/best "CC(=O)Nc1ccc(O)cc1" -n 5 -t 0.8
```

```lisp
; 5/5 distinct proposals passed validation
(use (ADD "addition to the reaction medium") (DRY "drying") ...)
(mat (a "Nc1ccc(O)cc1") (b "CC(=O)OC(C)=O") (c "O") (d "CCOC(C)=O") ...)
(def procede (REM (DRY (WAS (EXT (ADD (STI120 (ADD a10000 b100)) c) d) e) f)))
```

4-aminophenol plus acetic anhydride, stirred for two hours, then an aqueous
work-up. That is how paracetamol is made.

---

## What is here

| path | what |
|---|---|
| `src/lambdaproc/` | parser, canonicaliser, validator, DAG builder, CLI |
| `SPEC.md` | language reference |
| `examples/` | 20 cocktails and three industrial processes |
| `tests/` | 19 tests: round trip, canonical form, validation, DAG |
| `machines_procedes.json` | 444 unit operations in 24 categories |
| `lm/` | tokeniser, model, training, evaluation, generation |
| `lm/numpy_model.py` | pure-NumPy forward pass — runs under Pyodide, verified token-for-token against MLX |
| `models/micro18/best/` | 2.73 M params, target → procedure (11 MB) |
| `models/cond/best/` | 2.74 M params, target + inventory → procedure (11 MB) |
| `models/eval/` | every number in the paper, as JSON summaries |
| `scripts/` | patent corpus → λproc converter |
| `app.py` | Streamlit application |

## The models

Decoder-only transformers (RoPE, RMSNorm, SwiGLU, tied embeddings) trained
**from scratch** in MLX on 685 516 procedures derived from US patents, split by
target molecule so that no test molecule is ever seen in training.

| model | params | weights | valid documents | reactant recall |
|---|---|---|---|---|
| `micro18` | 2.73 M | 11 MB | 99.8 % | 0.253 |
| `cond` | 2.74 M | 11 MB | 99.9 % | **0.764** |

`cond` is conditioned on an inventory of available materials — the mode closest
to real use, *"here is my shelf, make this"*. It is also where the model is
most useful: 433 targets in 1000 receive a procedure whose reactants are all
correct. A control measurement in the paper shows it is selecting from the
shelf rather than copying it.

For comparison, a pretrained 350 M-parameter model fine-tuned on the same
corpus in five configurations reaches at best 0.091 reactant recall.

## Running in a browser

`lm/numpy_model.py` is a dependency-free NumPy port of the forward pass,
verified to reproduce the MLX implementation token for token
(`python -m lm.numpy_model models/micro18/best --check`). Because
`lambdaproc` and `lm.tokenizer` are pure standard library, the whole pipeline
— tokenise, generate, rebuild the `(use …)` preamble, parse, validate,
canonicalise — runs unchanged under [Pyodide](https://pyodide.org). Measured
in Pyodide: **0.25 s for one procedure, 3.3 s for ten candidates**, with
weights of 5.2 MiB in float16 (float16 storage leaves greedy output
unchanged).

## Rebuilding everything

```bash
python scripts/run_full_uspto.py --smiles        # 700,149 unique pairs
python -m lm.prepare                             # vocabulary of 391 tokens
python -m lm.prepare --inventory --out data/lm_cond
python -m lm.train --preset micro --epochs 6 --out models/micro
python -m lm.evaluate models/micro/best --n 1000
```

> **One GPU job at a time.** Three concurrent MLX trainings hung the GPU during
> this work and the kernel watchdog rebooted the machine. `lm.train` therefore
> takes an exclusive lock and refuses to start if another training holds it,
> and caps MLX memory (`--mem-gb`, default 32).

## Notes and known rough edges

- **`procede`.** The shipped checkpoints emit the definition name `procede`
  (French), because the corpus was built that way. It is an ordinary symbol,
  not a keyword — renaming it would require retraining, so it was left alone.
- **The machine database is in French.** `machines_procedes.json` comes from
  earlier work and its 444 labels have not been translated. The language, the
  specification, the examples and the operation labels used by the chemistry
  pipeline are in English.
- **The conditioned model wants a realistic shelf.** It was trained on
  inventories holding the true reactants *plus at least as many distractors*,
  so give it four entries or more. With two correct reactants and nothing else
  it is out of distribution and tends to drop one; the app warns you.
- **The conditioned model trusts your shelf.** If the right building block is
  not on it, the model does not say so: it substitutes the closest thing it was
  shown, in a procedure that still validates (paracetamol from a shelf without
  4-aminophenol comes out starting from 4-methoxyaniline). With the reactants
  absent it scores *below* the target-only model. The app therefore runs the
  target-only model as a cross-check and shows what it would have used. Read
  that as a hint, not a verdict: the two disagree on 62 % of adequate shelves
  too. See `lm/README.md`, *Inventory stress tests*.
- **Validity means well-formed, not feasible.** The validator checks the
  structure of a process, not its physics. No procedure in this repository was
  ever run.

## Licence

MIT, see `LICENSE`.

The reaction corpus derives from D. M. Lowe, *Chemical reactions from US
patents (1976–Sep2016)*, figshare, [doi:10.6084/m9.figshare.5104873](https://doi.org/10.6084/m9.figshare.5104873),
released into the public domain (CC0). Trained weights are released under the
same MIT licence as the code.
