# Target → procedure LM

A small decoder-only transformer trained from scratch (MLX, Apple Silicon) that
takes a **target molecule** (SMILES) and writes a **λproc procedure** to
make it: materials with indicative quantities, and the composition of unit
operations.

```
$ python -m lm.generate models/micro/best "CC(=O)Nc1ccc(O)cc1" -n 3 -t 0.8
```

## Pipeline

```bash
# 1. corpus: USPTO grants (CC0) -> (product SMILES, λproc) pairs
python scripts/run_full_uspto.py --smiles          # 700 149 unique pairs
# 2. tokens: split by target molecule (unseen targets in val/test)
python -m lm.prepare                               # data/lm/, vocab 391
# 3. train one model at a time (see "GPU discipline")
python -m lm.train --preset micro --epochs 6 --out models/micro
# 4. evaluate on held-out targets
python -m lm.evaluate models/micro/best --n 1000 --out models/eval/micro.jsonl
```

## Representation

The model never sees the `(use ...)` preamble: it is a pure function of the
operation symbols in the body, dropped by `lm.lmview.to_view` and rebuilt by
`from_view`, so every generated procedure is parsed and validated by
`lambdaproc` before being reported (`lm.lmview.check_view`).

`lm.tokenizer` is lossless on canonical documents (round trip verified on the
whole corpus during `lm.prepare`): SMILES are tokenized with the usual
atom-level regex, lproc structure with parentheses, symbols and single digits.
Vocabulary: **391 tokens**.

## Results (6 epochs, 1000 unseen target molecules, greedy decoding)

| preset | params | val loss | valid lproc | valid SMILES | elements ok | reactant recall | recall = 1 | train time |
|---|---|---|---|---|---|---|---|---|
| nano  | 0.90 M | 0.408 | 99.9 % | 96.6 % | 92.7 % | 0.154 | 2.4 % | 38 min |
| micro | 2.73 M | 0.375 | 99.5 % | 98.4 % | 96.9 % | 0.217 | 5.2 % | 84 min |
| small | 6.53 M | 0.362 | 99.8 % | 98.8 % | 97.4 % | 0.253 | 5.6 % | 160 min |
| micro, 18 epochs | 2.73 M | 0.365 | 99.8 % | 98.4 % | 96.4 % | 0.253 | 6.8 % | 252 min |

- **valid lproc** — the generated text parses, the `use` block rebuilds and
  `lambdaproc` validates it.
- **valid SMILES** — every material label parses with RDKit.
- **elements ok** — every heavy element of the target occurs in some material
  (atoms cannot appear from nothing).
- **reactant recall** — fraction of the real procedure's non-solvent materials
  the model proposed (best over the corpus procedures for that target).
- **recall = 1** — targets for which every reactant of some reference was found.
- `ops_exact` (≤ 1.3 %) and `exact` (0 %) are reported by `lm.evaluate` but are
  near-useless targets: a procedure has many legitimate writings.

Syntax saturates at the smallest size; chemistry keeps improving with scale.
**micro (2.7 M) is the recommended trade-off**: 11 MB of weights, 84 min of
training, 86 % of small's reactant recall for half its compute.

Capacity beats training time: tripling the epochs of micro (84 → 252 min) buys
exactly what going from 2.7 M to 6.5 M parameters buys (recall 0.217 → 0.253),
and costs more wall clock.

## Best-of-n: the validator as a filter

`lm.generate.propose` samples n candidates, always including the greedy one,
keeps those that parse, validate, and pass the chemistry checks, dedups them,
and ranks by mean token log-probability with the greedy candidate preferred on
ties. On 300 unseen targets with micro-18:

| decoding | top-1 recall | oracle recall | surviving candidates | kept |
|---|---|---|---|---|
| greedy | 0.253 | — | 1 | 95.9 % |
| best-of-10, t = 0.8 | 0.249 | **0.458** | 9.6 / 10 | 100 % |

Two readings. Sampling **guarantees** a usable answer — every target gets at
least one procedure that parses and survives RDKit, against 95.9 % for greedy —
and it offers ten distinct routes instead of one. But the model's own
log-probability is a poor judge of chemistry: the best of the ten reaches 0.458
recall while the one it ranks first stays at greedy level. The next gain is a
re-ranker, not a bigger model. (A first attempt at t = 0.9 without the
greedy-first rule *lost* recall, 0.214: sampling must never discard the
deterministic candidate.)

## Inventory conditioning: "here is my shelf, make this"

`lm.prepare --inventory` prefixes the procedure with the materials at hand: the
reference's non-solvent reactants, each kept with probability 0.8, plus at least
as many distractors drawn from the 20 000 most frequent corpus reactants, sorted
so the order carries nothing, and with the target itself excluded. Solvents are
assumed always available. Same micro preset, 6 epochs, 127 min:

| | reactant recall | recall = 1 | precision | kept |
|---|---|---|---|---|
| no inventory (micro, 18 ep) | 0.253 | 6.8 % | — | 95.9 % |
| **with inventory** | **0.764** | **45.0 %** | **0.785** | 91.4 % |
| control: use the whole shelf | 0.756 | — | 0.417 | — |

The control line is the point. A model that simply dumped every shelf item into
the procedure would reach 0.756 recall; this one reaches 0.764 **with nearly
twice the precision**, and it uses only 3.4 % of the distractors. It is
selecting, not copying.

Its limits were measured too. Of the reactants deliberately withheld from the
shelf it recovers **17.4 %** from its own knowledge; of the materials it brings
in from off the shelf only **21.4 %** are right. Strong at choosing and ordering
what it is given, weak at inventing what is missing. Validity also dips
(91.4 % vs 95.9 %): constrained to the shelf, it sometimes writes a procedure in
which an element of the target appears nowhere.

Validation losses across sections are **not** comparable — different datasets
and prompt lengths. Only the chemistry metrics are.

### Leakage check

Spotting a target among its own materials prompted a check: the target occurs as
a material in 74 / 6927 test references (1.1 %), and excluding those leaves
small's recall unchanged at 0.250. `lm.prepare` now excludes the target from
every shelf.

## Pretrained baseline: LFM2-350M

Does fine-tuning a small pretrained LLM beat a specialist trained from scratch?
`lm.baseline_lora` exports the same product-hash split in prompt/completion form
and evaluates with the same validator and chemistry checks.

```bash
python -m lm.baseline_lora export --n-train 0 --out data/lfm2_all   # 686 254 pairs
mlx_lm.lora --model LiquidAI/LFM2-350M --train --data data/lfm2_all \
    --fine-tune-type lora --num-layers -1 --mask-prompt --batch-size 16 \
    --iters 20000 --learning-rate 1e-4 --max-seq-length 640 \
    --adapter-path models/lfm2_lora_all
python -m lm.baseline_lora eval --adapter models/lfm2_lora_all --n 300
```

Five configurations were tried: LoRA at two data pool sizes and a full
fine-tune at two learning rates. Same 300 held-out targets, greedy decoding.

| configuration | trainable | examples seen | valid lproc | elements ok | reactant recall | copies the target |
|---|---|---|---|---|---|---|
| LoRA 1e-4, 100 k pool | 3.0 M / 354 M | 200 k | 98.7 % | 90.7 % | 0.090 | 25.0 % |
| LoRA 1e-4, 100 k pool | 3.0 M | 320 k | 98.3 % | 89.7 % | 0.082 | 28.3 % |
| LoRA 1e-4, full pool | 3.0 M | 240 k | 97.3 % | 88.0 % | 0.091 | 14.0 % |
| LoRA 1e-4, full pool | 3.0 M | 320 k | 98.0 % | 89.0 % | 0.085 | 17.3 % |
| full FT 1e-5, 100 k pool | 287 M | 128 k | 96.7 % | 85.3 % | 0.044 | 69.0 % |
| full FT 5e-5, full pool | 287 M | 64 k | 79.0 % | 71.3 % | 0.047 | 64.3 % |
| full FT 5e-5, full pool | 287 M | 128 k | 99.3 % | 90.0 % | 0.061 | 53.7 % |
| **micro from scratch, 18 ep** | **2.7 M / 2.7 M** | 12.3 M | **99.7 %** | **96.0 %** | **0.245** | **4.7 %** |
| small from scratch, 6 ep | 6.5 M | 4.1 M | 100 % | 95.7 % | 0.245 | 2.3 % |

**The specialist wins by ~3×, and no LFM2 configuration comes close.** Every
one lands between 0.044 and 0.091 reactant recall against 0.245. Pretrained
English is not useful prior knowledge here: the input is a SMILES string and
the output a formal language, neither resembling the pretraining distribution.

Two candidate explanations were tested and rejected:

- *Not enough distinct data.* The first export used `--n-train 100000`, giving
  LFM2 88 426 distinct targets against 685 516 for the from-scratch models —
  a sevenfold difference. Re-running the identical recipe on the full pool
  moved recall from 0.082 to 0.085. It did halve the copy shortcut (28 % →
  17 %), which makes sense — more distinct targets make "copy what you were
  shown" less rewarding — but it bought no chemistry.
- *LoRA is too small an adapter.* A full fine-tune of 287 M parameters reaches
  0.061, below LoRA's 0.091. It is still improving at the budget tested, but
  equalling micro would take a fourfold jump that nothing in the curves
  suggests.

The adapter's dominant failure is a surface shortcut — *material `a` is the
SMILES I was just shown*. It is not pure invention: **4.79 %** of training
examples legitimately list the target as a material (salt formation, where the
free base is a reactant and the salt the product). micro reproduces that base
rate almost exactly, 4.3 %; LFM2 over-applies it three- to fifteenfold
depending on configuration, and the under-trained full fine-tunes do little
else (up to 69 %).

No leakage: the 88 426 (resp. 686 254) LFM2 training targets share nothing with
the 441 validation or 6 001 test targets.

Caveat: the comparison runs on the alphabetically first 300 test targets, a
lighter-than-average slice (see the sampling note below). It is the same slice
for every model.

### Two traps this experiment fell into

**mlx-lm applies the chat template.** `CompletionsDataset.process` wraps every
prompt/completion pair in the model's user/assistant turns, whatever the docs
suggest about base models. Evaluating with a plain-text prompt is therefore
off-distribution, and gets *worse the longer you train*: the probability of the
correct first token `(` fell from 0.344 at 2 500 iterations to outside the top 5
at 7 500, while the model emitted documents starting mid-expression
(`PUR (REM (HEA ...`). Validity read 83 % → 9 % across the run. Apply the
template at inference (`chat_prompt`) and the same adapters read 99 %.

**Teacher-forced validation loss is blind to this.** Across the very run whose
free generation collapsed, val loss improved monotonically, 0.695 → 0.629. The
first completion token is one position in ~150: invisible in the average, and
decisive in generation, because once the first token is wrong the model is off
its own distribution for the rest of the document. Never conclude from the loss
curve alone — generate and validate.

## Representation ablation

Three views of the same corpus, one dataset each, same architecture:

```bash
python -m lm.prepare --view nested --out data/lm        # the language itself
python -m lm.prepare --view flat   --out data/lm_flat   # numbered steps, no nesting
python -m lm.prepare --view mats   --out data/lm_mats   # material list only

python -m lm.train --data data/lm_flat --preset micro --epochs 6 --out models/micro_flat
python -m lm.ablation models/micro_flat/best --view flat --data data/lm_flat \
    --n 1000 --out models/eval/abl_flat.jsonl
```

`lm.ablation` always scores against `data/lm/test.jsonl`, so the three models
answer the same questions against the same ground truth, on material-level
metrics that no view privileges.

Two traps. **Validation loss is not comparable across views** — the `mats`
view reaches 0.189 against the nested 0.375 purely because its sequences are
shorter and easier, not because it is a better model. And **the flat view has
no parser**, so `valid_lproc` is structurally unavailable there rather than
zero; `lm.ablation` reports it only for the nested view.

Result: flattening is lossless on this corpus (0 of 200 000 documents fail to
round-trip, mean 9.03 steps) and costs nothing measurable in chemistry. All
nine paired comparisons between views have confidence intervals containing
zero.

## Paired comparisons

Mean metrics over 1 000 targets hide differences of a few points under a
per-target standard deviation an order of magnitude larger. `lm.paired`
compares two saved evaluations target by target:

```bash
python -m lm.paired models/eval/micro18.jsonl models/eval/retrieval.jsonl \
    --key bb_recall
```

It reports the paired mean difference with a bootstrap 95 % interval over
resampled *targets* (both systems resampled together) and a two-sided Wilcoxon
signed-rank p-value. Use it before claiming any improvement: it is what
demoted "the model beats retrieval" on the old metric to p = 0.23.

## Is the oracle gap reachable? (`lm.rerank`)

`lm.evaluate --best-of N` stores every scored survivor under `candidates`, so a
re-ranking study needs no new generation:

```bash
python -m lm.evaluate models/micro18/best --n 300 --best-of 10 \
    --out models/eval/micro18_bo10_full.jsonl
python -m lm.rerank models/eval/micro18_bo10_full.jsonl
```

Measured on micro18, 9.2 survivors per target: greedy 0.260, oracle 0.422, and
the best reference-free selection rule 0.265 — the model's own mean logprob,
beating greedy by 0.005 and leaving 0.157 of the gap unclaimed.

Three traps this measures, worth knowing before building a re-ranker:

- **Tanimoto similarity between the proposed building blocks and the target
  carries no signal** (within-target rho +0.047 ± 0.075). The feature
  retrosynthetic locality suggests does not discriminate among candidates that
  already passed the validator.
- **Correlation and argmax come apart.** `sum_sim` correlates significantly
  (+0.163) yet selecting by it scores 0.146, *below* random, because it is
  confounded with the number of materials and its argmax picks the longest
  list. Judge a re-ranker by the selection it makes, not by its correlation.
- **Only 1.91 distinct quality values per target.** The oracle gap is not ten
  gradations to sort but a near-binary question, so the apparent headroom
  overstates the learnable structure.

Report `within_target_rho`, never a pooled correlation: a feature can track
target difficulty across targets while being useless within one.

## Inventory stress tests (`lm.shelf_stress`)

The conditioned model is trained and, by default, evaluated on shelves that
hold ~80 % of the true reactants plus random, chemically unrelated
distractors. That flatters it: **copying the whole shelf scores 0.760 bb
recall, more than the model's 0.738.** Its real contribution is precision
(0.753 against 0.320). Evaluate it on shelves it never saw:

```bash
python -m lm.shelf_stress build          # data/lm_cond_{trained,hard,absent,absent_hard,large}
for v in hard absent absent_hard large; do
  python -m lm.evaluate models/cond/best --data data/lm_cond_$v \
      --out models/eval/cond_$v.jsonl
done
cp models/eval/cond.jsonl models/eval/cond_trained.jsonl
python -m lm.shelf_stress score          # model vs copy-shelf vs top-m Tanimoto
```

| shelf | bb recall | bb precision |
|---|---|---|
| training-like | 0.738 | 0.753 |
| large (10 items) | 0.671 | 0.503 |
| hard (near-miss distractors) | 0.535 | 0.549 |
| reactants absent | 0.072 | 0.065 |
| absent + near misses | **0.019** | 0.018 |
| *no shelf at all* | *0.266* | — |

**An insufficient shelf makes the conditioned model worse than no shelf**
(−0.205, p = 2e-47), and with near-miss analogues on offer it substitutes them
inside documents that still validate. The validator cannot see this. Cause: a
shelf missing every true reactant occurs with probability 0.2^n in training,
so the model never learned to fall back on its prior. Until it is retrained on
insufficient and near-miss shelves, use the conditioned model only when the
inventory is known to be adequate, and compare its answer with the
unconditioned model's.

Trap: a 16-item shelf overflows the 512-token training context (median prompt
395 tokens), confounding shelf size with position extrapolation. `large` uses
10 items (median 253).

## GPU and memory discipline

Three concurrent MLX trainings hung the GPU on 2026-11-10 and the kernel
watchdog rebooted the Mac. `lm.train` therefore takes an exclusive lock
(`models/.train.lock`) and caps MLX memory (`--mem-gb`, default 32). Run
sweeps sequentially, ideally under `taskpolicy -c utility`. `data/` and
`models/` carry `.metadata_never_index` (Spotlight indexing of the extracted
corpus was itself a heavy load).

**Cap the reuse cache, not only the memory limit.** MLX retains freed buffers
for reuse and that cache is *unbounded by default*. During batched generation
at `--bs 250` it reached **196 GB of retained allocations against 2 GB of live
ones** — a memory-pressure alert on a 256 GB machine, with nothing visible in
`get_peak_memory()`, which reports only live memory. `set_memory_limit()` does
not bound it; `set_cache_limit()` does. Every entry point that generates now
sets both (`--mem-gb`, default 24, cache capped at a quarter of it), which
holds the cache at 6 GB with peak and outputs unchanged. If you add a script
that calls `lm.generate`, set both limits in it.
