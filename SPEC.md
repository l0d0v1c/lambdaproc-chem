# λproc — specification v0.1

λproc is a process description language: recipes, formulations, industrial
flowsheets. A process is written as a **composition of functions**: each unit
operation is a function `stream → stream`, sub-processes are named lambdas, and
the process graph (a DAG) emerges from **referential transparency** — two
mentions of the same expression denote the same physical node.

Inspirations: SMILES (molecular chemistry), FSL
([SymbolicFormulation](https://github.com/l0d0v1c/SymbolicFormulation),
Brunet 2021, doi:10.17601/rdmediation.2021.2) for composition, SFILES 2.0
(Vogel, Hirtreiter, Schulze Balhorn & Schweidtmann, *Optimization and
Engineering* 24:2911–2933, 2023) for topology, LISP and the lambda calculus for
structure.

## 1. Model

- **Stream**: matter in circulation (solid, liquid, gas, mixture). Every
  λproc expression denotes a stream, or a tuple of streams.
- **Unit operation**: a function taking one or more streams and returning one
  or more streams. Operations are bound, in a preamble, either to a machine of
  the `machines_procedes.json` database (444 machines, 24 categories, described
  as input → output transfer functions) or to a free-text definition.
- **Sub-process**: a lambda named by `def`, reusable — the unit in which
  knowledge is capitalised and the vocabulary compressed.
- **Document**: preamble (`use`, `mat`) + definitions (`def`) + expressions.

## 2. Syntax

### 2.1 Lexicon

- **Comments**: `;` to end of line.
- **Identifiers**: begin with a letter (Latin or Greek, Unicode), continue with
  letters, digits or `-`. Examples: `M`, `EC`, `lime`, `pure-salt`, `η`.
- **Index**: the trailing digits of a symbol are always its index: `EC78` is
  symbol `EC`, index 78; `rum-white4` is `rum-white`, index 4. Consequence: a
  declared name (`use`, `mat`, `def`, lambda parameter, output name) **may not
  end in a digit**.
- **Machine id**: `N.MM` (e.g. `5.02`), a direct reference to the database.
- **Strings**: `"..."`, only in the preamble (labels, free definitions).
- **Reserved words**: `use`, `mat`, `def`, `lambda`, flags `:minor`, `:comm`.

### 2.2 Grammar (EBNF)

```ebnf
document  = { form } ;
form      = usedecl | matdecl | defform | expr ;

usedecl   = "(" "use" { binding } ")" ;
binding   = "(" SYM ref [ outputs ] [ ":comm" ] ")" ;
ref       = MACHINE_ID | STRING ;
outputs   = "(" SYM { SYM } ")" ;

matdecl   = "(" "mat" { matbind } ")" ;
matbind   = "(" SYM STRING [ ":minor" ] ")" ;

defform   = "(" "def" NAME ( lambda | expr ) ")" ;
lambda    = "(" "lambda" "(" { NAME } ")" expr ")" ;

expr      = atom | "(" head { expr } ")" ;
head      = SYMIDX | MACHINE_ID ;        (* operation, selector or macro *)
atom      = SYMIDX ;                     (* material, parameter or 0-ary macro *)
SYMIDX    = SYM [ DIGITS ] ;
```

### 2.3 Semantics

- **Operation application** `(OP arg…)`: the arguments are the incoming
  streams; the result is the tuple of declared outputs (a single stream when no
  output is declared).
- **Multiple outputs and selectors**: output names declared in `use` become
  unary selectors. An operation with several outputs may be consumed **only
  through** one of them: `(crystals (K brine))`. Applying a selector to
  anything other than the operation declaring it is an error.
- **Sharing (DAG)**: two textual occurrences of the same canonical expression
  denote the same physical machine. `(distillate (D x))` and
  `(residue (D x))` consume the two outputs of a single column.
- **Indices**: purely **indicative**. On a material: a quantity (unit depends
  on context — cl, parts, kg…). On an operation: the principal parameter
  (temperature, duration…). The exact semantics depends on the equipment and
  the material; the language imposes no unit. Indices are forbidden on
  selectors, macros and parameters.
- **Macros**: `(def name expr)` creates a nullary macro usable as an atom;
  `(def name (lambda (p…) expr))` creates a parameterised sub-process, called
  as `(name arg…)` with exact arity. No recursion and no higher order in v0.1:
  λproc describes, it does not compute.
- **`:minor`**: an additive material (a colourant, a dash of bitters),
  inherited from FSL's minor-component concept; informative in v0.1.
- **`:comm`**: a commutative operation (the order of incoming streams is
  indifferent). Operations bound to categories 3 (Mixing/Agitation) and 11
  (Homogenisation) of the database are commutative by default.

## 3. Canonical form

Guarantees *one process, one string* (indispensable for training models):

1. one head form per line, single spaces, no comments;
2. `use` and `mat` bindings sorted by symbol;
3. arguments of commutative operations sorted by canonical form;
4. idempotence: `canon(parse(canon(d))) = canon(d)`.

## 4. Validation

The validator checks: all symbols declared; machine ids present in the
database; symbols unique (output names share the same namespace); selectors
applied to the operation that declares them; multi-output operations consumed
through a selector; arity of macro calls; indices forbidden where the semantics
does not define them.

```bash
python -m lambdaproc check examples/*.lproc --db machines_procedes.json
python -m lambdaproc canon examples/distillation.lproc
python -m lambdaproc stats examples/*.lproc
```

## 5. Limits of v0.1 (candidates for v0.2)

- **Recycle**: loops (fixed point, `letrec`) are not representable.
- **Control**: no control-and-command structure (cf. SFILES 2.0).
- **Rich operating conditions**: a single index per symbol.
- **Evaluation**: no material balance; indices are not conserved quantities.
- **Export**: no SFILES 2.0 conversion yet, and no SVG rendering (the database
  carries one pictogram per machine).

### 5.1 Sketched designs for the first two

The paper's Appendix D works these out in full (grammar deltas, validation
rules, costs). In brief, and so that nobody implements a different shape by
accident:

**Recycle** — one new binding form, `(letrec ((name expr) ...) expr)`, with
mutually recursive stream names. Not sugar for `def`, whose dependency relation
must stay a strict partial order so that inlining terminates. Two new
validation rules: *productivity*, every cycle passes through at least one
operation declared in `use` (rejecting `(letrec ((s (top s))) s)`, a stream
that is a selection of itself), and a warning when a cycle has no purge.

What it costs: the denotation stops being a DAG, so canonicalisation stops
being structural recursion and becomes graph canonical labelling. "One process,
one string" weakens to "one string up to graph isomorphism". The repair is the
one SMILES made for rings — Morgan-style iterative refinement of node
invariants, ties broken exhaustively — which is cheap on labelled graphs of
tens of nodes and worst-case exponential in the symmetry group. **Fail loudly
if the exhaustive step is bounded out; never return a non-canonical string**,
which would silently corrupt deduplication.

Do not train on the `letrec` surface form: it resolves names non-locally, which
is what small autoregressive models are bad at. Train on the indexed-step flat
view (`lm/flatview.py`), which the paper's ablation finds equivalent in
chemistry on acyclic documents and which extends to cycles for free by letting
a step refer to a later index. Nesting for humans and the parser, indexed steps
for the model.

**Operating conditions** — keyed parameter blocks,
`((HEA (:T 350 K) (:t 2 h)) stream)`, backward compatible by desugaring `HEA2`
to `(HEA (:cond 2))`, an uninterpreted ordinal in a reserved key. Every
existing document keeps its meaning. Declare a dimension per key in the machine
database and validate that units reduce to it, so `(:T 350 bar)` is an error.
This is not material balance: dimensional consistency of annotations is not
conservation of mass.

Canonicalisation sorts parameters by key and normalises units to SI base, which
keeps idempotence by the same argument as rule 3. **One warning**: a small
vocabulary cannot absorb real-valued parameters, so the training view will want
to bin them into per-key corpus quantiles — and binning is *lossy*, unlike
every other step of canonicalisation here. Keep the exact value in the document
and the bin in the training view, and never let the lossy form be mistaken for
the canonical one.
