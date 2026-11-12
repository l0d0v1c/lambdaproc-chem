"""LambdaProc core: lexer, parser, AST, canonicalizer, validator, DAG builder.

LambdaProc is an S-expression language describing processes (recipes,
industrial flowsheets) as compositions of unit operations, inspired by
SMILES/SFILES on the process side and by LISP/lambda calculus on the
structural side. Pure stdlib, no dependencies.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Union

KEYWORDS = {"use", "mat", "def", "lambda"}
FLAGS = {":minor", ":comm"}
MACHINE_ID_RE = re.compile(r"^\d+\.\d+$")
# Commutative-by-default DB categories: 3 Mixing/Blending, 11 Homogenization
COMMUTATIVE_CATEGORIES = {3, 11}


# ---------------------------------------------------------------- AST nodes

@dataclass(frozen=True)
class Ref:
    """A symbol reference: material, op, macro, param or selector head.

    ``base`` never ends with a digit; trailing digits of the source token
    are the (indicative) ``index``.
    """
    base: str
    index: Optional[int] = None

    def __str__(self) -> str:
        return self.base + ("" if self.index is None else str(self.index))


@dataclass(frozen=True)
class App:
    head: Ref
    args: tuple = ()


@dataclass(frozen=True)
class Lambda:
    params: tuple
    body: Union[Ref, App]


Expr = Union[Ref, App]


@dataclass(frozen=True)
class UseBinding:
    sym: str
    ref: str            # machine id ("5.02") or free-text definition
    outputs: tuple = ()  # declared output names (selectors)
    comm: bool = False
    is_machine: bool = True


@dataclass(frozen=True)
class MatBinding:
    sym: str
    label: str
    minor: bool = False


@dataclass(frozen=True)
class UseDecl:
    bindings: tuple


@dataclass(frozen=True)
class MatDecl:
    bindings: tuple


@dataclass(frozen=True)
class Def:
    name: str
    body: Union[Ref, App, Lambda]


@dataclass(frozen=True)
class Document:
    forms: tuple


# ------------------------------------------------------------------- lexer

class LambdaProcError(Exception):
    pass


class ParseError(LambdaProcError):
    pass


def tokenize(text: str):
    tokens = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "()":
            tokens.append((c, c, line))
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\n":
                    raise ParseError(f"line {line}: unterminated string")
                j += 1
            if j >= n:
                raise ParseError(f"line {line}: unterminated string")
            tokens.append(("STRING", text[i + 1:j], line))
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '();"':
                j += 1
            tokens.append(("ATOM", text[i:j], line))
            i = j
    return tokens


def split_index(atom: str):
    """Split trailing digits as an indicative index: 'EC78' -> ('EC', 78)."""
    m = re.fullmatch(r"(.*?)(\d+)", atom)
    if m and m.group(1) and not m.group(1)[-1].isdigit():
        return m.group(1), int(m.group(2))
    return atom, None


def check_identifier(base: str, line: int) -> None:
    if not base:
        raise ParseError(f"line {line}: empty identifier")
    if not (base[0].isalpha()):
        raise ParseError(f"line {line}: identifier must start with a letter: {base!r}")
    for ch in base[1:]:
        if not (ch.isalnum() or ch == "-"):
            raise ParseError(f"line {line}: invalid character {ch!r} in {base!r}")


# ------------------------------------------------------------------ parser

class Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of input")
        self.pos += 1
        return tok

    def expect(self, kind):
        tok = self.next()
        if tok[0] != kind:
            raise ParseError(f"line {tok[2]}: expected {kind}, got {tok[1]!r}")
        return tok

    # ---- S-expression layer (generic nested lists) ----
    def read_sexpr(self):
        tok = self.next()
        if tok[0] == "(":
            items = []
            while True:
                nxt = self.peek()
                if nxt is None:
                    raise ParseError(f"line {tok[2]}: unclosed '('")
                if nxt[0] == ")":
                    self.next()
                    return ("LIST", items, tok[2])
                items.append(self.read_sexpr())
        if tok[0] == ")":
            raise ParseError(f"line {tok[2]}: unexpected ')'")
        return tok  # ATOM or STRING

    def parse_document(self) -> Document:
        forms = []
        while self.peek() is not None:
            forms.append(self.parse_form(self.read_sexpr()))
        return Document(tuple(forms))

    # ---- document forms ----
    def parse_form(self, sx):
        if sx[0] == "LIST" and sx[1] and sx[1][0][0] == "ATOM":
            head = sx[1][0][1]
            if head == "use":
                return self.parse_use(sx)
            if head == "mat":
                return self.parse_mat(sx)
            if head == "def":
                return self.parse_def(sx)
            if head == "lambda":
                raise ParseError(
                    f"line {sx[2]}: top-level lambda must be bound with (def name (lambda ...))")
        return self.parse_expr(sx)

    def parse_use(self, sx) -> UseDecl:
        bindings = []
        for item in sx[1][1:]:
            if item[0] != "LIST" or not item[1]:
                raise ParseError(f"line {sx[2]}: use binding must be a list")
            parts, line = item[1], item[2]
            if parts[0][0] != "ATOM":
                raise ParseError(f"line {line}: use binding must start with a symbol")
            sym = parts[0][1]
            check_identifier(sym, line)
            base, idx = split_index(sym)
            if idx is not None:
                raise ParseError(f"line {line}: declared symbol {sym!r} must not end with digits")
            if len(parts) < 2:
                raise ParseError(f"line {line}: use binding {sym!r} needs a machine id or a string definition")
            ref_tok = parts[1]
            if ref_tok[0] == "STRING":
                ref, is_machine = ref_tok[1], False
            elif ref_tok[0] == "ATOM" and MACHINE_ID_RE.match(ref_tok[1]):
                ref, is_machine = ref_tok[1], True
            else:
                raise ParseError(
                    f"line {line}: use binding {sym!r}: expected machine id (e.g. 5.02) or \"definition\"")
            outputs, comm = (), False
            for extra in parts[2:]:
                if extra[0] == "LIST":
                    outs = []
                    for o in extra[1]:
                        if o[0] != "ATOM":
                            raise ParseError(f"line {line}: output names must be symbols")
                        check_identifier(o[1], line)
                        b, i2 = split_index(o[1])
                        if i2 is not None:
                            raise ParseError(f"line {line}: output name {o[1]!r} must not end with digits")
                        outs.append(o[1])
                    outputs = tuple(outs)
                elif extra[0] == "ATOM" and extra[1] == ":comm":
                    comm = True
                else:
                    raise ParseError(f"line {line}: unexpected element {extra[1]!r} in use binding {sym!r}")
            bindings.append(UseBinding(sym, ref, outputs, comm, is_machine))
        return UseDecl(tuple(bindings))

    def parse_mat(self, sx) -> MatDecl:
        bindings = []
        for item in sx[1][1:]:
            if item[0] != "LIST" or not item[1]:
                raise ParseError(f"line {sx[2]}: mat binding must be a list")
            parts, line = item[1], item[2]
            if parts[0][0] != "ATOM":
                raise ParseError(f"line {line}: mat binding must start with a symbol")
            sym = parts[0][1]
            check_identifier(sym, line)
            base, idx = split_index(sym)
            if idx is not None:
                raise ParseError(f"line {line}: declared symbol {sym!r} must not end with digits")
            if len(parts) < 2 or parts[1][0] != "STRING":
                raise ParseError(f"line {line}: mat binding {sym!r} needs a \"label\"")
            label, minor = parts[1][1], False
            for extra in parts[2:]:
                if extra[0] == "ATOM" and extra[1] == ":minor":
                    minor = True
                else:
                    raise ParseError(f"line {line}: unexpected element in mat binding {sym!r}")
            bindings.append(MatBinding(sym, label, minor))
        return MatDecl(tuple(bindings))

    def parse_def(self, sx) -> Def:
        parts, line = sx[1], sx[2]
        if len(parts) != 3 or parts[1][0] != "ATOM":
            raise ParseError(f"line {line}: expected (def name body)")
        name = parts[1][1]
        check_identifier(name, line)
        base, idx = split_index(name)
        if idx is not None:
            raise ParseError(f"line {line}: def name {name!r} must not end with digits")
        body_sx = parts[2]
        if (body_sx[0] == "LIST" and body_sx[1]
                and body_sx[1][0][0] == "ATOM" and body_sx[1][0][1] == "lambda"):
            body = self.parse_lambda(body_sx)
        else:
            body = self.parse_expr(body_sx)
        return Def(name, body)

    def parse_lambda(self, sx) -> Lambda:
        parts, line = sx[1], sx[2]
        if len(parts) != 3 or parts[1][0] != "LIST":
            raise ParseError(f"line {line}: expected (lambda (params) body)")
        params = []
        for p in parts[1][1]:
            if p[0] != "ATOM":
                raise ParseError(f"line {line}: lambda params must be symbols")
            check_identifier(p[1], line)
            base, idx = split_index(p[1])
            if idx is not None:
                raise ParseError(f"line {line}: param {p[1]!r} must not end with digits")
            params.append(p[1])
        return Lambda(tuple(params), self.parse_expr(parts[2]))

    def parse_expr(self, sx) -> Expr:
        if sx[0] == "ATOM":
            atom, line = sx[1], sx[2]
            if atom in KEYWORDS or atom in FLAGS:
                raise ParseError(f"line {line}: keyword {atom!r} used as expression")
            if MACHINE_ID_RE.match(atom):
                return Ref(atom, None)
            base, idx = split_index(atom)
            check_identifier(base, line)
            return Ref(base, idx)
        if sx[0] == "STRING":
            raise ParseError(f"line {sx[2]}: string literal not allowed in expression")
        items, line = sx[1], sx[2]
        if not items:
            raise ParseError(f"line {line}: empty application ()")
        head_sx = items[0]
        if head_sx[0] != "ATOM":
            raise ParseError(f"line {line}: application head must be a symbol")
        atom = head_sx[1]
        if atom in KEYWORDS:
            raise ParseError(f"line {line}: {atom!r} form not allowed here")
        if MACHINE_ID_RE.match(atom):
            head = Ref(atom, None)
        else:
            base, idx = split_index(atom)
            check_identifier(base, line)
            head = Ref(base, idx)
        args = tuple(self.parse_expr(a) for a in items[1:])
        return App(head, args)


def parse(text: str) -> Document:
    return Parser(text).parse_document()


# ------------------------------------------------------------- environment

@dataclass
class Env:
    """Resolved document scope, built from use/mat/def declarations."""
    ops: dict = field(default_factory=dict)        # sym -> UseBinding
    mats: dict = field(default_factory=dict)       # sym -> MatBinding
    defs: dict = field(default_factory=dict)       # name -> Def
    selectors: dict = field(default_factory=dict)  # output name -> set of op syms
    errors: list = field(default_factory=list)

    def op_is_commutative(self, binding: UseBinding) -> bool:
        if binding.comm:
            return True
        if binding.is_machine:
            cat = int(binding.ref.split(".")[0])
            return cat in COMMUTATIVE_CATEGORIES
        return False


def build_env(doc: Document) -> Env:
    env = Env()
    for form in doc.forms:
        if isinstance(form, UseDecl):
            for b in form.bindings:
                if b.sym in env.ops or b.sym in env.mats:
                    env.errors.append(f"duplicate symbol {b.sym!r} in use")
                env.ops[b.sym] = b
                for out in b.outputs:
                    env.selectors.setdefault(out, set()).add(b.sym)
        elif isinstance(form, MatDecl):
            for b in form.bindings:
                if b.sym in env.mats or b.sym in env.ops:
                    env.errors.append(f"duplicate symbol {b.sym!r} in mat")
                env.mats[b.sym] = b
        elif isinstance(form, Def):
            if form.name in env.defs:
                env.errors.append(f"duplicate def {form.name!r}")
            env.defs[form.name] = form
    # namespace overlap: selector names must not collide with mats/ops/defs
    for out in env.selectors:
        for space, label in ((env.mats, "mat"), (env.ops, "use"), (env.defs, "def")):
            if out in space:
                env.errors.append(f"output name {out!r} collides with a {label} symbol")
    return env


# ------------------------------------------------------------- canonical form

def canon_expr(expr: Expr, env: Env) -> str:
    if isinstance(expr, Ref):
        return str(expr)
    parts = [canon_expr(a, env) for a in expr.args]
    binding = env.ops.get(expr.head.base)
    is_machine_head = MACHINE_ID_RE.match(expr.head.base) is not None
    commutative = False
    if binding is not None:
        commutative = env.op_is_commutative(binding)
    elif is_machine_head:
        commutative = int(expr.head.base.split(".")[0]) in COMMUTATIVE_CATEGORIES
    if commutative:
        parts.sort()
    inner = " ".join([str(expr.head)] + parts)
    return f"({inner})"


def canon_form(form, env: Env) -> str:
    if isinstance(form, UseDecl):
        items = []
        for b in sorted(form.bindings, key=lambda b: b.sym):
            s = b.sym + " " + (b.ref if b.is_machine else f'"{b.ref}"')
            if b.outputs:
                s += " (" + " ".join(b.outputs) + ")"
            if b.comm:
                s += " :comm"
            items.append(f"({s})")
        return "(use " + " ".join(items) + ")"
    if isinstance(form, MatDecl):
        items = []
        for b in sorted(form.bindings, key=lambda b: b.sym):
            s = f'{b.sym} "{b.label}"' + (" :minor" if b.minor else "")
            items.append(f"({s})")
        return "(mat " + " ".join(items) + ")"
    if isinstance(form, Def):
        if isinstance(form.body, Lambda):
            body = (f"(lambda ({' '.join(form.body.params)}) "
                    f"{canon_expr(form.body.body, env)})")
        else:
            body = canon_expr(form.body, env)
        return f"(def {form.name} {body})"
    return canon_expr(form, env)


def canonical(doc: Document) -> str:
    """Deterministic normal form: one top-level form per line."""
    env = build_env(doc)
    return "\n".join(canon_form(f, env) for f in doc.forms) + "\n"


# ----------------------------------------------------------------- machines DB

def load_machines(path: str) -> dict:
    """Return {machine_id: machine_dict} from machines_procedes.json."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    machines = {}
    for cat in data["categories"]:
        for m in cat["machines"]:
            machines[m["id"]] = m
    return machines


# ------------------------------------------------------------------ validator

class Validator:
    def __init__(self, doc: Document, machines: Optional[dict] = None):
        self.doc = doc
        self.env = build_env(doc)
        self.machines = machines
        self.errors = list(self.env.errors)
        self.warnings = []

    # ---- helpers ----
    def op_outputs(self, base: str) -> tuple:
        b = self.env.ops.get(base)
        if b is not None:
            return b.outputs
        return ()

    def is_op(self, base: str) -> bool:
        return base in self.env.ops or MACHINE_ID_RE.match(base) is not None

    def validate(self) -> bool:
        if self.machines is not None:
            for b in self.env.ops.values():
                if b.is_machine and b.ref not in self.machines:
                    self.errors.append(
                        f"use binding {b.sym!r}: machine id {b.ref!r} not found in database")
        for form in self.doc.forms:
            if isinstance(form, Def):
                if isinstance(form.body, Lambda):
                    self.check_expr(form.body.body, parent_head=None,
                                    params=set(form.body.params))
                else:
                    self.check_expr(form.body, parent_head=None, params=set())
            elif isinstance(form, (Ref, App)):
                self.check_expr(form, parent_head=None, params=set())
        return not self.errors

    def check_expr(self, expr: Expr, parent_head: Optional[Ref], params: set) -> None:
        if isinstance(expr, Ref):
            base = expr.base
            if base in params:
                if expr.index is not None:
                    self.errors.append(f"index not allowed on lambda param {base!r}")
                return
            if base in self.env.mats:
                return
            if base in self.env.defs:
                d = self.env.defs[base]
                if isinstance(d.body, Lambda):
                    self.errors.append(
                        f"macro {base!r} takes {len(d.body.params)} argument(s), used as a bare symbol")
                if expr.index is not None:
                    self.errors.append(f"index not allowed on macro reference {base!r}")
                return
            if base in self.env.ops or base in self.env.selectors:
                self.errors.append(f"operation/selector {base!r} used as a stream")
                return
            self.errors.append(f"undeclared symbol {base!r}")
            return

        head = expr.head
        base = head.base

        # selector application
        if base in self.env.selectors:
            if head.index is not None:
                self.errors.append(f"index not allowed on selector {base!r}")
            if len(expr.args) != 1:
                self.errors.append(f"selector {base!r} takes exactly one argument")
            else:
                arg = expr.args[0]
                if not (isinstance(arg, App)
                        and arg.head.base in self.env.selectors.get(base, set())):
                    self.errors.append(
                        f"selector {base!r} must be applied directly to an operation declaring it")
                self.check_expr(arg, parent_head=head, params=params)
            return

        # macro call
        if base in self.env.defs:
            d = self.env.defs[base]
            if not isinstance(d.body, Lambda):
                self.errors.append(f"macro {base!r} takes no argument")
            elif len(expr.args) != len(d.body.params):
                self.errors.append(
                    f"macro {base!r} expects {len(d.body.params)} argument(s), got {len(expr.args)}")
            if head.index is not None:
                self.errors.append(f"index not allowed on macro call {base!r}")
            for a in expr.args:
                self.check_expr(a, parent_head=head, params=params)
            return

        # operation application (bound symbol or raw machine id)
        if base in self.env.ops or MACHINE_ID_RE.match(base):
            if MACHINE_ID_RE.match(base) and self.machines is not None \
                    and base not in self.machines:
                self.errors.append(f"machine id {base!r} not found in database")
            outs = self.op_outputs(base)
            if len(outs) > 1:
                if parent_head is None or parent_head.base not in outs:
                    self.errors.append(
                        f"multi-output operation {base!r} must be consumed through one of "
                        f"its selectors ({', '.join(outs)})")
            if not expr.args:
                self.warnings.append(f"operation {base!r} applied to no stream")
            for a in expr.args:
                self.check_expr(a, parent_head=head, params=params)
            return

        if base in self.env.mats:
            self.errors.append(f"material {base!r} used as an operation")
            return
        if base in params:
            self.errors.append(f"lambda param {base!r} used as an operation")
            return
        self.errors.append(f"undeclared operation {base!r}")


def validate(doc: Document, machines: Optional[dict] = None) -> Validator:
    v = Validator(doc, machines)
    v.validate()
    return v


# ------------------------------------------------------------------- DAG

def to_dag(doc: Document):
    """Hash-cons every sub-expression of the document into a shared DAG.

    Returns (nodes, tree_size) where ``nodes`` maps the canonical string of
    each distinct sub-expression to its (expr, selector-free) node and
    ``tree_size`` is the total number of sub-expression occurrences. Two
    textual mentions of the same expression are the same process node.
    """
    env = build_env(doc)
    nodes = {}
    count = [0]

    def visit(expr: Expr):
        count[0] += 1
        key = canon_expr(expr, env)
        if key not in nodes:
            nodes[key] = expr
        if isinstance(expr, App):
            for a in expr.args:
                visit(a)

    def expand(expr: Expr) -> Expr:
        """Inline 0-ary macro references so shared sub-processes merge."""
        if isinstance(expr, Ref) and expr.base in env.defs:
            d = env.defs[expr.base]
            if not isinstance(d.body, Lambda):
                return expand(d.body)
        if isinstance(expr, App):
            return App(expr.head, tuple(expand(a) for a in expr.args))
        return expr

    for form in doc.forms:
        if isinstance(form, Def) and not isinstance(form.body, Lambda):
            visit(expand(form.body))
        elif isinstance(form, (Ref, App)):
            visit(expand(form))
    return nodes, count[0]
