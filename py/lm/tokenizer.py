"""Lossless tokenizer for the target->procedure LM.

Training line = product SMILES, then the LM view of a LambdaProc document:
the `(mat ...)` and `(def procede ...)` forms on one line, with the `(use ...)`
block dropped (it is fully derivable from the operation symbols used, see
`lm.lmview`). Material labels inside quotes are SMILES.

Token classes
  - SMILES tokens (Schwaller et al. regex) inside the product and inside quotes;
  - lproc tokens outside quotes: "(", ")", '"', keywords, op symbols, material
    symbols, and single digits (indices are spelled digit by digit, glued to
    the preceding symbol on decode);
  - specials: <pad> <bos> <sep> <eos> <unk>.
Whitespace outside quotes is dropped and re-inserted deterministically on
decode, so encode/decode is an exact round trip on canonical strings.
"""

import json
import re

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>", "<inv>"]
PAD, BOS, SEP, EOS, UNK, INV = range(6)
# Vocabularies saved before <inv> existed have only the first five; decoding
# therefore skips specials by name, never by index.
SPECIAL_SET = set(SPECIALS)

SMILES_RE = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|<|\*|\$|%[0-9]{2}|[0-9])"
)
# '|' separates steps in the flat view of the representation ablation
# (lm.flatview); it never occurs in a LambdaProc document, so it simply never
# enters the vocabulary of the nested view.
LPROC_RE = re.compile(r'\s+|\(|\)|"|[0-9]|\||[A-Za-z][A-Za-z-]*')


def smiles_tokens(s: str) -> list:
    toks = SMILES_RE.findall(s)
    if "".join(toks) != s:
        raise ValueError(f"untokenizable SMILES: {s!r}")
    return toks


def lproc_tokens(text: str) -> list:
    """Tokenize the LM view (one line). Quoted spans are SMILES."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':
            j = text.index('"', i + 1)
            out.append('"')
            out.extend(smiles_tokens(text[i + 1:j]))
            out.append('"')
            i = j + 1
            continue
        m = LPROC_RE.match(text, i)
        if not m:
            raise ValueError(f"untokenizable lproc at {i}: {text[i:i + 20]!r}")
        if not m.group(0).isspace():
            out.append(m.group(0))
        i = m.end()
    return out


def lproc_detok(toks: list) -> str:
    """Inverse of lproc_tokens (canonical spacing)."""
    out, in_str, prev = [], False, None
    for t in toks:
        if in_str:
            out.append(t)
            if t == '"':
                in_str = False
        elif t == '"':
            out.append(" " + t if prev not in (None, "(") else t)
            in_str = True
        elif t == ")" or t.isdigit() or prev in (None, "("):
            # digits glue to the symbol; ')' glues; nothing after '('
            out.append(t)
        elif prev == ")" and t == "(" and out and out[-1] == ")" and _depth(out) == 0:
            out.append("\n(")  # top-level forms on separate lines
        else:
            out.append(" " + t)
        prev = t
    return "".join(out)


def _depth(parts) -> int:
    s = "".join(parts)
    d, in_str = 0, False
    for c in s:
        if c == '"':
            in_str = not in_str
        elif not in_str:
            d += (c == "(") - (c == ")")
    return d


class Tokenizer:
    def __init__(self, vocab: list):
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, pairs, min_count: int = 1):
        """pairs: (product, view) or (product, inventory, view)."""
        counts = {}
        for pair in pairs:
            product, view = pair[0], pair[-1]
            toks = smiles_tokens(product) + lproc_tokens(view)
            for inv in pair[1:-1]:
                for smi in inv:
                    toks += smiles_tokens(smi)
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        vocab = SPECIALS + sorted(t for t, c in counts.items() if c >= min_count)
        return cls(vocab)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.itos, fh, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def __len__(self):
        return len(self.itos)

    def ids(self, toks):
        return [self.stoi.get(t, UNK) for t in toks]

    def encode_prompt(self, product: str, inventory=None) -> list:
        """BOS target [<inv> m1 . m2 ...] SEP.

        The inventory is the set of materials the chemist has at hand; it is
        emitted in sorted order so its order carries no information.
        """
        ids = [BOS] + self.ids(smiles_tokens(product))
        if inventory is not None:
            ids.append(INV)
            for k, smi in enumerate(sorted(inventory)):
                if k:
                    ids += self.ids(["."])
                ids += self.ids(smiles_tokens(smi))
        return ids + [SEP]

    def encode_pair(self, product: str, view: str,
                    inventory=None) -> "tuple[list, int]":
        """Return (ids, prompt_len). Loss is applied after prompt_len."""
        p = self.encode_prompt(product, inventory)
        return p + self.ids(lproc_tokens(view)) + [EOS], len(p)

    def decode_view(self, ids) -> str:
        toks = []
        for i in ids:
            if i == EOS:
                break
            if self.itos[i] in SPECIAL_SET:
                continue
            toks.append(self.itos[i])
        return lproc_detok(toks)
