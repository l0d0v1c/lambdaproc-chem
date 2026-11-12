#!/usr/bin/env python3
"""Convert Lowe's USPTO reaction CML files into LambdaProc documents.

Source: D. Lowe, "Chemical reactions from US patents (1976-Sep2016)",
figshare doi:10.6084/m9.figshare.5104873 (CC0). Each <reaction> carries a
<dl:reactionActionList> of structured synthesis actions (Add, Stir, Heat,
Filter...) with chemical references and normalized Time/Temperature
parameters. Each accepted reaction becomes one self-contained .lproc
document; every document is parsed and validated with lambdaproc before
being written, so the corpus is valid by construction.

Usage:
    python scripts/uspto_to_lproc.py data/uspto/extracted/grants/2014/*.xml \
        -o data/uspto/lproc --jsonl data/uspto/corpus.jsonl [--limit N]
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from lambdaproc import canonical, parse, validate  # noqa: E402

CML = "{http://www.xml-cml.org/schema}"
DL = "{http://bitbucket.org/dan2097}"

# Lowe action type -> (lproc op symbol, French label)
ACTION_OPS = {
    "Add":         ("ADD", "addition to the reaction medium"),
    "Dissolve":    ("DIS", "dissolution"),
    "Heat":        ("HEA", "heating / reflux"),
    "Cool":        ("COO", "cooling"),
    "Stir":        ("STI", "stirring"),
    "Filter":      ("FIL", "filtration"),
    "Extract":     ("EXT", "liquid-liquid extraction"),
    "Wash":        ("WAS", "washing"),
    "Dry":         ("DRY", "drying"),
    "Concentrate": ("CON", "concentration / evaporation"),
    "Remove":      ("REM", "removal (evaporation, decantation)"),
    "Purify":      ("PUR", "purification (chromatography, recrystallisation)"),
    "Quench":      ("QUE", "quench"),
    "Precipitate": ("PRE", "precipitation"),
    "Partition":   ("PAR", "partition between phases"),
    "Degass":      ("DEG", "degassing"),
    "Irradiate":   ("IRR", "irradiation"),
    "Mill":        ("MIL", "milling"),
    "Recover":     ("REC", "recovery"),
    "Wait":        ("WAI", "standing / ageing"),
}
SKIPPED_ACTIONS = {"Sample", "Synthesize", "Yield", "ApparatusAction", "Unknown"}

# --smiles mode: inline chemicals Lowe could not resolve to a structure.
# Names that denote the current stream itself are dropped (the stream is
# already the first argument); common solvents/media get a fixed SMILES;
# mixtures ("ethyl acetate hexanes") are resolved component-wise.
STREAM_NAMES = {
    "crude product", "crude material", "crude compound", "crude residue",
    "crude", "residue", "resultant residue", "resulting residue",
    "resultant mixture", "resulting mixture", "reaction mixture", "mixture",
    "solution", "resultant solution", "resulting solution", "solid", "oil",
    "product", "filtrate", "organic layer", "organic phase", "organic layers",
    "aqueous layer", "aqueous phase", "title compound", "precipitate",
    "crude oil", "crude solid", "solvent", "solvents", "aqueous solution",
    "resultant solid", "resulting solid", "reaction solution",
}
MEDIA_SMILES = {
    "silica gel": "O=[Si]=O", "silica": "O=[Si]=O", "sio2": "O=[Si]=O",
    "ice": "O", "ice water": "O", "water": "O", "brine": "O.[Na+].[Cl-]",
    "hexanes": "CCCCCC", "hexane": "CCCCCC", "heptanes": "CCCCCCC",
    "heptane": "CCCCCCC", "petroleum ether": "CCCCC", "petroleum": "CCCCC", "petrol": "CCCCC",
    "pentane": "CCCCC", "pentanes": "CCCCC",
    "ethyl acetate": "CCOC(C)=O", "etoac": "CCOC(C)=O",
    "dichloromethane": "ClCCl", "dcm": "ClCCl", "methanol": "CO",
    "meoh": "CO", "ethanol": "CCO", "etoh": "CCO", "chloroform": "ClC(Cl)Cl",
    "ether": "CCOCC", "diethyl ether": "CCOCC", "acetone": "CC(C)=O",
    "toluene": "Cc1ccccc1", "naso4": "O=S(=O)([O-])[O-].[Na+].[Na+]",
    "mgso4": "O=S(=O)([O-])[O-].[Mg+2]", "celite": "O=[Si]=O",
    "thf": "C1CCOC1", "acetonitrile": "CC#N", "dmf": "CN(C)C=O",
    "tea": "CCN(CC)CC", "dmap": "CN(C)c1ccncc1", "dipea": "CCN(C(C)C)C(C)C",
    "cuprous iodide": "[Cu]I", "so2cl2": "O=S(=O)(Cl)Cl",
}
_MEDIA_KEYS = sorted(MEDIA_SMILES, key=len, reverse=True)

MIN_OPS = 4
MIN_MATS = 2
MAX_LABEL = 120


def mat_sym(i: int) -> str:
    """0 -> a, 1 -> b, ..., 25 -> z, 26 -> aa ... (letters only)."""
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def clean_label(name: str) -> str:
    name = re.sub(r"\s+", " ", name.replace('"', "'")).strip()
    return name[:MAX_LABEL] if name else "unknown"


def first_int(text) -> "int | None":
    if not text:
        return None
    m = re.search(r"-?\d+", str(text))
    if not m:
        return None
    v = int(m.group(0))
    return v if 0 <= v < 100000 else None


def molecule_name(elem) -> str:
    mol = elem.find(f"{CML}molecule")
    if mol is not None:
        nm = mol.find(f"{CML}name")
        if nm is not None and nm.text:
            return nm.text
    return ""


def norm_name(name: str) -> str:
    return re.sub(r"[\s\-/:,()]+", " ", (name or "").lower()).strip()


def resolve_media(name: str) -> "str | None":
    """Greedy component-wise lookup of a solvent/medium (or mixture) name."""
    rest, comps = norm_name(name), []
    while rest:
        for key in _MEDIA_KEYS:
            if rest == key or rest.startswith(key + " "):
                comps.append(MEDIA_SMILES[key])
                rest = rest[len(key):].strip()
                break
        else:
            return None
    return ".".join(dict.fromkeys(comps)) if comps else None


_CANON_CACHE = {}


def canon_smiles(smi: str) -> "str | None":
    """RDKit canonical SMILES (cached); None if RDKit rejects it."""
    if smi in _CANON_CACHE:
        return _CANON_CACHE[smi]
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi)
    out = Chem.MolToSmiles(mol) if mol is not None else None
    if len(_CANON_CACHE) < 2_000_000:
        _CANON_CACHE[smi] = out
    return out


def chem_smiles(elem) -> "str | None":
    ident = elem.find(f"{CML}identifier[@dictRef='cml:smiles']")
    return ident.get("value") if ident is not None else None


def amount_index(elem) -> "int | None":
    """Indicative quantity: ml, else g, else mmol (first available)."""
    amounts = {}
    for am in elem.findall(f"{CML}amount"):
        amounts[am.get(f"{DL}propertyType")] = am.get(f"{DL}normalizedValue")
    for prop, scale in (("VOLUME", 1000), ("MASS", 1000), ("AMOUNT", 1000)):
        raw = amounts.get(prop)
        if raw:
            try:
                v = round(float(raw) * scale)
            except ValueError:
                continue
            if 0 < v < 100000:
                return v
    return None


def op_index(action_elem) -> "int | None":
    """Indicative op parameter: temperature (°C) first, else time (minutes)."""
    temp, time_s = None, None
    for p in action_elem.findall(f"{DL}parameter"):
        kind = p.get("propertyType")
        if kind == "Temperature" and temp is None:
            temp = first_int(p.get("normalizedValue"))
        elif kind == "Time" and time_s is None:
            time_s = first_int(p.get("normalizedValue"))
    if temp is not None:
        return temp
    if time_s is not None and time_s > 0:
        return max(1, round(time_s / 60))
    return None


class Converter:
    """smiles=False: material labels are names (human-readable corpus).
    smiles=True: labels are RDKit-canonical SMILES and the product SMILES is
    recorded, for target -> procedure training (see lm/)."""

    def __init__(self, smiles: bool = False):
        self.smiles = smiles
        self.stats = {"reactions": 0, "accepted": 0, "too_few_ops": 0,
                      "too_few_mats": 0, "missing_ref": 0, "invalid": 0}
        if smiles:
            self.stats.update({"no_product": 0, "unresolved_chem": 0})
        self.product = None  # product SMILES of the last accepted reaction

    def convert_reaction(self, rx) -> "tuple[str, str, str] | None":
        """Return (doc_id, paragraph, lproc_text) or None if rejected."""
        self.stats["reactions"] += 1
        src = rx.find(f"{DL}source")
        doc_id = src.findtext(f"{DL}documentId", "unknown") if src is not None else "unknown"
        para = src.findtext(f"{DL}paragraphNum", "0") if src is not None else "0"
        heading = src.findtext(f"{DL}headingText", "") if src is not None else ""

        product = None
        if self.smiles:
            for prod in rx.iterfind(f"{CML}productList/{CML}product"):
                smi = chem_smiles(prod)
                product = canon_smiles(smi) if smi else None
                if product:
                    break
            if not product:
                self.stats["no_product"] += 1
                return None

        # molecules referenced by id (reactants, spectators, products)
        by_ref = {}
        for lst in ("reactantList", "spectatorList", "productList"):
            parent = rx.find(f"{CML}{lst}")
            if parent is None:
                continue
            for chem in parent:
                mol = chem.find(f"{CML}molecule")
                if mol is None:
                    continue
                name = molecule_name(chem)
                if self.smiles:
                    if norm_name(name) in STREAM_NAMES:
                        by_ref[mol.get("id")] = None  # the stream itself
                        continue
                    smi = chem_smiles(chem)
                    name = (canon_smiles(smi) if smi else None) or resolve_media(name)
                    if name is None:
                        self.stats["unresolved_chem"] += 1
                        return None
                by_ref[mol.get("id")] = (name, amount_index(chem))

        actions = rx.find(f"{DL}reactionActionList")
        if actions is None:
            self.stats["too_few_ops"] += 1
            return None

        mats = {}        # key -> (sym, label)
        mat_order = []   # emission order
        used_ops = {}    # op sym -> label

        def mat_for(key, name, qty):
            if self.smiles:
                key = name  # one symbol per distinct structure
            if key not in mats:
                sym = mat_sym(len(mats))
                mats[key] = (sym, name if self.smiles else clean_label(name))
                mat_order.append(key)
            sym = mats[key][0]
            return sym + (str(qty) if qty else "")

        stream = None
        n_ops = 0
        for act in actions.findall(f"{DL}reactionAction"):
            kind = act.get("action")
            if kind in SKIPPED_ACTIONS or kind not in ACTION_OPS:
                continue
            op, label = ACTION_OPS[kind]
            args = []
            for ch in act.findall(f"{DL}chemical"):       # by reference
                ref = ch.get("ref")
                if ref not in by_ref:
                    self.stats["missing_ref"] += 1
                    return None
                if by_ref[ref] is None:
                    continue
                name, qty = by_ref[ref]
                args.append(mat_for(ref, name, qty))
            for ch in act.findall(f"{CML}chemical"):      # inline (solvents...)
                name = molecule_name(ch)
                if self.smiles:
                    if norm_name(name) in STREAM_NAMES:
                        continue
                    smi = chem_smiles(ch)
                    name = (canon_smiles(smi) if smi else None) or resolve_media(name)
                    if name is None:
                        self.stats["unresolved_chem"] += 1
                        return None
                args.append(mat_for(("inline", name), name, amount_index(ch)))
            if stream is None and not args:
                continue  # nothing to act on yet
            used_ops[op] = label
            idx = op_index(act)
            head = op + (str(idx) if idx is not None else "")
            parts = ([stream] if stream is not None else []) + args
            stream = "(" + " ".join([head] + parts) + ")"
            n_ops += 1

        if n_ops < MIN_OPS or stream is None:
            self.stats["too_few_ops"] += 1
            return None
        if len(mats) < MIN_MATS:
            self.stats["too_few_mats"] += 1
            return None

        lines = []
        if heading:
            lines.append(f"; {doc_id} §{para} — {clean_label(heading)}")
        else:
            lines.append(f"; {doc_id} §{para}")
        lines.append("(use")
        for op in sorted(used_ops):
            lines.append(f'  ({op} "{used_ops[op]}")')
        lines[-1] += ")"
        lines.append("(mat")
        for key in mat_order:
            sym, label = mats[key]
            lines.append(f'  ({sym} "{label}")')
        lines[-1] += ")"
        lines.append(f"(def procede {stream})")
        text = "\n".join(lines) + "\n"

        # valid-by-construction gate
        try:
            doc = parse(text)
            v = validate(doc)
            if v.errors:
                self.stats["invalid"] += 1
                return None
        except Exception:
            self.stats["invalid"] += 1
            return None
        self.stats["accepted"] += 1
        self.product = product
        return doc_id, para, text


def convert_files(xml_files, outdir=None, jsonl_path=None, limit=None,
                  smiles=False) -> dict:
    """Convert CML files; write .lproc files (if outdir) and/or JSONL."""
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    conv = Converter(smiles=smiles)
    jsonl = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
    done = False
    for path in xml_files:
        if done:
            break
        for _, elem in ET.iterparse(path):
            if elem.tag != f"{CML}reaction":
                continue
            res = conv.convert_reaction(elem)
            elem.clear()
            if res is None:
                continue
            doc_id, para, text = res
            if outdir:
                fname = f"{doc_id}_{para}.lproc"
                with open(os.path.join(outdir, fname), "w", encoding="utf-8") as fh:
                    fh.write(text)
            if jsonl:
                rec = {"id": f"{doc_id}_{para}"}
                if smiles:
                    rec["product"] = conv.product
                rec["canonical"] = canonical(parse(text))
                jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if limit and conv.stats["accepted"] >= limit:
                done = True
                break
    if jsonl:
        jsonl.close()
    return conv.stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("xml_files", nargs="+")
    ap.add_argument("-o", "--outdir", default=None,
                    help="write one .lproc file per reaction (omit for JSONL only)")
    ap.add_argument("--jsonl", help="write canonical strings as JSONL")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N accepted reactions")
    ap.add_argument("--smiles", action="store_true",
                    help="SMILES material labels + product field (LM corpus)")
    args = ap.parse_args(argv)
    if not args.outdir and not args.jsonl:
        ap.error("nothing to do: give --outdir and/or --jsonl")

    stats = convert_files(args.xml_files, args.outdir, args.jsonl, args.limit,
                          smiles=args.smiles)
    print(json.dumps(stats, indent=2))
    acc = stats["accepted"]
    print(f"accepted {acc}/{stats['reactions']} reactions "
          f"({100 * acc / max(1, stats['reactions']):.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
