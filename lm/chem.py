"""Chemical sanity checks on a LambdaProc document (shared by proposal and eval)."""

import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MAT_RE = re.compile(r'\(([a-z]+) "([^"]*)"\)')
OP_RE = re.compile(r"\(([A-Z]{3})")

# Materials that carry no synthetic information (solvents, drying agents,
# chromatography media): excluded from reactant recall and from inventories.
SOLVENTS = {"O", "CCOC(C)=O", "ClCCl", "CO", "CCO", "ClC(Cl)Cl", "CCOCC", "CC(C)=O",
            "Cc1ccccc1", "C1CCOC1", "CC#N", "CN(C)C=O", "CCCCCC", "CCCCCCC", "CCCCC",
            "O=[Si]=O", "O.[Cl-].[Na+]", "O=S(=O)([O-])[O-].[Na+].[Na+]",
            "O=S(=O)([O-])[O-].[Mg+2]", "CS(C)=O", "C1COCCO1",
            "CCCCCC.CCOC(C)=O", "CC(C)O", "CC(C)(C)O", "CCCCO", "ClCCCl"}


def mats_of(canon: str) -> set:
    return {smi for _, smi in MAT_RE.findall(canon)}


def reactants_of(canon: str) -> set:
    return mats_of(canon) - SOLVENTS


def skeleton(canon: str) -> str:
    """Operation sequence of the body, indices stripped."""
    return " ".join(OP_RE.findall(canon.strip().split("\n")[-1]))


def elements(smi: str) -> set:
    mol = Chem.MolFromSmiles(smi)
    return {a.GetSymbol() for a in mol.GetAtoms()} - {"H"} if mol else set()


def smiles_ok(canon: str) -> bool:
    return all(Chem.MolFromSmiles(s) is not None for s in mats_of(canon))


def elements_ok(target: str, canon: str) -> bool:
    """Every heavy element of the target must occur in some material."""
    mats = mats_of(canon)
    have = set().union(*(elements(s) for s in mats)) if mats else set()
    return elements(target) <= have


# Ubiquitous auxiliary reagents: acids, bases, salts, hydride and borohydride
# reductants, coupling additives, phosphine ligands, palladium. The thirty
# commonest non-solvent materials of the corpus, covering 32.9 % of all
# material mentions. They are excluded from `bb_recall` for the same reason
# solvents are excluded from `reactant_recall`: proposing them is nearly free
# and tells us nothing about whether a model understood the target. Measured:
# a constant answer made of the four commonest scores 0.133 reactant_recall
# and 0.000 bb_recall. Percentages are the share of mentions each accounts for.
COMMON_REAGENTS = {
    "Cl",  # 4.20%
    "O=C([O-])O.[Na+]",  # 3.92%
    "[Na+].[OH-]",  # 2.54%
    "CCN(CC)CC",  # 2.51%
    "O=C([O-])[O-].[K+].[K+]",  # 1.96%
    "[Cl-].[Na+]",  # 1.52%
    "CC(=O)O",  # 1.33%
    "[Cl-].[NH4+]",  # 1.23%
    "CCN(C(C)C)C(C)C",  # 1.22%
    "[H-].[Na+]",  # 1.17%
    "O=C(O)C(F)(F)F",  # 0.95%
    "c1ccncc1",  # 0.92%
    "CO.ClCCl",  # 0.89%
    "O=C([O-])[O-].[Na+].[Na+]",  # 0.86%
    "[Pd]",  # 0.81%
    "N#N",  # 0.79%
    "CCOC(C)=O.CCCCCC",  # 0.51%
    "O=C([O-])[O-].[Cs+].[Cs+]",  # 0.50%
    "N",  # 0.49%
    "[Li][CH2]CCC",  # 0.48%
    "c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",  # 0.45%
    "[K+].[OH-]",  # 0.45%
    "O=S(=O)(O)O",  # 0.45%
    "[BH4-].[Na+]",  # 0.44%
    "[H][H]",  # 0.42%
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1",  # 0.40%
    "CI",  # 0.40%
    "c1ccccc1",  # 0.38%
    "On1nnc2ccccc21",  # 0.36%
    "CN(C)c1ccncc1",  # 0.34%
}


def building_blocks_of(canon: str) -> set:
    """Materials that carry synthetic information: no solvents, no ubiquitous
    reagents. The basis of `bb_recall`, the metric that separates a model from
    a nearest-neighbour lookup (see lm/baseline_retrieval.py)."""
    return mats_of(canon) - SOLVENTS - COMMON_REAGENTS
