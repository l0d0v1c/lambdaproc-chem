"""Inventory conditioning: "here is what I have, make this molecule".

An inventory is a shelf of available materials shown to the model before the
procedure. It is built from the reference procedure's non-solvent materials,
each kept with probability KEEP_PROB (so the model cannot assume the shelf is
complete), plus distractors drawn from the corpus material distribution (so
copying the whole shelf is not a winning strategy). Solvents are never listed:
they are assumed always available.
"""

import json

KEEP_PROB = 0.8
MIN_DISTRACTORS = 2

# Mixture used by `--shelf-mix robust`. The training-like regime alone taught
# the model that the shelf always contains the answer: a shelf missing every
# true reactant occurs with probability 0.2**n under independent keeping,
# under 1 % for three reactants, so "nothing here will do" was never seen.
# Note that the *target document is unchanged* in every regime -- only the
# shelf differs. An insufficient shelf therefore teaches exactly the wanted
# behaviour: write the procedure you would have written anyway.
REGIMES = {"trained": 0.70, "absent": 0.15, "hard": 0.15}


def build(reactants, pool, rng, keep_prob=KEEP_PROB,
          min_distractors=MIN_DISTRACTORS) -> list:
    """reactants: set of true non-solvent SMILES. pool: list for distractors."""
    kept = [r for r in sorted(reactants) if rng.random() < keep_prob]
    n_dist = max(min_distractors, len(reactants))
    shelf = set(kept)
    tries = 0
    while len(shelf) < len(kept) + n_dist and tries < 50 * n_dist:
        tries += 1
        cand = pool[rng.integers(len(pool))]
        if cand not in reactants:
            shelf.add(cand)
    return sorted(shelf)


def draw_regime(rng, regimes=None) -> str:
    """Pick a regime. Separate from `build_mixed` so that the caller can learn
    which examples need near-neighbour distractors *before* computing them:
    only ~15 % do, and the Tanimoto sweep over the pool is the expensive part
    of preparing the dataset."""
    regimes = regimes or REGIMES
    names, probs = zip(*sorted(regimes.items()))
    return names[int(rng.choice(len(names), p=probs))]


def build_mixed(reactants, pool, rng, regime=None, regimes=None,
                hard_pool=None, keep_prob=KEEP_PROB,
                min_distractors=MIN_DISTRACTORS):
    """A shelf drawn from the mixture of regimes, with the regime name.

    `hard_pool` is an ordered list of pool materials similar to the target
    (most similar first), used as distractors in the `hard` regime; when it is
    None or empty that regime degrades to `trained`.
    """
    which = regime or draw_regime(rng, regimes)
    n_dist = max(min_distractors, len(reactants))

    if which == "absent":
        shelf, tries = set(), 0
        while len(shelf) < n_dist + 1 and tries < 50 * (n_dist + 1):
            tries += 1
            cand = pool[rng.integers(len(pool))]
            if cand not in reactants:
                shelf.add(cand)
        return sorted(shelf), which

    kept = [r for r in sorted(reactants) if rng.random() < keep_prob]
    if which == "hard" and hard_pool:
        shelf = set(kept)
        for cand in hard_pool:
            if len(shelf) >= len(kept) + n_dist:
                break
            if cand not in reactants:
                shelf.add(cand)
        if len(shelf) >= len(kept) + min_distractors:
            return sorted(shelf), which
    return build(reactants, pool, rng, keep_prob, min_distractors), "trained"


def save_pool(path, pool):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pool, fh)


def load_pool(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
