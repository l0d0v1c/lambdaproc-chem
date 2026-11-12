"""λproc test suite: round-trip, canonicalization, validation, DAG."""

import glob
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lambdaproc import (  # noqa: E402
    ParseError, canonical, load_machines, parse, to_dag, validate,
)

# The machine database is the industrial half of the language: it maps a
# `(use (EC 6.01))` binding to a catalogue entry. It is optional -- a document
# parses, canonicalises and validates without it, only the machine ids go
# unchecked -- and it is not distributed with every copy of this repository,
# so the two tests that need it skip rather than fail.
_DB_PATH = os.path.join(ROOT, "machines_procedes.json")
DB = load_machines(_DB_PATH) if os.path.exists(_DB_PATH) else None
needs_db = unittest.skipIf(DB is None, "machines_procedes.json not present")
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "examples", "*.lproc")))

PREAMBLE = """
(use (M "melange" :comm) (D 5.02 (distillat residu)) (K 13.01 (cristaux liqueur)))
(mat (a "A") (b "B") (w "additif" :minor))
"""


class TestExamples(unittest.TestCase):
    def test_examples_exist(self):
        self.assertGreaterEqual(len(EXAMPLES), 4)

    @needs_db
    def test_examples_valid_against_db(self):
        for path in EXAMPLES:
            with open(path, encoding="utf-8") as fh:
                doc = parse(fh.read())
            v = validate(doc, DB)
            self.assertEqual(v.errors, [], f"{path}: {v.errors}")

    def test_examples_roundtrip_idempotent(self):
        for path in EXAMPLES:
            with open(path, encoding="utf-8") as fh:
                doc = parse(fh.read())
            c1 = canonical(doc)
            c2 = canonical(parse(c1))
            self.assertEqual(c1, c2, f"{path}: canonical form not idempotent")

    def test_cocktails_count(self):
        with open(os.path.join(ROOT, "examples", "cocktails.lproc"),
                  encoding="utf-8") as fh:
            doc = parse(fh.read())
        from lambdaproc.core import Def
        defs = [f for f in doc.forms if isinstance(f, Def)]
        self.assertEqual(len(defs), 20)


class TestCanonical(unittest.TestCase):
    def test_commutative_sorting(self):
        d1 = parse(PREAMBLE + "(def x (M a b w))")
        d2 = parse(PREAMBLE + "(def x (M w b a))")
        self.assertEqual(canonical(d1), canonical(d2))

    def test_non_commutative_preserved(self):
        d = parse('(use (F "filtre"))(mat (a "A") (b "B"))(def x (F b a))')
        self.assertIn("(F b a)", canonical(d))

    def test_index_split(self):
        d = parse(PREAMBLE + "(def x (M a4 b12))")
        self.assertIn("a4", canonical(d))
        self.assertIn("b12", canonical(d))


class TestValidator(unittest.TestCase):
    def check(self, body, db=None):
        return validate(parse(PREAMBLE + body), db)

    def test_valid_simple(self):
        v = self.check("(def x (M a2 b3 w))")
        self.assertEqual(v.errors, [])

    def test_undeclared_symbol(self):
        v = self.check("(def x (M a zz))")
        self.assertTrue(any("undeclared symbol 'zz'" in e for e in v.errors))

    @needs_db
    def test_unknown_machine_id(self):
        v = validate(parse('(use (Z 99.99))(mat (a "A"))(def x (Z a))'), DB)
        self.assertTrue(any("99.99" in e for e in v.errors))

    def test_multi_output_requires_selector(self):
        v = self.check("(def x (M a (D b)))")
        self.assertTrue(any("multi-output" in e for e in v.errors))

    def test_selector_ok(self):
        v = self.check("(def x (M a (distillat (D b))))")
        self.assertEqual(v.errors, [])

    def test_selector_wrong_op(self):
        v = self.check("(def x (cristaux (D b)))")
        self.assertTrue(any("selector 'cristaux'" in e for e in v.errors))

    def test_lambda_macro_arity(self):
        v = self.check("(def f (lambda (s) (M s a)))(def x (f a b))")
        self.assertTrue(any("expects 1 argument" in e for e in v.errors))

    def test_lambda_macro_ok(self):
        v = self.check("(def f (lambda (s) (M s a)))(def x (f b))")
        self.assertEqual(v.errors, [])

    def test_material_as_operation(self):
        v = self.check("(def x (a b))")
        self.assertTrue(any("material 'a' used as an operation" in e for e in v.errors))

    def test_def_name_trailing_digits_rejected(self):
        with self.assertRaises(ParseError):
            parse("(def x2 (M a))")


class TestDag(unittest.TestCase):
    def test_shared_subexpression_merges(self):
        d = parse(PREAMBLE
                  + "(def tete (distillat (D a)))"
                  + "(def pied (residu (D a)))")
        nodes, tree = to_dag(d)
        # distinct: a, (D a), (distillat ...), (residu ...) -> 4 nodes, 6 occurrences
        self.assertEqual(len(nodes), 4)
        self.assertEqual(tree, 6)

    def test_macro_expansion_shares(self):
        d = parse(PREAMBLE
                  + "(def base (M a b))"
                  + "(def u (distillat (D base)))"
                  + "(def v (residu (D base)))")
        nodes, _ = to_dag(d)
        keys = list(nodes)
        self.assertTrue(any(k == "(M a b)" for k in keys))
        self.assertEqual(sum(1 for k in keys if k.startswith("(D ")), 1)


if __name__ == "__main__":
    unittest.main()
