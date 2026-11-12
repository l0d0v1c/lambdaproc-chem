"""λproc command-line interface.

Usage:
    python -m lambdaproc check <file.lproc>... [--db machines_procedes.json]
    python -m lambdaproc canon <file.lproc>    [--db machines_procedes.json]
    python -m lambdaproc stats <file.lproc>...
"""

import argparse
import sys

from .core import canonical, load_machines, parse, to_dag, validate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lambdaproc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "canon", "stats"):
        p = sub.add_parser(name)
        p.add_argument("files", nargs="+")
        p.add_argument("--db", default=None,
                       help="path to machines_procedes.json for id validation")
    args = ap.parse_args(argv)

    machines = load_machines(args.db) if args.db else None
    status = 0
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        try:
            doc = parse(text)
        except Exception as exc:
            print(f"{path}: PARSE ERROR: {exc}")
            status = 1
            continue
        if args.cmd == "canon":
            sys.stdout.write(canonical(doc))
            continue
        v = validate(doc, machines)
        nodes, tree = to_dag(doc)
        if args.cmd == "stats":
            print(f"{path}: {len(doc.forms)} forms, DAG {len(nodes)} nodes "
                  f"({tree} tree occurrences, sharing x{tree / max(len(nodes), 1):.2f})")
            continue
        if v.errors:
            print(f"{path}: INVALID ({len(v.errors)} error(s))")
            for e in v.errors:
                print(f"  error: {e}")
            status = 1
        else:
            db = "db-checked" if machines else "no-db"
            print(f"{path}: OK ({len(doc.forms)} forms, DAG {len(nodes)} nodes, {db})")
        for w in v.warnings:
            print(f"  warning: {w}")
    return status


if __name__ == "__main__":
    sys.exit(main())
