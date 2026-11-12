#!/usr/bin/env python3
"""Full USPTO grants (1976-Sep2016) -> λproc JSONL corpus.

Extracts data/uspto/grants_cml.7z once, converts year by year (deleting
each year's XML right after conversion to cap disk usage), writes one
JSONL per year under data/uspto/corpus_years/, then merges everything
into corpus_grants_full.jsonl and a deduplicated
corpus_grants_full_dedup.jsonl (unique canonical strings).

With --smiles, material labels are canonical SMILES and each record carries
the product SMILES (target -> procedure LM corpus); outputs are then named
corpus_smiles_* and deduplication keys on (product, canonical).

Run from the repository root:
    python scripts/run_full_uspto.py [--smiles] [--jobs N]
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import py7zr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from uspto_to_lproc import convert_files  # noqa: E402

USPTO = os.path.join(ROOT, "data", "uspto")
ARCHIVE = os.path.join(USPTO, "grants_cml.7z")
EXTRACT = os.path.join(USPTO, "extracted")


def out_paths(smiles: bool):
    tag = "corpus_smiles" if smiles else "corpus_grants"
    return (os.path.join(USPTO, f"{tag}_years"),
            os.path.join(USPTO, f"{tag}_full.jsonl"),
            os.path.join(USPTO, f"{tag}_full_dedup.jsonl"),
            os.path.join(USPTO, f"{tag}_full_stats.json"))


def convert_year(ydir, out, smiles):
    xmls = sorted(glob.glob(os.path.join(ydir, "*.xml")))
    stats = convert_files(xmls, outdir=None, jsonl_path=out + ".part",
                          smiles=smiles)
    os.replace(out + ".part", out)
    return stats


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--keep-xml", action="store_true",
                    help="do not delete extracted XML after conversion")
    args = ap.parse_args()
    YEARS_DIR, FULL, DEDUP, STATS = out_paths(args.smiles)
    os.makedirs(YEARS_DIR, exist_ok=True)
    grants_root = os.path.join(EXTRACT, "grants")

    if not os.path.isdir(grants_root) or len(os.listdir(grants_root)) < 41:
        log("extracting full archive (~14 GB, one pass)...")
        with py7zr.SevenZipFile(ARCHIVE) as z:
            z.extractall(path=EXTRACT)
        log("extraction done")

    years = sorted(d for d in os.listdir(grants_root)
                   if os.path.isdir(os.path.join(grants_root, d)))
    total = None
    per_year = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {}
        for year in years:
            out = os.path.join(YEARS_DIR, f"{year}.jsonl")
            if os.path.exists(out):
                log(f"{year}: already converted, skipping")
                continue
            futs[pool.submit(convert_year, os.path.join(grants_root, year),
                             out, args.smiles)] = year
        for fut in as_completed(futs):
            year = futs[fut]
            stats = fut.result()
            per_year[year] = stats
            total = total or {k: 0 for k in stats}
            for k in stats:
                total[k] += stats[k]
            log(f"{year}: {stats['accepted']}/{stats['reactions']} accepted "
                f"(cumulative {total['accepted']})")
            if not args.keep_xml:
                shutil.rmtree(os.path.join(grants_root, year), ignore_errors=True)

    log("merging and deduplicating...")
    seen = set()
    n_all = n_uniq = 0
    with open(FULL, "w", encoding="utf-8") as f_all, \
         open(DEDUP, "w", encoding="utf-8") as f_uniq:
        for year in years:
            path = os.path.join(YEARS_DIR, f"{year}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    f_all.write(line)
                    n_all += 1
                    rec = json.loads(line)
                    key = rec.get("product", "") + "\t" + rec["canonical"]
                    h = hashlib.sha1(key.encode()).digest()
                    if h not in seen:
                        seen.add(h)
                        f_uniq.write(line)
                        n_uniq += 1

    summary = {"total": total, "per_year": per_year,
               "merged_lines": n_all, "unique_canonical": n_uniq,
               "elapsed_s": round(time.time() - t0)}
    with open(STATS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log(f"DONE: {n_all} documents, {n_uniq} unique "
        f"({100 * n_uniq / max(1, n_all):.1f} %), "
        f"{round(time.time() - t0) // 60} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
