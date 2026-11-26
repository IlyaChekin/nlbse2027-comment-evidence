"""Cache versioned rule counts for held-out projects."""
import argparse
import csv
import os
from pathlib import Path
import time

import numpy as np

from runtime import scanner
from stream import selection
from replay import HERE, checksum, rule_data, write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split",choices=("heldout","train","public"))
    p.add_argument("--prefix",type=Path)
    args = p.parse_args()
    texts,rows = selection(args.split)
    start = time.perf_counter()
    prefix = None
    rest = rows
    if args.prefix:
        old_keys, prefix, past = rule_data(args.prefix)
        if not np.isin(prefix,rows).all():
            raise ValueError("Resumed rule rows are not part of this split")
        rest = np.setdiff1d(rows,prefix)
    keys, fresh = scanner([texts[i] for i in rest])
    values = fresh
    if prefix is not None:
        if keys != old_keys:
            raise ValueError("Partial rule scan reports different columns")
        order = np.argsort(np.r_[prefix,rest])
        values = np.concatenate((past,fresh))[order]
    elapsed = time.perf_counter()-start
    target = HERE/f"{args.split}_parallel.csv"
    temporary = target.with_suffix(".part")
    with temporary.open("w",newline="") as f:
        w = csv.writer(f)
        w.writerow(("row",*keys))
        w.writerows((i,*v) for i,v in zip(rows,values,strict=True))
    temporary.replace(target)
    write_json(target.with_suffix(".json"),dict(rows=len(rows), scanned_now=len(rest), seconds=elapsed, sha256=checksum(target),
               code_sha256=checksum(HERE/"scan.mjs"), source_sha256=checksum(HERE/"rules_manifest.json"),
               worker_limit=int(os.environ.get("NLBSE_SCANNER_WORKERS","6")),
               prefix_sha256=checksum(args.prefix) if args.prefix else None,
               columns=keys))
    print(args.split,len(rows),elapsed,flush=True)
