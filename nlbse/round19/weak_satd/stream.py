"""Stream only checked function text and row IDs to the passive detector."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/"round14"))
from audit import dataset


def selection(split):
    texts, _, _, _ = dataset(split=="public")
    ids = np.arange(len(texts))
    if split == "heldout":
        parts = []
        for group in ("test_a","test_b"):
            with np.load(HERE.parent/f"{group}.npz") as f:
                parts.extend((f["dev"],f["test"]))
        ids = np.unique(np.concatenate(parts))
    return texts,ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("train","heldout","public"))
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    texts, ids = selection(args.split)
    for row in ids[:args.limit]:
        code = texts[row]
        row = int(row)
        sys.stdout.write(json.dumps((row,code),separators=(",",":"))+"\n")
