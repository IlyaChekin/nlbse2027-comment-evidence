"""Check formatting sensitivity of the existing static classification features."""
import argparse
import csv
from pathlib import Path
import re
import time

import numpy as np

from project_eval import HERE, PARTS, checksum, json_at, raw_values, write_json, f
from comment_spans import LEXEME

EXCLUDED = re.compile(r"(?m)^\s*#|\\[ \t]*\r?\n")
LAYOUT = re.compile(r"\(|\)|[ \t]*\r?\n[ \t]*")
BRACE = re.compile(r"\{[ \t]*(?=[^\s}])")
MODES = ("join_parentheses","break_braces")
PROGRAM = f.HERE/"scan_extended.mjs"


def reflow(text,mode):
    if EXCLUDED.search(text):
        return text
    pieces = []
    cursor,depth = 0,0
    def transform(chunk, protect_first_line=False):
        if mode=="break_braces":
            return BRACE.sub("{\n",chunk)
        def match_layout(match):
            nonlocal depth
            token = match[0]
            if token=="(":
                depth += 1
            elif token==")":
                depth = max(0,depth-1)
            elif depth and not (protect_first_line and not chunk[:match.start()].strip()):
                return " "
            return token
        return LAYOUT.sub(match_layout,chunk)
    protected = False
    for span in LEXEME.finditer(text):
        pieces.extend((transform(text[cursor:span.start()],protected),span[0]))
        cursor = span.end()
        protected = span[0].startswith("//")
    pieces.append(transform(text[cursor:],protected))
    return "".join(pieces)


def subsets(pilot):
    texts,y,projects,_,_ = f.inputs()
    root = f.R19
    repo = json_at(root/"partition.json")
    if checksum(f.ROOT/"data/raw/train-00000-of-00001.parquet")!=repo["train_sha256"]:
        raise ValueError("Versioned training data changed")
    rng = np.random.default_rng(42)
    data = {}
    for r in PARTS:
        index = repo["rotations"][r]
        p = root/index["partition_file"]
        if checksum(p)!=index["partition_sha256"]:
            raise ValueError("A rotation changed")
        with np.load(p) as z:
            for role in ("dev",) if pilot else ("dev","test"):
                ids = z[role]
                data[f"{r}/{role}"] = np.sort(rng.choice(ids,min(6000,len(ids)),replace=False)) if pilot else ids
    return texts,y[:,0],projects,data


def main(pilot):
    texts,gold,projects,parts = subsets(pilot)
    rows = np.unique(np.concatenate(list(parts.values())))
    old = raw_values(rows)
    y = gold[rows].astype(bool)
    hit = f.valid(old) & f.weak(old)
    available = f.valid(old)
    result = {}
    out = HERE/"reflow"/("pilot" if pilot else "full")
    out.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        t = time.perf_counter()
        # Scan only changed, previously negative functions for the recovery experiment.
        changed = []
        scan_text = []
        for i in np.flatnonzero(available & ~hit):
            original = texts[rows[i]]
            modified = reflow(original,mode)
            if modified!=original:
                changed.append(i)
                scan_text.append(modified)
        cols, raw = f.scanner(scan_text,program=PROGRAM)
        source_cols = json_at(f.SOURCE/"policy.json")["rule_columns"] + ["rule_195"]
        if cols!=source_cols:
            raise ValueError("Reflow scan used a different set of rules")
        signals = np.zeros_like(old)
        signals[changed] = raw
        added = f.weak(signals)
        details = {}
        for key, ids in parts.items():
            role = np.isin(rows,ids)
            stats = {}
            for project in ("all",*sorted(set(projects[rows[role]]))):
                flag = role if project=="all" else role & (projects[rows]==project)
                stats[project] = dict(
                    cohort=int((flag & available & ~hit).sum()),
                    positive=int((flag & available & ~hit & y).sum()),
                    new_tp=int((flag & added & y).sum()),
                    new_fp=int((flag & added & ~y).sum()),
                )
            details[key] = stats
        timing = time.perf_counter()-t
        result[mode] = dict(scanned=len(changed),wall_seconds=timing,partitions=details,
                            column_hits=dict(zip(cols,(signals>0).sum(0).tolist(),strict=True)))
        np.savez_compressed(out/f"{mode}.npz",rows=rows,changed=np.asarray(changed),raw=signals)
        print(mode,len(changed),round(timing,3),{k:v["all"] for k,v in details.items()},flush=True)
    write_json(out/"results.json",dict(rows=len(rows),results=result,code_sha256=checksum(Path(__file__)),
                                       scanner_sha256=checksum(PROGRAM),
                                       partition_sha256=checksum(f.R19/"partition.json")))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",action="store_true")
    main(p.parse_args().pilot)
