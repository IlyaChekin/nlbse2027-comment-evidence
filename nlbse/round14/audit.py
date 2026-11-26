"""Audit MAT marker evidence in the supplied function without exporting source text."""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "round13"), str(ROOT / "round12"), str(ROOT)]
from rotate import read_partition
from project_split import family, checksum
from comment_spans import LEXEME
from pipeline import PUBLIC, TRAIN, load, write_json


TAG = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b", re.I | re.ASCII)
STEM = re.compile(r"TODO|FIXME|XXX|HACK", re.I | re.ASCII)
LOOSE_COMMENT = re.compile(r"//[^\r\n]*|/\*.*?(?:\*/|\Z)", re.S)
VARIANT = re.compile(r"\b(?:to[ \t_-]+do|fix[ \t_-]+me|x[ \t_-]+x[ \t_-]+x)\b", re.I | re.ASCII)
DEBT = re.compile(r"\b(?:kludge|work[ -]?around|temporary|tbd|broken|not implemented|stub|needs? (?:fixing|to be fixed))\b", re.I)
A = ("known_tag", "known_tag_later", "stem_in_comment", "stem_head_in_comment", "stem_upper_in_comment",
     "split_tag_in_comment", "debt_phrase", "literal_or_code_tag", "loose_comment_tag", "any_comment",
     "leading_comment", "preprocessor_stub", "comment_spacing_variant", "concat_word_tag")


def flags(text):
    comments = [m for m in LEXEME.finditer(text) if m[0].startswith(("//", "/*"))]
    tags = [(m, tag) for m in comments for tag in TAG.finditer(text, *m.span())]
    stems = [(m, tag) for m in comments for tag in STEM.finditer(text, *m.span())]
    any_tag = TAG.search(text) is not None
    return (bool(tags), any(tag.end() > 1024 for _, tag in tags), bool(stems),
            any(not text[m.start()+2:tag.start()].strip("* \t\r\n") for m, tag in stems),
            any(tag[0].isupper() for _, tag in stems), any(VARIANT.search(m[0]) for m in comments),
            any(DEBT.search(m[0]) for m in comments), any_tag and not tags,
            any_tag and any(TAG.search(text, *m.span()) for m in LOOSE_COMMENT.finditer(text)), bool(comments),
            bool(comments and not text[:comments[0].start()].strip()),
            re.search(r"(?m)^\s*#\s*(?:if\s+0\b|error\b|warning\b)", text) is not None,
            any(re.search(r"(?i)\b(?:fix|to|x)\s*[-_ ]+\s*(?:me|do|x\s*[-_ ]*x)\b", m[0]) for m in comments),
            any(tag.end() < m.end() and text[tag.end()].isdigit() for m, tag in stems))


@lru_cache(maxsize=2)
def dataset(public=False):
    path = PUBLIC if public else TRAIN
    key = "test" if public else "train"
    manifest = json.loads((ROOT / "data/manifest.json").read_text())["files"]
    expected = next(r["sha256"] for r in manifest if r["split"] == key)
    if checksum(path) != expected:
        raise ValueError(f"Dataset changed: {key}")
    x, label, meta = load(path, audit=True)
    project = np.asarray([family(p) for p in meta["Projectname"]])
    return x, label, project, meta["Filepath"]


def split():
    ids = read_partition()
    _, _, project, _ = dataset()
    dev = ids["dev"]
    mask = project[dev] == "tcpdump"
    return {"fit": ids["fit"], "dev": np.sort(np.r_[ids["probe"], dev[mask]]),
            "test_a": dev[~mask], "test_b": ids["test"]}


def scan(texts):
    return np.asarray([flags(t) for t in texts], dtype=bool)


def describe(texts, gold, x):
    p = gold == 1
    base = x[:, 0]
    fn = p & ~base
    c = {"rows": len(gold), "mat_positive": int(p.sum()), "tp": int((p & base).sum()),
         "fp": int((~p & base).sum()), "fn": int(fn.sum())}
    c["false_negative_scope"] = dict(Counter("comment_without_base_tag" if parsed else "no_parsed_comment"
                                                 for parsed in x[fn, A.index("any_comment")]))
    c["missed_evidence"] = {name: {"fn": int((fn & x[:, col]).sum()), "other_negative": int((~p & ~base & x[:, col]).sum())}
                            for col, name in enumerate(A)}
    c["miss_chars_quartiles"] = np.quantile([len(t) for t, missed in zip(texts, fn, strict=True) if missed], [.25, .50, .75]).tolist() if fn.any() else []
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("discovery", "test", "public"))
    stage = parser.parse_args().stage
    texts, label, project, _ = dataset(stage == "public")
    indices = {"public": np.arange(len(texts))} if stage == "public" else split()
    parts = {"discovery": ("fit", "dev"), "test": ("test_a", "test_b"), "public": ("public",)}[stage]
    output = {}
    for name in parts:
        ids = indices[name]
        local = [texts[i] for i in ids]
        x = scan(local)
        output[name] = describe(local, label[ids, 1], x)
        output[name]["projects"] = dict(Counter(project[ids].tolist()))
        np.savez_compressed(HERE / f"evidence_{name}.npz", ids=ids, flags=x)
        print(name, output[name], flush=True)
    write_json(HERE / f"audit_{stage}.json", output)


if __name__ == "__main__":
    main()
