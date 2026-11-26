"""Prepare independently annotated SATD comments with auditable source alignment."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from comment_spans import LEXEME
from fetch_inputs import fetch

DATA = HERE / "data"
TAG = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b", re.IGNORECASE | re.ASCII)


@dataclass(frozen=True)
class Example:
    identifier: str
    group: str
    cluster: str
    text: str
    start: int
    end: int
    label: bool

    @property
    def comment(self):
        return self.text[self.start:self.end]


def normalize(text):
    return " ".join(text.casefold().split())


def comments(text):
    return [(match.start(), match.end()) for match in LEXEME.finditer(text)
            if match[0].startswith(("//", "/*"))]


def cppsatd(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            prefix = row["preceding_code"].rstrip() + "\n"
            comment = row["commenttext"].strip()
            text = prefix + comment + "\n" + row["succeeding_code"]
            project = row["projectname"]
            yield Example(f"{project}/{row['comment_id']}", project,
                          f"{project}/{row['file_id']}", text, len(prefix), len(prefix)+len(comment),
                          row["manual_annotation"] != "NON-SATD")


def isolate_comment(text, start, end):
    """Keep the labelled Java comment and blank out the others, preserving line breaks."""
    target_found = False
    pieces = []
    cursor = 0
    new_start = start
    for first, last in comments(text):
        if first < end and last > start:
            if first < start or last > end:
                return None
            target_found = True
            continue
        replacement = " " + "\n" * text[first:last].count("\n")
        pieces.extend((text[cursor:first], replacement))
        cursor = last
        if last <= start:
            new_start -= last-first-len(replacement)
    if not target_found:
        return None
    pieces.append(text[cursor:])
    return "".join(pieces), new_start


def tesoro(path, isolate=False):
    for row in pq.read_table(path).to_pylist():
        text, comment = row["code"], row["comment"].strip()
        start = text.find(comment)
        if not row["isFinished"] or start < 0 or not comment or text.find(comment, start+len(comment)) != -1:
            yield None
            continue
        if isolate:
            aligned = isolate_comment(text, start, start+len(comment))
            if aligned is None:
                yield None
                continue
            text, start = aligned
            if text[start:start+len(comment)] != comment:
                raise ValueError("Removing other comments changed the annotated span")
        function = str(row["id"])
        yield Example(f"{function}/{row['comment_id']}", function, function,
                      text, start, start+len(comment), row["classification"] != "NONSATD")


def load_source(name, isolate_tesoro=False):
    record = json.loads((DATA / "manifest.json").read_text())[name]
    path = DATA / record["file"]
    fetch(path, record["url"], record["sha256"])
    stats = Counter()
    rows = defaultdict(list)
    dataset = cppsatd(path) if name == "cppsatd" else tesoro(path, isolate=isolate_tesoro)
    for entry in dataset:
        stats["source_rows"] += 1
        if entry is None:
            stats["target_not_unique_verbatim"] += 1
            continue
        if TAG.search(entry.text):
            stats["any_mat_tag_in_input"] += 1
            continue
        spans = comments(entry.text)
        if not spans or any(start < entry.start or end > entry.end for start, end in spans):
            stats["unaligned_or_other_comment"] += 1
            continue
        label_surface = normalize(entry.comment)
        if not label_surface:
            stats["missing_comment"] += 1
            continue
        rows[label_surface].append(entry)
    samples = []
    for candidates in rows.values():
        if len({entry.label for entry in candidates}) != 1:
            stats["conflicting_duplicate_comments"] += len(candidates)
            continue
        samples.append(candidates[0])
        stats["duplicate_comments_removed"] += len(candidates)-1
    stats["retained"] = len(samples)
    stats["positive"] = sum(entry.label for entry in samples)
    if len({entry.identifier for entry in samples}) != len(samples):
        raise ValueError(f"IDs are ambiguous in {name}")
    return samples, stats


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()
