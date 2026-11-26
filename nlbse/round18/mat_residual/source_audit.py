"""Offline provenance audit against cited source snapshots, separate from prediction."""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from residual import HERE, LEXEME, TAG, dataset, example, split, write_json
from project_split import checksum


CHROMIUM = "57f97b28b5e571f2a1c9ad54398c8efa6e1043b2"
LINUX = "e2ca6ba6ba0152361aa4fcbf6067db71b2c7a770"
SOURCES = {
    "chromium": (
        ("https://raw.githubusercontent.com/chromium/chromium/" + CHROMIUM + "/", ""),
        ("https://chromium.googlesource.com/chromium/src/+/" + CHROMIUM + "/", "?format=TEXT"),
    ),
    "mozilla firefox": (
        ("https://hg-edge.mozilla.org/mozilla-central/raw-file/4d46db3ff28b/", ""),
        ("https://hg.mozilla.org/mozilla-central/raw-file/4d46db3ff28b/", ""),
    ),
    "linux": (
        ("https://raw.githubusercontent.com/torvalds/linux/" + LINUX + "/", ""),
        ("https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/" + LINUX + "/", "?format=TEXT"),
    ),
}
LEXICAL = re.compile(r"\w+|[^\s\w]", re.ASCII)


def stream(text):
    tokens, comments = [], []
    cursor = 0
    for part in LEXEME.finditer(text):
        tokens.extend((m[0], m.start(), m.end()) for m in LEXICAL.finditer(text, cursor, part.start()))
        if part[0].startswith(("//", "/*")):
            comments.append(part)
        else:
            tokens.append((part[0], part.start(), part.end()))
        cursor = part.end()
    tokens.extend((m[0], m.start(), m.end()) for m in LEXICAL.finditer(text, cursor))
    return tokens, comments


def locate(tokens, hay, starts, query):
    signature = tuple(token[0] for token in query)
    if len(signature) < 12:
        return []
    return [(tokens[i][1], tokens[i+len(signature)-1][2], tokens[i-1][2] if i else 0)
            for i in starts[signature[0]] if hay[i:i+len(signature)] == signature]


def placement(text, comments, regions):
    if len(regions) != 1:
        return "multiple_token_matches" if regions else "no_token_match"
    begin, end, prior_token_end = regions[0]
    contained = [m for m in comments if begin <= m.start() < end]
    if any(TAG.search(m[0]) for m in contained):
        return "marker_inside_matched_span"
    leading = [m for m in comments if prior_token_end <= m.start() and m.end() <= begin]
    for comment in reversed(leading):
        if text[comment.end():begin].strip() or text[comment.end():begin].count("\n") > 2:
            break
        line_start = text.rfind("\n", 0, comment.start())+1
        if not text[line_start:comment.start()].strip() and TAG.search(comment[0]):
            return "marker_in_adjacent_preceding_comment"
        begin = comment.start()
    return "match_without_adjacent_marker"


def fetch(family, filepath):
    # Only retrieve relative source paths from the published repositories at the cited snapshots.
    path = PurePosixPath(filepath)
    if path.is_absolute() or ".." in path.parts:
        return None, {"status": "invalid_relative_path"}
    errors = {}
    for host, suffix in SOURCES[family]:
        address = host + quote(path.as_posix(), safe="/") + suffix
        try:
            request = Request(address, headers={"User-Agent": "NLBSE-source-provenance-audit"})
            with urlopen(request, timeout=7) as response:
                raw = response.read(6_000_001)
                actual_url = response.url
            if len(raw) > 6_000_000:
                return None, {"status": "oversize", "url": address}
            if suffix == "?format=TEXT":
                raw = base64.b64decode(raw)
            text = raw.decode("utf-8", errors="surrogateescape")
            return text, {"status": "retrieved", "url": actual_url,
                          "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        except HTTPError as error:
            errors[address] = int(error.code)
        except (URLError, OSError) as error:
            errors[address] = type(error).__name__
    return None, {"status": "unavailable", "responses": errors}


def candidates(texts, labels, projects, files, public):
    groups = {"public": np.arange(len(texts))} if public else split()
    entries = {}
    pairs = [("public", p) for p in SOURCES] if public else [
        ("dev", "linux"), ("test_a", "chromium"), ("test_b", "mozilla firefox")]
    for role, family in pairs:
        ids = groups[role][projects[groups[role]] == family]
        by_file = defaultdict(list)
        positives = []
        for row in ids:
            base, legacy, _, _ = example(texts[row], code_required=False)
            if base or legacy["has_comment"]:
                continue
            if labels[row, 1]:
                positives.append(int(row))
            else:
                by_file[files[row]].append(int(row))
        for row in positives:
            group = entries.setdefault((family, files[row]), {})
            group[row] = True
            for ctrl in by_file[files[row]]:
                group[ctrl] = False
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    texts, labels, projects, files = dataset(args.public)
    entries = candidates(texts, labels, projects, files, args.public)
    def audit_one(item):
        (family, f), cases = item
        source, receipt = fetch(family, f)
        records = []
        if source is not None:
            tokens, comments = stream(source)
            hay = tuple(t[0] for t in tokens)
            starts = defaultdict(list)
            for i, token in enumerate(hay):
                starts[token].append(i)
        for row, positive in sorted(cases.items()):
            if source is None:
                outcome = "source_unavailable"
            else:
                query, _ = stream(texts[row])
                outcome = placement(source, comments, locate(tokens, hay, starts, query))
            records.append(dict(project=family, row=row, mat_positive=positive, observation=outcome))
        return dict(project=family, file=f, **receipt), records

    summary = defaultdict(Counter)
    files_found, observations = [], []
    with ThreadPoolExecutor(max_workers=8) as workers:
        for source, records in workers.map(audit_one, sorted(entries.items())):
            files_found.append(source)
            observations.extend(records)
            for value in records:
                key = value["project"] + ("/missed_positive" if value["mat_positive"] else "/same_file_control")
                summary[key][value["observation"]] += 1
    output = dict(sources=SOURCES, totals=summary, source_receipts=files_found,
                  observations=observations, code_sha256=checksum(Path(__file__)),
                  partition_sha256=checksum(HERE.parents[1]/"round13/partition.npz"))
    write_json(HERE/("source_audit_public.json" if args.public else "source_audit.json"), output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
