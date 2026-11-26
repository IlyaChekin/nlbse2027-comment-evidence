"""Audit omitted tags even when other comments remain visible in the function."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sys

import numpy as np

from neighbors import HERE, ROOT, checksum, dataset, write_json

sys.path.insert(0, str(ROOT/'round18/mat_residual'))
from source_audit import SOURCES, TAG, fetch, locate, placement, stream


def inputs(split):
    if split == 'public':
        file = HERE/'public/canonical.csv'
        with file.open() as handle:
            for row in csv.DictReader(handle):
                yield int(row['row']), bool(int(row['label_mat'])), bool(int(row['pred_MAT']))
    else:
        with np.load(HERE/f'{split}_predictions.npz') as z:
            yield from zip(z['ids'], z['y'], z['canonical'], strict=True)


def examine(snapshot):
    (project, path), queries = snapshot
    source, receipt = fetch(project, path)
    output = []
    if source is not None:
        tokens, comments = stream(source)
        hay = tuple(t[0] for t in tokens)
        starts = defaultdict(list)
        for i, token in enumerate(hay):
            starts[token].append(i)
    for key, row, gold, query, tagged in queries:
        if source is None:
            result = 'source_unavailable'
        else:
            result = placement(source, comments, locate(tokens, hay, starts, query))
        output.append(dict(split=key, row=int(row), gold=bool(gold), project=project,
                           observation=result, whole_word_tag_visible=tagged))
    return dict(project=project, path=path, **receipt), output


def main():
    visible = {}
    groups = defaultdict(list)
    for side in ('test_a', 'test_b', 'public'):
        text, _, project, files = dataset(side == 'public')
        cases = Counter()
        positives = defaultdict(list)
        for i, y, found in inputs(side):
            if not y or found:
                continue
            query, comments = stream(text[i])
            if not comments:
                continue
            cases['all_misses_with_lexical_comments'] += 1
            p = project[i]
            if p not in SOURCES:
                cases['family_has_no_pinned_snapshot'] += 1
                continue
            positives[(p, files[i])].append((side, i, y, query, any(TAG.search(m[0]) for m in comments)))
        for key, entries in positives.items():
            groups[key].extend(entries)
        for i, y, found in inputs(side):
            key = (project[i], files[i])
            if y or found or key not in positives:
                continue
            query, comments = stream(text[i])
            if comments:
                groups[key].append((side, i, y, query, any(TAG.search(m[0]) for m in comments)))
        visible[side] = cases
    hist = defaultdict(Counter)
    details, receipts = [], []
    with ThreadPoolExecutor(max_workers=6) as workers:
        for receipt, tries in workers.map(examine, sorted(groups.items())):
            receipts.append(receipt)
            details.extend(tries)
            for r in tries:
                name = f"{r['split']}/" + ('miss' if r['gold'] else 'control')
                tagged = r['observation']
                hist[name][tagged] += 1
                if not r['whole_word_tag_visible'] and tagged.startswith('marker_'):
                    hist[name]['different_marker_absent_from_visible_comments'] += 1
    output = dict(scope='provenance_only', visible=visible, outcome=hist, source_receipts=receipts,
                  observations=details, code_sha256=checksum(Path(__file__)))
    write_json(HERE/'visible_provenance.json', output)
    print('visible:', visible, 'outcome:', hist, flush=True)


if __name__ == '__main__':
    main()
