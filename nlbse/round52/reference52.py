"""Test MAT references to functions mentioned in tagged comments of other training files."""
from collections import defaultdict
import csv
from pathlib import Path
import re

import numpy as np

from mat52 import HERE, ROOT, LEXEME, PARTS, declaration, cutoff, table, dataset, split, checksum, write_json
from transfer import upstream_tag

DEST = HERE / 'references'
NAME = re.compile(r'\b[A-Za-z_][A-Za-z_0-9]{5,}\b', re.ASCII)
CALL = re.compile(r'\b([A-Za-z_][A-Za-z_0-9]{5,})\s*\(', re.ASCII)
REPAIR = re.compile(r'\b(fix|repair|replace|remove|refactor|implement|rewrite|update|finish)\b', re.I)
NAMED = re.compile(r'[_A-Z]')


def parse(text):
    header, comments = [], []
    body = False
    for token in LEXEME.finditer(text):
        if token['comment'] is not None:
            comments.append(token[0])
        elif not body:
            body = token[0] == '{'
            if not body:
                header.append(token[0])
    return declaration(header)[0], comments


def annotated(comments):
    strong, exact, weak = set(), set(), set()
    for block in comments:
        if not upstream_tag(block):
            continue
        for line in block.splitlines():
            if not upstream_tag(line):
                continue
            exact.update(CALL.findall(line))
            words = {x for x in NAME.findall(line) if NAMED.search(x[1:]) and not x.isupper()}
            weak.update(words)
            repair = REPAIR.search(line)
            if repair:
                strong.update(words.intersection(NAME.findall(line[repair.end():repair.end() + 60])))
    return dict(tagged_call=exact, tagged_named=weak, repair_named=strong)


def develop(fit, texts, gold, project, files):
    docs = []
    marked = defaultdict(lambda: defaultdict(set))
    for i in fit:
        name, comments = parse(texts[i])
        source = project[i], files[i]
        if not source[1]:
            continue
        for c in comments:
            docs.append((source, name, c))
        if not gold[i]:
            continue
        for mode, mentions in annotated(comments).items():
            for m in mentions - {name}:
                marked[mode][m].add(source)
    names = {k for row in marked.values() for k in row}
    untagged = defaultdict(set)
    for source, name, text in docs:
        if not upstream_tag(text):
            for m in (set(NAME.findall(text)) & names) - {name}:
                untagged[m].add(source)
    index = {mode: {name: (len(sources), len(untagged[name])) for name, sources in pairs.items()}
             for mode, pairs in marked.items()}
    return index


def main():
    DEST.mkdir(exist_ok=True)
    texts, label, projects, files = dataset()
    gold = label[:, 1].astype(bool)
    folds = split()
    fit = folds['fit']
    assert not set(projects[fit]) & set(projects[np.r_[folds['test_a'], folds['test_b']]])
    bank = develop(fit, texts, gold, projects, files)
    assert set(bank) == {'tagged_call', 'tagged_named', 'repair_named'}
    book = DEST / 'index.json'
    write_json(book, bank)
    thresholds, reports = {}, {}
    for part in PARTS:
        source = HERE / 'contract' / f'{part}_predictions.npz'
        ids = folds[part]
        with np.load(source) as ref:
            assert np.array_equal(ref['ids'], ids) and np.array_equal(ref['y'], gold[ids])
            y, base, comment = ref['y'], ref['base'], ref['comment']
        names = [parse(texts[i])[0] for i in ids]
        result, coverage = {}, {}
        for arm, stats in bank.items():
            count, opposing = np.asarray([stats.get(n, (0, 0)) for n in names], dtype=np.int32).T
            value = np.log1p(count) / (1. + np.log1p(opposing))
            result[arm] = value
            result[arm + '_exclusive'] = np.where(opposing == 0, value, 0.)
            coverage[arm] = dict(names=len(stats), misses_with_reference=int((y & ~base & (count > 0)).sum()),
                                 negatives_with_reference=int((~y & ~base & (count > 0)).sum()))
        if part == 'dev':
            thresholds = {k: cutoff(y, base, v) for k, v in result.items()}
            write_json(DEST / 'policy.json', dict(cutoffs=thresholds, index_sha256=checksum(book),
                       train_manifest_sha256=checksum(ROOT / 'data/manifest.json'),
                       partition_sha256=checksum(ROOT / 'round13/partition.npz'),
                       code_sha256=checksum(Path(__file__))))
        report = table(part, ids, y, projects[ids], result, comment, base, thresholds, dest=DEST)
        reports[part] = report | {'coverage': coverage, 'reference_sha256': checksum(source)}
        print(part, coverage, {k: [v['matched']['recovered'], v['frozen']['recovered'], v['frozen']['delta_fp']]
                              for k, v in report['results'].items()}, flush=True)
    write_json(DEST / 'results.json', reports)
    with (DEST / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('partition', 'arm', 'same_fp_tp', 'frozen_tp', 'frozen_fp'))
        for part, report in reports.items():
            for arm, value in report['results'].items():
                writer.writerow((part, arm, value['matched']['recovered'], value['frozen']['recovered'],
                                 value['frozen']['delta_fp']))


if __name__ == '__main__':
    main()
