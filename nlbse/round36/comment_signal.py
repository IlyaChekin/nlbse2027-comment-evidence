"""Learn the wording around MAT markers and test transfer with the markers absent."""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'round31/mat'))
from transfer import canonical_mat, checksum, dataset, split, write_json, TOKEN
from placement import cutoff, table
from marker import upstream_tag

PARTS = ('dev', 'test_a', 'test_b')
WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+", re.I | re.ASCII)
BOUNDARY = re.compile(r'\r?\n|(?<=[.!?;])\s+')
SPLIT_MARKER = re.compile(r'(?i)\b(?:to[ _.-]+do|fix[ _.-]+me|x[ _.-]+x[ _.-]+x)\b')
MISSING = -100.
VECTORIZERS = (
    HashingVectorizer(n_features=2**17, analyzer='word', ngram_range=(1, 2),
                      token_pattern=r"\b[a-z0-9]+(?:'[a-z]+)?\b", dtype=np.float32),
    HashingVectorizer(n_features=2**17, analyzer='char_wb', ngram_range=(3, 5), dtype=np.float32),
)


def erase(text):
    words = []
    for item in SPLIT_MARKER.sub(' ', text).split():
        if not upstream_tag(item):
            words.extend(x.casefold() if not x.isdigit() else 'number' for x in WORDS.findall(item))
    if len(words) >= 3:
        return ' '.join(words[:80])
    return ''


def units(text, mode):
    for token in TOKEN.finditer(text):
        if token['comment'] is None:
            continue
        span = token[0]
        marked = upstream_tag(span)
        chunks = BOUNDARY.split(span) if mode == 'clause' else [span]
        for chunk in chunks:
            prompt = erase(chunk)
            if prompt:
                yield prompt, marked and upstream_tag(chunk)


def matrix(snippets, mode):
    if mode == 'word':
        return VECTORIZERS[0].transform(snippets)
    return hstack([v.transform(snippets) for v in VECTORIZERS], format='csr')


def collect(texts, rows, y, base, unit):
    snippets, owners, targets, context = [], [], [], []
    for j, i in enumerate(rows):
        # A true natural miss has no attributable span, so it cannot supervise this classifier.
        if y is not None and y[i] and not base[j]:
            continue
        for snippet, marked in units(texts[i], unit):
            snippets.append(snippet)
            owners.append(j)
            targets.append(marked and y is not None and y[i])
            context.append(y is not None and y[i])
    return snippets, np.asarray(owners, dtype=int), np.asarray(targets, bool), np.asarray(context, bool)


def train(texts, ids, gold, base):
    heads, info = {}, {}
    for unit in ('comment', 'clause'):
        snippets, owners, target, nearby = collect(texts, ids, gold, base, unit)
        multiplicity = np.bincount(owners)[owners]
        weights = np.where(target, 14., np.where(nearby, .5, 1.)) / np.sqrt(multiplicity)
        info[unit] = dict(units=len(owners), positive_units=int(target.sum()),
                          positive_functions=int(np.unique(owners[target]).size),
                          same_function_negatives=int((nearby & ~target).sum()))
        for mode in ('word', 'both'):
            x = matrix(snippets, mode)
            head = LogisticRegression(solver='liblinear', C=1., max_iter=150, random_state=42)
            heads[f'{unit}_{mode}'] = head.fit(x, target, sample_weight=weights)
        print(unit, info[unit], flush=True)
    return heads, info


def scores(texts, heads):
    out = {}
    rows = np.arange(len(texts))
    for unit in ('comment', 'clause'):
        snippets, owner, _, _ = collect(texts, rows, None, None, unit)
        for view in ('word', 'both'):
            key = f'{unit}_{view}'
            score = np.full(len(rows), MISSING)
            if snippets:
                values = heads[key].decision_function(matrix(snippets, view))
                np.maximum.at(score, owner, values)
            out[key] = score
    return out


def coverage(texts, gold, base, available, ranking, projects):
    value = {}
    for family in np.unique(projects):
        local = projects == family
        positive = local & gold & ~base
        negative = local & ~gold & ~base
        scope = available & positive
        first = {}
        for key, score in ranking.items():
            viable = positive & (score > MISSING)
            first[key] = (int((negative & (score >= score[viable].max())).sum())
                          if viable.any() else None)
        value[str(family)] = dict(rows=int(local.sum()), misses=int(positive.sum()),
                                  misses_with_comments=int(scope.sum()), fp_at_first_recovery=first)
    return value


def main():
    texts, labels, projects, _ = dataset()
    gold = labels[:, 1].astype(bool)
    ids = split()
    fit = ids['fit']
    base = canonical_mat([texts[i] for i in fit])
    heads, details = train(texts, fit, gold, base)
    model = HERE/'heads.joblib'
    joblib.dump(heads, model, compress=3)
    frozen, reports = {}, {}
    for part in PARTS:
        start = perf_counter()
        rows = ids[part]
        local = [texts[i] for i in rows]
        base = canonical_mat(local)
        roster = ROOT/'round31/mat/paired_provenance'/f'{part}_predictions.npz'
        with np.load(roster, allow_pickle=True) as reference:
            if not (np.array_equal(reference['ids'], rows) and np.array_equal(reference['base'], base)
                    and np.array_equal(reference['y'], gold[rows])):
                raise ValueError(f'{part}: reference predictions changed')
            comments = reference['comment']
        # Tagged functions keep their prediction and never need the extra pass at inference.
        active = np.flatnonzero(~base)
        rank = {}
        for key, value in scores([local[i] for i in active], heads).items():
            rank[key] = np.full(len(rows), MISSING)
            rank[key][active] = value
        if part == 'dev':
            frozen = {k:cutoff(gold[rows], base, v) for k,v in rank.items()}
            write_json(HERE/'policy.json', dict(cutoffs=frozen, arms=list(rank), training=details,
                       model_sha256=checksum(model), code_sha256=checksum(Path(__file__)),
                       partition_sha256=checksum(ROOT/'round13/partition.npz'),
                       data_manifest_sha256=checksum(ROOT/'data/manifest.json'),
                       runtime_sha256=checksum(ROOT/'round26/mat/mat_runtime.py'),
                       features='word and character grams of marker-erased comment clauses or whole spans',
                       eval_labels_used_for_threshold='dev only', functions_only=True))
        result = table(part, rows, gold[rows], projects[rows], rank, comments, base, frozen, dest=HERE)
        result['coverage'] = coverage(local, gold[rows], base, comments, rank, projects[rows])
        result['reference_sha256'] = checksum(roster)
        result['seconds'] = perf_counter()-start
        reports[part] = result
        print(part, result['baseline'], {
            key: (row['frozen']['recovered'], row['frozen']['delta_fp'], row['matched']['recovered'])
            for key, row in result['results'].items()}, flush=True)
    write_json(HERE/'results.json', reports)
    columns = ('tp','fp','fn','f1','recovered','delta_fp','recovered_without_comment')
    with (HERE/'comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('partition','arm','protocol',*columns))
        for part, result in reports.items():
            writer.writerow((part, 'canonical', 'released', *(result['baseline'][k] for k in columns)))
            for key, row in result['results'].items():
                for mode in ('frozen','matched'):
                    writer.writerow((part, key, mode, *(row[mode][k] for k in columns)))


if __name__ == '__main__':
    main()
