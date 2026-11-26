"""Test whether rare code fragments predict a hidden MAT comment."""
import csv
import json
from pathlib import Path
import hashlib
import sys
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'round14'), str(ROOT / 'round21/mat')]
from audit import checksum, dataset, split, write_json
from mat import cut_at_budget, evaluate
from marker import upstream_tag
from annotation import SOURCE_COMMENT
from project_split import TOKEN


def stripped(text):
    return [m[0] for m in TOKEN.finditer(text) if m['comment'] is None]


def token_keys(texts):
    return np.asarray([hashlib.blake2b(repr(sequence).encode(), digest_size=16).hexdigest()
                       for sequence in texts])


def anchors(sequence):
    return [t for t in sequence if t.isidentifier() and len(t) >= 3 or t.startswith(('"', 'R"'))]


def identity(sequence):
    return sequence


def vectorizer(view):
    words, span = (identity, (3, 4)) if view == 'ordered' else (anchors, (2, 3))
    return HashingVectorizer(analyzer='word', tokenizer=words, preprocessor=identity, token_pattern=None,
                             ngram_range=span, lowercase=False, norm=None, n_features=2**22,
                             alternate_sign=False, binary=True, dtype=np.float32)


def index(view, text, gold, rows, keys):
    raw = vectorizer(view).transform(text)
    df = np.bincount(raw.indices, minlength=raw.shape[1])
    feature = np.flatnonzero((df >= 2) & (df <= 160))
    scale = np.log1p(len(text) / df[feature]).astype(np.float32)
    matrix = normalize(raw[:, feature].multiply(scale), copy=False)
    matrix.sort_indices()
    return dict(view=view, features=feature, scale=scale, rows=rows, keys=keys, gold=gold,
                matrix=matrix.T.tocsr())


def search(index, texts, keys):
    view = index['view']
    vector = vectorizer(view)
    value = np.zeros((len(texts), 4), dtype=np.float32)
    neighbor = np.full((len(texts), 2), -1, dtype=np.int32)
    for first in range(0, len(texts), 96):
        records = texts[first:first+96]
        raw = vector.transform(records)[:, index['features']]
        query = normalize(raw.multiply(index['scale']), copy=False)
        dots = (query @ index['matrix']).tocsr()
        for j in range(len(records)):
            where = slice(dots.indptr[j], dots.indptr[j+1])
            ix = dots.indices[where]
            sim = dots.data[where]
            present = index['keys'][ix] != keys[first+j]
            gold = index['gold'][ix]
            for label in (False, True):
                hits = np.flatnonzero((gold == label) & present)
                if len(hits):
                    strongest = hits[np.argmax(sim[hits])]
                    value[first+j, int(label)] = sim[strongest]
                    neighbor[first+j, int(label)] = index['rows'][ix[strongest]]
                    value[first+j, 2+int(label)] = (sim[hits] >= .60).sum()
    return value, neighbor


def current(which, ids, y, texts):
    path = ROOT / 'round21/mat' / f'annotation_{which}.csv'
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert np.array_equal(ids, [int(r['row']) for r in rows])
    assert np.array_equal(y, [int(r['gold']) for r in rows])
    exact = np.asarray([any(upstream_tag(m[0]) for m in SOURCE_COMMENT.finditer(t)) for t in texts])
    assert np.array_equal(exact, [int(r['loose_annotation']) for r in rows])
    baseline = np.asarray([bool(int(r['union_lexical_frozen'])) for r in rows])
    comment = np.asarray([bool(TOKEN.search(t) and any(m['comment'] is not None for m in TOKEN.finditer(t)))
                          for t in texts])
    return baseline, exact, comment


def adjustments(values):
    negative, positive, negative_count, positive_count = values.T
    return {'positive': positive, 'contrast_half': positive-.5*negative,
            'contrast': positive-negative,
            'consensus': np.where(positive_count >= 2, positive-negative, -1.)}


def score_arrays(gold, baseline, exact, comment, score, cut):
    prediction = exact | (~baseline & (score > cut))
    record = evaluate(gold, baseline, prediction)
    record['recovered_without_comment'] = int((gold & ~baseline & prediction & ~comment).sum())
    return record, prediction


def calibrate(y, base, exact, score):
    eligible = ~base
    budget = int((base & ~y).sum()-(exact & ~y).sum())
    if budget < 0:
        raise ValueError('Canonical tagging already exceeds the baseline FP budget')
    return cut_at_budget(score[eligible], y[eligible], budget, .05)


def policies(available):
    return {f'{view}/{key}': value for view, signals in available.items()
            for key, value in adjustments(signals).items()}


def part(which, ids, text, label, project, indices, frozen=None):
    local = [text[i] for i in ids]
    y, p = label[ids], project[ids]
    base, exact, comment = current(which, ids, y, local)
    canonical = evaluate(y, base, exact)
    active = np.flatnonzero(~base)
    start = time.perf_counter()
    sequence = [stripped(local[i]) for i in active]
    keys = token_keys(sequence)
    duration = {'tokenize': time.perf_counter()-start}
    views = {}
    raw = {}
    for view, model in indices.items():
        start = time.perf_counter()
        values, near = search(model, sequence, keys)
        duration[view] = time.perf_counter()-start
        views[view] = np.full((len(ids), 4), -1., dtype=np.float32)
        views[view][active] = values
        raw[view + '_scores'] = values
        raw[view + '_neighbor_fit_ids'] = near
    signals = policies(views)
    if frozen is None:
        frozen = {name: calibrate(y, base, exact, score) for name, score in signals.items()}
    results = {}
    for name, signal in signals.items():
        global_result, pred = score_arrays(y, base, exact, comment, signal, frozen[name])
        matched = np.zeros_like(base)
        groups = {}
        for family in np.unique(p):
            use = p == family
            cut = calibrate(y[use], base[use], exact[use], signal[use])
            checked, matched[use] = score_arrays(y[use], base[use], exact[use], comment[use], signal[use], cut)
            if checked['extra_fp'] > 0:
                raise ValueError(f'Exceeded FP budget in {which} {family} {name}')
            groups[family] = dict(cutoff=cut, **checked)
        exact_fp, _ = score_arrays(y, base, exact, comment, np.where(matched & ~base, 1., 0.), .5)
        results[name] = dict(cutoff=frozen[name], frozen=global_result, matched=exact_fp, project=groups)
        raw[name.replace('/', '_') + '_frozen'] = pred
        raw[name.replace('/', '_') + '_matched'] = matched
    output = HERE / f'{which}_predictions.npz'
    np.savez_compressed(output, ids=ids, active=active, y=y, project=p, base=base, canonical=exact,
                       comment=comment, **raw)
    info = dict(canonical=canonical, has_comment_misses=int((y & ~base & comment).sum()),
                no_comment_misses=int((y & ~base & ~comment).sum()), elapsed_seconds=duration,
                options=results, predictions_sha256=checksum(output))
    print(which, canonical, [(n, o['frozen']['recovered'], o['frozen']['extra_fp'],
                               o['matched']['recovered'], o['matched']['extra_fp']) for n,o in results.items()],
          duration, flush=True)
    return info, frozen


def main():
    texts, label, project, _ = dataset()
    label = label[:, 1].astype(bool)
    parts = split()
    fit = parts['fit']
    tokenized = [stripped(texts[i]) for i in fit]
    keys = token_keys(tokenized)
    # Already excluded by the project split. The search independently rejects exact token copies.
    models = {}
    checkpoints = {}
    for view in ('ordered', 'anchors'):
        at = time.perf_counter()
        models[view] = index(view, tokenized, label[fit], fit, keys)
        path = HERE / f'{view}_index.joblib'
        joblib.dump(models[view], path, compress=2)
        checkpoints[view] = dict(sha256=checksum(path), dimensions=models[view]['matrix'].shape,
                                 indexed_rows=len(fit), seconds=time.perf_counter()-at)
        print(view, checkpoints[view], flush=True)
    dev, thresholds = part('dev', parts['dev'], texts, label, project, models)
    selected = max(thresholds, key=lambda k: (dev['options'][k]['frozen']['recovered']-dev['options'][k]['frozen']['lost'],
                                              -dev['options'][k]['frozen']['extra_fp']))
    if dev['options'][selected]['frozen']['recovered'] == 0:
        selected = 'canonical'
    policy = dict(thresholds=thresholds, selected=selected, checkpoints=checkpoints,
                  training_rows=len(fit), training_positives=int(label[fit].sum()),
                  split_sha256=checksum(ROOT/'round13/partition.npz'), code_sha256=checksum(Path(__file__)),
                  data_manifest_sha256=checksum(ROOT/'data/manifest.json'))
    write_json(HERE/'policy.json', policy)
    reports = {'dev': dev}
    for which in ('test_a', 'test_b'):
        reports[which], _ = part(which, parts[which], texts, label, project, models, thresholds)
    write_json(HERE/'results.json', reports)


if __name__ == '__main__':
    main()
