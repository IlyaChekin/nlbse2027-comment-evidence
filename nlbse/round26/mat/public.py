"""Measure a frozen canonical marker detector against the released Round 25 model."""
import csv
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R25 = ROOT / 'round25'
sys.path[:0] = [str(ROOT / 'round21/mat'), str(R25)]
from view_public import (R24, VIEWS, adjust, f, REFERENCE, checksum, json_at, write_json,
                         recipe, scan, PROGRAM, recover, load, PUBLIC)
from mat_runtime import canonical_mat

OUT = HERE / 'public'
REPEATS = 5


def archived(name, report, expected):
    path = VIEWS / 'public' / f'{name}.csv'
    if checksum(path) != report['results'][name]['predictions_sha256']:
        raise ValueError('Round 25 public predictions changed')
    with path.open(newline='') as fd:
        rows = list(csv.DictReader(fd))
    expected_rows = np.arange(len(rows))
    assert np.array_equal([int(r['row']) for r in rows], expected_rows)
    assert np.array_equal([[int(r['label_vulnerability']), int(r['label_mat'])] for r in rows], expected)
    return np.asarray([[int(r['pred_vulnerability']), int(r['pred_MAT'])] for r in rows], dtype=bool)


def policy():
    # The tie-break is fixed from fit/dev. Canonical parsing removed noisy tags in fit.
    data = ROOT / 'round21/mat' / 'annotation_results.json'
    checked = json_at(data)
    metrics = {g: {k: checked[g]['metrics'][k]['default']
                   for k in ('union_lexical', 'replace_upstream')} for g in ('fit', 'dev', 'test_a', 'test_b')}
    for side in metrics.values():
        baseline, canonical = side['union_lexical']['candidate'], side['replace_upstream']['candidate']
        if canonical['tp'] < baseline['tp'] or canonical['fp'] > baseline['fp']:
            raise ValueError('Canonical marker did not transfer from training to held-out projects')
    if metrics['fit']['replace_upstream']['candidate']['fp'] >= metrics['fit']['union_lexical']['candidate']['fp']:
        raise ValueError('Fit supplies no rationale for the frozen replacement')
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(selected='canonical', metrics=metrics, annotation_sha256=checksum(data),
                  canonical_sha256=checksum(HERE/'mat_runtime.py'), marker_sha256=checksum(ROOT/'round21/mat/marker.py'),
                  baseline_policy_sha256=checksum(VIEWS/'public/policy.json'))
    write_json(OUT/'policy.json', result)
    return result


def run():
    frozen = policy()
    region = json_at(VIEWS/'public/policy.json')
    option = region['support'][region['selected']]
    config = json_at(R24/'official/tuned/policy.json')
    package_path = R24/'official/tuned/models.joblib'
    calibration = VIEWS/'official/calibrator.joblib'
    if checksum(package_path) != config['models_sha256'] or checksum(calibration) != region['calibrator_sha256']:
        raise ValueError('Verified vulnerability readouts changed')
    package = joblib.load(package_path)['token']
    head = joblib.load(calibration)[option['region']]
    sparse = config['candidates'][config['selected']]
    ref, route, heads = recipe()
    texts, truth, _ = load(PUBLIC)
    public = json_at(VIEWS/'public/results.json')
    base = archived(region['selected'], public, truth)
    model = f.pipe('official', f.small.tokenizer())

    def forward(local, rule):
        pred, aux, cost = f.infer_student(model, local, f.pinned()[0], columns=22, return_all=True)
        start = time.perf_counter()
        values = np.column_stack((pred[:, 0], aux[:, 0], aux[:, -2:]))
        est = f.estimate(heads, values)
        score, no, yes, ask = f.choose(values, est, route, ref['gate'])
        flags = np.zeros(len(pred), dtype=bool)
        known = np.zeros_like(flags)
        if ask.any():
            outcomes = scan([s for s, k in zip(local, ask, strict=True) if k])
            known[ask] = outcomes >= 0
            flags[ask] = outcomes == 1
            score[known] = np.where(flags[known], yes[known], no[known])
        state = dict(p=score.astype(np.float32), known=known, flag=flags, w=est['w'], no=no, yes=yes)
        estimate, _, touched, nnz, _ = adjust(local, values, state, ref['threshold'], sparse, package,
                                               option, region['length_cutoff'], head)
        mat_start = time.perf_counter()
        mat = canonical_mat(local) if rule == 'canonical' else recover(local, pred[:, 1] > .5)
        elapsed = time.perf_counter()-start
        cost['wall_seconds'] += elapsed
        cost['forward_seconds'] += elapsed
        cost['analytic_gflops'] += 4*nnz/1e9
        cost.update(mat_seconds=elapsed-(mat_start-start), segment_calls=int(touched.sum()))
        return np.column_stack((estimate, mat)), cost

    names = ('reference', 'canonical')
    trials = {n: [] for n in names}
    stored = {'reference': base}
    for n in names:
        forward(texts[:2048], n)
    for i in range(REPEATS):
        for n in names if i % 2 == 0 else names[::-1]:
            bits, cost = forward(texts, n)
            if not np.array_equal(bits[:, 0], base[:, 0]):
                raise ValueError('The MAT change changed a vulnerability prediction')
            if n in stored and not np.array_equal(stored[n], bits):
                raise ValueError('Repeated decisions or Round 25 reference changed')
            stored[n] = bits
            trials[n].append(cost)
            print(i, n, cost, flush=True)
    stats = {}
    for n, bits in stored.items():
        quality = f.metric(truth, bits)
        cost = {k: median(a[k] for a in trials[n]) for k in trials[n][0]}
        score = f.competition_formula(quality['mean']['f1'], cost['forward_seconds'], cost['analytic_gflops'])
        file = OUT/f'{n}.csv'
        with file.open('w', newline='') as fd:
            w = csv.writer(fd)
            w.writerow(('row', 'label_vulnerability', 'label_mat', 'pred_vulnerability', 'pred_MAT'))
            w.writerows((i, *yy, *bb) for i, (yy, bb) in enumerate(zip(truth, bits.astype(int), strict=True)))
        stats[n] = dict(score=score, quality=quality, cost=cost, trials=trials[n], predictions_sha256=checksum(file),
                        mat=f.comparison(truth[:, 1], base[:, 1], bits[:, 1]))
    pair = [f.competition_formula(stats['canonical']['quality']['mean']['f1'], b['forward_seconds'], b['analytic_gflops'])-
            f.competition_formula(stats['reference']['quality']['mean']['f1'], a['forward_seconds'], a['analytic_gflops'])
            for a, b in zip(trials['reference'], trials['canonical'], strict=True)]
    write_json(OUT/'results.json', dict(results=stats, rows=len(texts), selected=frozen['selected'],
                                      public_sha256=checksum(PUBLIC), scanner_sha256=checksum(PROGRAM),
                                      policy_sha256=checksum(OUT/'policy.json'), paired_delta=pair))
    print({k: v['score'] for k, v in stats.items()}, 'paired_delta', pair, flush=True)


if __name__ == '__main__':
    run()
