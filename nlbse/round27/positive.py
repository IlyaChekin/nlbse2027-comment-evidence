"""Develop a scanner-positive ranker on file-disjoint training predictions."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'round26')]
from disagreement import encoded_context, f, json_at, checksum, write_json, R24, PARTS
from length_eval import REFERENCE, recipe, match, load_sparse, calibrate
from compact_oof import signed_source
from positive_runtime import scope, signals, layout, directional, proposed, ranking, apply

SOURCES = ('token', 'both')
WIDTHS = (.35, .8, 1.8, 3.8, None)
WEIGHTS = (.35, .7)
MODES = ('local', 'compact', 'residual', 'focused', 'shape', 'contrast')
CUTS = ('fp', 'tp', 'fixed')


def logit(y, x, weight=None):
    return LogisticRegression(C=.6, solver='liblinear', max_iter=250, random_state=42).fit(x, y, sample_weight=weight)


def oof(rotation, mode):
    folder = R24/rotation
    source, known = signed_source(rotation)
    scores = folder/f'{mode}_oof.npz'
    sparse = json_at(folder/'fit.json')['models'][mode]
    if checksum(scores) != sparse['predictions_sha256']:
        raise ValueError('Stored sparse fit predictions changed')
    with np.load(scores) as z:
        if not all(np.array_equal(z[k], source[k]) for k in ('ids', 'fold', 'dup', 'y', 'weak')):
            raise ValueError('Sparse folds changed')
        flagged = z['weak'] == 1
        rows, y, fold, p = (z[k][flagged] for k in ('ids', 'y', 'fold', 'logits'))
    folder = ROOT/'round26/compact'/rotation
    proof = json_at(folder/'manifest.json')
    path = folder/'scores.npz'
    if proof['sparse_oof_sha256'] != known or proof['sha256'] != checksum(path):
        raise ValueError('Stored compact fit predictions changed')
    with np.load(path) as z:
        if not all(np.array_equal(z[k], source[k]) for k in ('ids', 'fold', 'y')):
            raise ValueError('Compact OOF scores do not align with sparse scores')
        x = signals(z['values'][flagged], None, p[:, 1])
    return x, rows, y, fold, (proof['sha256'], sparse['predictions_sha256'])


def base(rotation):
    d, t, fit, cal, review, old = encoded_context(rotation)
    if rotation == 'official':
        thresh = recipe()[0]['threshold']
    else:
        policy = json_at(f.WEAK/rotation/'policy.json')
        frames = {k: c.frame for k, c in (('dev', d), ('test', t))}
        budget = json_at(f.HERE/'factor_numeric'/rotation/'study.json')['dev_fp']
        _, _, thresh, _, _ = calibrate(frames, REFERENCE, fit, cal, budget)
    if t is None:
        t = d
    _, p, cut = scope(d.texts, d.state, thresh, None)
    expected = (p > 0.)
    w1 = d.state['known'] & d.state['flag']
    if not np.array_equal(expected[w1], old[0][0][w1]):
        raise ValueError('Positive-outcome baseline cutoff disagrees with archived decisions')
    return d, t, fit, cal, review, old[0][0], old[-1][0], thresh, cut


def model_bank(rotation, kind, d, t, fit, threshold, x_cache=None):
    folder = R24/rotation
    recipe_sparse = json_at(folder/'fit.json')['models'][kind]
    path = folder/f'{kind}.joblib'
    if checksum(path) != recipe_sparse['weights_sha256']:
        raise ValueError('Sparse weights changed')
    expert = joblib.load(path)[1]
    devx = signals(d.frame[REFERENCE], None, load_sparse(folder, kind, 'dev', d.frame, recipe_sparse)[:, 1])
    tx = devx if t is d else signals(t.frame[REFERENCE], None, load_sparse(folder, kind, 'test', t.frame, recipe_sparse)[:, 1])
    w1 = (d.frame['raw'] > 0).any(1)
    use = fit & w1
    models = dict(local=dict(joint=logit(d.frame['y'][use], devx[use])),
                  compact=dict(joint=logit(d.frame['y'][use], devx[use, :2])))
    stats = dict(local_fit=int(use.sum()), local_fit_positive=int(d.frame['y'][use].sum()))
    if x_cache is not None:
        x, rows, y, fold, hashes = oof(rotation, kind)
        ref = np.empty(len(y))
        for k in np.unique(fold):
            on = fold == k
            ref[on] = logit(y[~on], x[~on, :2]).decision_function(x[on, :2])
        local = [x_cache[i] for i in rows]
        weights = .25+3.*np.exp(-abs(ref-threshold))
        anchor = logit(y, x[:, :2])
        focus = logit(y, x[:, :2], weights)
        models.update(residual=dict(joint=logit(y, x), anchor=anchor),
                      focused=dict(joint=logit(y, x, weights), anchor=focus),
                      shape=dict(joint=logit(y, layout(x, local), weights), anchor=focus),
                      contrast=dict(joint=logit(y, directional(x), weights), anchor=focus))
        stats.update(oof_sha256=hashes, oof_rows=len(y), oof_positive=int(y.sum()),
                     compact_ap=float(average_precision_score(y, ref)),
                     sparse_ap=float(average_precision_score(y, x[:, -1])))
    return models, expert, devx, tx, stats


def strict_cut(yp, old, q, how):
    if how == 'fixed':
        return 0.
    if how == 'fp':
        n = int((old & (yp == 0)).sum())
        if not len(q):
            return 0.
        values = np.sort(q[yp == 0])[::-1]
        return float(values[n]) if n < len(values) else float(np.nextafter(q.min(), -np.inf))
    n = int((old & (yp == 1)).sum())
    if not n:
        return float(max(0., np.max(q, initial=0.)))
    selected = np.sort(q[yp == 1])[::-1]
    return float(np.nextafter(selected[n-1], -np.inf))


def scoped(indices, scores, width):
    if width is None:
        return indices.copy()
    return indices & (np.abs(scores) < width)


def evaluate(rotation, stage):
    d, t, fit, cal, review, old, origin, threshold, cutoff = base(rotation)
    x_cache = f.inputs()[0] if stage == 'joint' else None
    active, before, _ = scope(d.texts, d.state, threshold, None)
    on, prior, _ = scope(t.texts, t.state, threshold, None)
    if not np.array_equal(prior[on] > 0., origin[on]):
        raise ValueError('Test baseline threshold changed inside this cohort')
    dy, ty = d.frame['y'], t.frame['y']
    held = np.ones(len(ty), dtype=bool) if review is None else review
    folder = HERE/stage/rotation
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    sources = {}
    weights = {}
    bits = dict(rows=t.frame['ids'][held], gold=ty[held], old=origin[held],
                state=on[held], margins=prior[held], cal_rows=d.frame['ids'][cal])
    f1 = f.comparison(dy[cal], old[cal], old[cal])['f1_before']
    for sparse in SOURCES:
        models, expert, x, z, data = model_bank(rotation, sparse, d, t, fit, cutoff, x_cache)
        sources[sparse] = data
        for mode, model in models.items():
            if mode == 'compact' and sparse != 'token':
                continue
            ls = [s for s, yes in zip(d.texts, active, strict=True) if yes]
            lt = ls if t is d else [s for s, yes in zip(t.texts, on, strict=True) if yes]
            view = slice(0, 2) if mode == 'compact' else slice(None)
            v = proposed(x[active, view], ls, before[active], cutoff, mode, model)
            w = v if t is d else proposed(z[on, view], lt, prior[on], cutoff, mode, model)
            package = dict(model=model, expert=expert)
            weights[f'{sparse}_{mode}'] = package
            for width in WIDTHS:
                selected = scoped(active, before, width)
                queried = scoped(on, prior, width)
                yc = dy[selected & cal]
                compare = old[selected & cal]
                if not selected.any() or not queried.any() or not len(yc):
                    continue
                mode_width = 'all' if width is None else f'{width:g}'
                timing = {}
                if review is not None:
                    sample = [s for s, enabled in zip(d.texts, cal, strict=True) if enabled]
                    state = {k: p[cal] for k, p in d.state.items()}
                    control = old[cal]
                    test_options = dict(mode=mode, sparse=sparse, width=width, mix=.7, direction='blend', cut=0.)
                    config = (sample, d.frame[REFERENCE][cal], state, threshold, control, test_options, package)
                    apply(*config)
                    trials = [apply(*config)[-1] for _ in range(3)]
                    timing = {k: median(v[k] for v in trials) for k in trials[0]}
                    if timing['rows'] != int((selected & cal).sum()):
                        raise ValueError('Measured scope changed')
                for mix, direction in product(WEIGHTS, ('blend', 'veto')):
                    qc = ranking(before[selected], v[selected[active]], mix, direction)
                    qt = ranking(prior[queried], w[queried[on]], mix, direction)
                    key = f'{sparse}_{mode}_{mode_width}_{mix:g}_{direction}'
                    evidence = dict(width=width, mix=mix, sparse=sparse, mode=mode, direction=direction,
                                    test_calls=int((queried & held).sum()), cal_calls=int((selected & cal).sum()))
                    if review is None:
                        matched, per = match(ty, t.frame['project'], origin, queried, qt)
                        evidence.update(matched=f.comparison(ty, origin, matched), project=per)
                        bits[key+'_matched'] = matched
                    else:
                        evidence['cost'] = timing
                        cfg = dict(mode=mode, sparse=sparse, width=width, mix=mix, direction=direction, cut=0.)
                        expected = qc[cal[selected]]
                        online = apply(sample, d.frame[REFERENCE][cal], state, threshold, control, cfg, package)[2]
                        if not np.allclose(online, expected, rtol=2e-5, atol=2e-5):
                            raise ValueError('A cached ranking cannot be reproduced at live inference')
                    evaluations = {}
                    for how in CUTS:
                        cut = strict_cut(yc, compare, qc[cal[selected]], how)
                        a = old.copy()
                        a[selected] = qc > cut
                        b = origin.copy()
                        b[queried] = qt > cut
                        development = f.comparison(dy[cal], old[cal], a[cal])
                        validation = f.comparison(ty[held], origin[held], b[held])
                        values = dict(cut=cut, calibration=development, frozen=validation)
                        if review is not None:
                            debit = .2*f.PUBLIC_N/cal.sum()*(timing['seconds']/1000.+timing['nnz']/3e15)
                            values['proxy'] = .3*(development['f1_after']-f1)-debit
                        evaluations[how] = values
                        bits[key+'_'+how] = b[held]
                    evidence['cuts'] = evaluations
                    results[key] = evidence
                    bits[key+'_scope'] = queried[held]
                    bits[key+'_rank'] = qt[held[queried]]
            print(rotation, sparse, mode, len(results), flush=True)
    path = folder/'weights.joblib'
    joblib.dump(weights, path, compress=3)
    final = folder/'predictions.npz'
    np.savez_compressed(final, **bits)
    write_json(folder/'results.json', dict(options=results, sources=sources, reference_thresh=threshold,
                 scope={'cal': f.comparison(dy[cal & active], old[cal & active], old[cal & active])['before'],
                        'test': f.comparison(ty[held & on], origin[held & on], origin[held & on])['before']},
                 weights_sha256=checksum(path), predictions_sha256=checksum(final), code_sha256=checksum(Path(__file__))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation', choices=(*PARTS, 'official'))
    p.add_argument('--stage', choices=('cached', 'joint'), default='joint')
    args = p.parse_args()
    evaluate(args.rotation, args.stage)
