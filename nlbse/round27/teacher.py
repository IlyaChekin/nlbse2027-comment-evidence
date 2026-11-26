"""Evaluate a cached independent encoder on positive auxiliary outcomes."""
from itertools import product
from statistics import median

import joblib
import numpy as np

from positive import HERE, ROOT, f, json_at, checksum, write_json, logit, base, match, strict_cut
from positive_runtime import scope, signals, ranking, clip_logit

MODEL = 'unixcoder9'
WIDTHS = (.28, .65, 1.60)
CUTS = ('fp', 'tp', 'fixed')
OUT = HERE/'teacher'


def archive(rotation, d, t):
    y = f.inputs()[1][:, 0]
    if rotation == 'official':
        directory = ROOT/'round10/runs'/MODEL
        setup = json_at(directory/'selected.json')
        checkpoint = directory/setup['checkpoint']
        if checksum(checkpoint) != setup['settings']['sha256']:
            raise ValueError('Adviser weights changed')
        path = directory/f'{checkpoint.stem}_holdout.npz'
        with np.load(path) as v, np.load(ROOT/'round8/frozen/holdout.npz') as split:
            if (not np.array_equal(v['ids'], d.frame['ids']) or not np.array_equal(v['y'][:, 0], d.frame['y']) or
                    not np.array_equal(v['fold'], split['folds'])):
                raise ValueError('Archived formal adviser predictions did not align')
            return v['p'][:, 0], v['p'][:, 0], {'predictions': checksum(path), 'checkpoint': checksum(checkpoint)}
    result = np.full(len(y), np.nan, dtype=np.float32)
    record = {}
    for side in ('probe', 'dev', 'test'):
        path = ROOT/'round13/backbones/unix'/f'{side}.npz'
        sig = checksum(path)
        if sig != json_at(path.with_suffix('.json'))['sha256']:
            raise ValueError('Archived project adviser predictions changed')
        with np.load(path) as v:
            ids = v['ids']
            if not np.array_equal(v['y'][:, 0], y[ids]) or np.any(np.isfinite(result[ids])):
                raise ValueError('Archived project predictions changed row order')
            result[ids] = v['probabilities'][:, 0]
        record[side] = sig
    a, b = [result[c.frame['ids']] for c in (d, t)]
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Cached independent scores do not cover the project rotation')
    return a, b, record


def fit(rotation):
    d, t, fit_rows, cal, review, ref, test_ref, threshold, cut = base(rotation)
    p, q, source = archive(rotation, d, t)
    before = signals(d.frame['reference'], None, clip_logit(p))
    after = signals(t.frame['reference'], None, clip_logit(q))
    active, base_d, _ = scope(d.texts, d.state, threshold, None)
    test_active, base_t, _ = scope(t.texts, t.state, threshold, None)
    dy, ty = d.frame['y'], t.frame['y']
    w1 = (d.frame['raw'] > 0).any(1)
    mask = fit_rows & w1
    x = before[mask]
    y = dy[mask]
    pooled = logit(y, x)
    anchor = logit(y, x[:, :2])
    predict_d, predict_t = [pooled.decision_function(v) for v in (before, after)]
    anchor_d, anchor_t = [anchor.decision_function(v[:, :2]) for v in (before, after)]
    variants = dict(blend=(predict_d-cut, predict_t-cut),
                    residual=(base_d+predict_d-anchor_d, base_t+predict_t-anchor_t))
    held = review if review is not None else np.ones(len(ty), dtype=bool)
    dest = OUT/rotation
    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    output = dict(ids=t.frame['ids'][held], base=test_ref[held], y=ty[held])
    if review is not None:
        from students import load_selected, scores
        teacher, tokenizer, _ = load_selected(MODEL)
    for width in WIDTHS:
        inside = active & (abs(base_d) < width)
        ask = test_active & (abs(base_t) < width)
        if not np.any(inside & cal):
            continue
        costs = None
        if review is not None:
            selected = [s for s, enabled in zip(d.texts, inside & cal, strict=True) if enabled]
            scores(teacher, tokenizer, selected, MODEL)
            trials = [scores(teacher, tokenizer, selected, MODEL) for _ in range(3)]
            if not np.allclose(trials[0][0][:, 0], p[inside & cal], atol=.025, rtol=0):
                raise ValueError('The live adviser does not agree with cached scores')
            times = [v[1] for v in trials]
            costs = {m: median(v[m] for v in times) for m in times[0]}
            costs['n'] = len(selected)
        for (key, (a, b)), gain, direction in product(variants.items(), (.35, .65), ('blend', 'veto')):
            da = ranking(base_d[inside], a[inside], gain, direction)
            dt = ranking(base_t[ask], b[ask], gain, direction)
            name = f'{key}_{width:g}_{gain:g}_{direction}'
            v = dict(width=width, weight=gain, mode=key, direction=direction, cal_calls=int((inside & cal).sum()),
                     test_calls=int((ask & held).sum()), cost=costs, cuts={})
            if review is None:
                pool, per = match(ty, t.frame['project'], test_ref, ask, dt)
                v['matched'] = f.comparison(ty, test_ref, pool)
                v['project'] = per
                output[name+'_matched'] = pool
            output[name+'_mask'] = ask[held]
            output[name+'_score'] = dt[held[ask]]
            for label in CUTS:
                th = strict_cut(dy[inside & cal], ref[inside & cal], da[cal[inside]], label)
                applied_d, applied_t = ref.copy(), test_ref.copy()
                applied_d[inside], applied_t[ask] = da > th, dt > th
                dev = f.comparison(dy[cal], ref[cal], applied_d[cal])
                e = dict(cut=th, calibration=dev, frozen=f.comparison(ty[held], test_ref[held], applied_t[held]))
                if costs is not None:
                    debit = .2*f.PUBLIC_N/cal.sum()*(costs['forward_seconds']/1000 + costs['analytic_gflops']/3e6)
                    e['proxy'] = .3*(dev['f1_after']-dev['f1_before'])-debit
                v['cuts'][label] = e
                output[name+'_'+label] = applied_t[held]
            summary[name] = v
    save = dest/'models.joblib'
    joblib.dump(dict(pooled=pooled, anchor=anchor), save)
    prediction = dest/'predictions.npz'
    np.savez_compressed(prediction, **output)
    write_json(dest/'results.json', dict(options=summary, source=source, fit_rows=int(mask.sum()),
                                         predictions_sha256=checksum(prediction), weights_sha256=checksum(save)))
    print(rotation, len(summary), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation', choices=('test_a', 'test_b', 'official'))
    fit(parser.parse_args().rotation)
