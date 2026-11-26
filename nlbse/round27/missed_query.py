"""Test whether previously skipped auxiliary queries carry useful information."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from positive import HERE, ROOT, base, f, json_at, checksum, write_json, match, strict_cut
from fuse_sparse import clip_logit

OUT = HERE/'query_replay'
METHODS = ('distance', 'expected_flip', 'learned')


def prepare(ctx, threshold, original):
    state = ctx.state
    z = state['p']
    p = clip_logit(z.astype(np.float64))
    cutoff = clip_logit(np.asarray(np.float32(threshold), dtype=np.float64))
    margin = p-cutoff
    valid = f.valid(ctx.frame['raw'])
    skipped = ~state['asked']
    if not np.array_equal(original[skipped], margin[skipped]>0):
        raise ValueError('The skipped-query cohort no longer uses the fixed baseline')
    no, yes, w = [np.asarray(state[k]) for k in ('no', 'yes', 'w')]
    do = np.where(f.weak(ctx.frame['raw']), yes, no).astype(np.float32)
    exact = np.where(valid, clip_logit(do.astype(np.float64))-cutoff, margin)
    width = np.clip(ctx.lengths.astype(np.float64), 48., 2048.)
    pos_flip = (yes > threshold) != original
    neg_flip = (no > threshold) != original
    possible = skipped & (pos_flip | neg_flip)
    gains = ((exact>0.) == ctx.frame['y']).astype(np.float64) - (original == ctx.frame['y'])
    predictors = np.column_stack((margin, w, no-z, yes-z, np.log1p(width), pos_flip, neg_flip))
    flip_chance = (1.-w)*neg_flip+w*pos_flip
    return dict(margin=margin, proposal=exact, width=width, possible=possible, target=gains,
                x=predictors, chance=flip_chance)


def rank(option, values, model):
    margin, length = values['margin'], values['width']
    if option == 'distance':
        return 1./((.15+abs(margin))*np.sqrt(length))
    if option == 'expected_flip':
        return values['chance']/np.sqrt(length)
    return model.predict(values['x'])/np.sqrt(length)


def eval_rotation(part):
    dev, test, fit, cal, review, baseline, orig, threshold, _ = base(part)
    d = prepare(dev, threshold, baseline)
    t = d if test is dev else prepare(test, threshold, orig)
    y, yy = dev.frame['y'], test.frame['y']
    held = np.ones(len(yy), dtype=bool) if review is None else review
    learning = fit & d['possible']
    h = HistGradientBoostingRegressor(max_iter=85, max_leaf_nodes=9, min_samples_leaf=30,
                                      l2_regularization=2., random_state=42)
    h.fit(d['x'][learning], d['target'][learning])
    pool = OUT/part
    pool.mkdir(parents=True, exist_ok=True)
    states = {}
    output = dict(ids=test.frame['ids'][held], gold=yy[held], reference=orig[held], possible=t['possible'][held])
    volume = {}
    for mode in METHODS:
        q, r = rank(mode, d, h), rank(mode, t, h)
        cal_rows = np.flatnonzero(cal & d['possible'])
        q_sort = cal_rows[np.argsort(q[cal_rows], kind='stable')[::-1]]
        for budget, mix in product((.10, .25, .50, 1.), (.5, 1.)):
            count = min(len(q_sort), max(1, round(budget*len(q_sort))))
            if not count:
                continue
            cutoff = float(np.nextafter(q[q_sort[count-1]], -np.inf))
            asked = d['possible'] & (q > cutoff)
            request = t['possible'] & (r > cutoff)
            v = (1.-mix)*d['margin'][asked]+mix*d['proposal'][asked]
            u = (1.-mix)*t['margin'][request]+mix*t['proposal'][request]
            name = f'{mode}_{budget:g}_{mix:g}'
            scope_cal = asked[asked] & cal[asked]
            cal_gold = y[asked][scope_cal]
            old = baseline[asked][scope_cal]
            result = dict(route=cutoff, calls_cal=int((asked&cal).sum()), calls_held=int((request&held).sum()), cuts={})
            if review is None:
                exact, per = match(yy, test.frame['project'], orig, request, u)
                result.update(matched=f.comparison(yy, orig, exact), projects=per)
                output[name+'_matched'] = exact
            elif (mode, budget) not in volume:
                from union_scanner import scan
                rows = [s for s, k in zip(dev.texts, asked&cal, strict=True) if k]
                scan(rows)
                timings = []
                actual = None
                for _ in range(3):
                    start = time.perf_counter()
                    actual = scan(rows)
                    timings.append(time.perf_counter()-start)
                logical = np.where(f.valid(dev.frame['raw'][asked&cal]), f.weak(dev.frame['raw'][asked&cal]), -1)
                if not np.array_equal(actual, logical):
                    raise ValueError('Live counterfactual decision disagrees with the cached one')
                volume[mode, budget] = median(timings)
            for k in ('fixed', 'fp', 'tp'):
                cut = strict_cut(cal_gold, old, v[scope_cal], k)
                a, b = baseline.copy(), orig.copy()
                a[asked], b[request] = v>cut, u>cut
                c = f.comparison(y[cal], baseline[cal], a[cal])
                g = f.comparison(yy[held], orig[held], b[held])
                row = dict(cut=cut, calibration=c, frozen=g)
                if review is not None:
                    added = volume[mode, budget]
                    row['measured_seconds'] = added
                    row['proxy'] = .3*(c['f1_after']-c['f1_before'])-.2*f.PUBLIC_N/cal.sum()*added/1000.
                result['cuts'][k] = row
                output[name+'_'+k] = b[held]
            states[name] = result
            output[name+'_scope'] = request[held]
            output[name+'_score'] = u[held[request]]
    receipt = dict(learned_rows=int(learning.sum()), dev_eligible=int((d['possible']&cal).sum()),
                   held_eligible=int((t['possible']&held).sum()), options=states)
    archive = pool/'predictions.npz'
    np.savez_compressed(archive, **output)
    receipt['predictions_sha256'] = checksum(archive)
    write_json(pool/'results.json', receipt)
    print(part, receipt['dev_eligible'], receipt['held_eligible'],
          sorted([(x['matched']['delta_tp'], k) for k,x in states.items() if 'matched' in x], reverse=True)[:4],
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rotation', choices=('test_a', 'test_b', 'official'))
    eval_rotation(ap.parse_args().rotation)
