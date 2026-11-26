"""Measure the marginal cost of extra queries using one shared scanner batch."""
from statistics import median
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from missed_query import HERE, OUT, ROOT, base, prepare, rank, f, json_at, checksum, write_json
from union_scanner import scan

TARGET = HERE/'query_joint'


def timing(text, initial, new, gold):
    rows = np.flatnonzero(initial | new)
    first = [t for t, use in zip(text, initial, strict=True) if use]
    extended = [text[i] for i in rows]
    baseline = scan(first)
    combined = scan(extended)
    if not np.array_equal(baseline, gold[initial]) or not np.array_equal(combined, gold[rows]):
        raise ValueError('The batched scanner disagrees with the locked outcomes')
    runs = {'base': [], 'combined': []}
    samples = {'base': first, 'combined': extended}
    for turn in range(5):
        order = ('base', 'combined') if turn % 2 == 0 else ('combined', 'base')
        for k in order:
            start = time.perf_counter()
            scan(samples[k])
            runs[k].append(time.perf_counter()-start)
    pairs = [a-b for a, b in zip(runs['combined'], runs['base'], strict=True)]
    return dict(reference_queries=int(initial.sum()), added_queries=int(new.sum()), trials=runs,
                paired_delta=pairs, seconds=max(0., median(pairs)))


def run():
    a, b, c = [json_at(OUT/part/'results.json')['options'] for part in ('test_a', 'test_b', 'official')]
    supported = {}
    for name, row in c.items():
        gains = [d[name]['matched']['delta_tp'] for d in (a, b)]
        if min(gains) < 0 or max(gains) <= 0:
            continue
        for k, val in row['cuts'].items():
            cal, review = val['calibration'], val['frozen']
            boost = .3*(cal['f1_after']-cal['f1_before'])
            if boost > 0 and cal['delta_fp'] <= 0 and cal['delta_tp'] >= 0 and review['f1_after'] >= review['f1_before']:
                supported[(name, k)] = dict(val, gain=gains, boost=boost, route=row['route'], calls=row['calls_cal'])
    d, _, fit, cal, _, baseline, _, threshold, _ = base('official')
    p = prepare(d, threshold, baseline)
    train = fit & p['possible']
    model = HistGradientBoostingRegressor(max_iter=85, max_leaf_nodes=9, min_samples_leaf=30,
                                         l2_regularization=2., random_state=42).fit(p['x'][train], p['target'][train])
    gold = np.where(f.valid(d.frame['raw']), f.weak(d.frame['raw']), -1)
    current = d.state['asked'] & cal
    TARGET.mkdir(exist_ok=True)
    records = {}
    costs = {}
    for (name, choice), row in supported.items():
        method, budget, mix = name.split('_') if not name.startswith('expected_flip') else (
            'expected_flip', *name[len('expected_flip_'):].split('_'))
        k = method, budget
        if k not in costs:
            priority = rank(method, p, model)
            ask = cal & p['possible'] & (priority > row['route'])
            if (ask & current).any() or ask.sum() != row['calls']:
                raise ValueError('The extended scanner set does not match the offline policy')
            costs[k] = timing(d.texts, current, ask, gold)
            print(name, costs[k], flush=True)
        result = costs[k]
        score = row['boost'] - .2*f.PUBLIC_N/cal.sum()*(result['seconds']/1000.)
        records[f'{name}:{choice}'] = dict(row, pooled_cost=result, measured_proxy=score,
                                          method=method, fraction=float(budget), mixture=float(mix))
    winner = max(records, key=lambda k: records[k]['measured_proxy'], default='reference')
    if winner != 'reference' and records[winner]['measured_proxy'] <= 0:
        winner = 'reference'
    w = TARGET/'ranker.joblib'
    joblib.dump(model, w)
    write_json(TARGET/'policy.json', dict(selected=winner, options=records, ranker_sha256=checksum(w), threshold=threshold))
    print('selected', winner, records.get(winner), flush=True)


if __name__ == '__main__':
    run()
