"""Select a project-supported positive-cohort model and time it on public validation."""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'round25'), str(ROOT/'round26/mat')]
from view_public import (R24, VIEWS, adjust, f, REFERENCE, checksum, json_at, write_json,
                         recipe, scan, PROGRAM, load, PUBLIC)
from mat_runtime import canonical_mat
from positive_runtime import apply

OUT = HERE/'public'
OLD = ROOT/'round26/mat/public'


def freeze():
    split = HERE/'joint'
    refs = [split/k/'results.json' for k in ('test_a', 'test_b', 'official')]
    a, b, official = [json_at(k) for k in refs]
    supported = {}
    for key, row in official['options'].items():
        matched = [p['options'][key]['matched']['delta_tp'] for p in (a, b)]
        if min(matched) < 0 or max(matched) == 0:
            continue
        for label, cut in row['cuts'].items():
            m = cut['calibration']
            k = f'{key}:{label}'
            supported[k] = dict(cut, gains=matched, weights_key=f"{row['sparse']}_{row['mode']}",
                                online={k: row[k] for k in ('width', 'mix', 'mode', 'direction', 'sparse')} | {'cut': cut['cut']},
                                cost=row['cost'], acceptable=m['delta_tp'] >= 0 and m['delta_fp'] <= 0)
    viable = [n for n, k in supported.items() if k['acceptable'] and k['proxy'] > 0]
    winner = max(viable, key=lambda k: supported[k]['proxy'], default=REFERENCE)
    record = dict(selected=winner, supported=supported, table_sha256={k.parent.name: checksum(k) for k in refs},
                  weights_sha256=official['weights_sha256'], archive_sha256=checksum(OLD/'canonical.csv'))
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'policy.json', record)
    if winner == REFERENCE:
        print('selected', winner, 'supported', len(supported), flush=True)
    else:
        print('selected', winner, 'supported', len(supported), record['supported'][winner], flush=True)
    return record


def run(frozen, queue=None, out=OUT):
    positive_name = frozen['selected']
    chosen = positive_name if queue is None else queue['selected']
    if chosen == REFERENCE:
        return
    option = frozen['supported'][positive_name]
    new = HERE/'joint/official/weights.joblib'
    if checksum(new) != frozen['weights_sha256']:
        raise ValueError('Frozen positive-cohort model changed')
    positive = joblib.load(new)[option['weights_key']]
    info = json_at(VIEWS/'public/policy.json')
    section = info['support'][info['selected']]
    previous = VIEWS/'official/calibrator.joblib'
    base = R24/'official/tuned'
    cfg = json_at(base/'policy.json')
    if checksum(previous) != info['calibrator_sha256'] or checksum(base/'models.joblib') != cfg['models_sha256']:
        raise ValueError('Reference weights changed')
    negative = joblib.load(base/'models.joblib')['token']
    section_head = joblib.load(previous)[section['region']]
    variant = cfg['candidates'][cfg['selected']]
    ref, route, heads = recipe()
    text, y, _ = load(PUBLIC)
    old = OLD/'canonical.csv'
    old_name = 'canonical'
    old_record = json_at(OLD/'results.json')
    if queue is not None:
        from query_runtime import add_requests, update
        old_record = json_at(OUT/'results.json')
        old = OUT/'candidate.csv'
        old_name = positive_name
        q_weights = HERE/'query_joint/ranker.joblib'
        if checksum(q_weights) != queue['ranker_sha256']:
            raise ValueError('Frozen query ranker changed')
        q_model = joblib.load(q_weights)
        q_option = queue['options'][chosen]
    if checksum(old) != old_record['results'][old_name]['predictions_sha256'] or checksum(PUBLIC) != old_record['public_sha256']:
        raise ValueError('Round 26 source files changed')
    with old.open(newline='') as stream:
        archive = np.asarray([[int(r[k]) for k in ('pred_vulnerability', 'pred_MAT')] for r in csv.DictReader(stream)], dtype=bool)
    model = f.pipe('official', f.small.tokenizer())

    def forward(which, texts):
        raw, aux, cost = f.infer_student(model, texts, f.pinned()[0], columns=22, return_all=True)
        start = time.perf_counter()
        values = np.column_stack((raw[:, 0], aux[:, 0], aux[:, -2:]))
        est = f.estimate(heads, values)
        p, no, yes, ask = f.choose(values, est, route, ref['gate'])
        flag = np.zeros(len(raw), dtype=bool)
        known = np.zeros_like(flag)
        extra = np.zeros_like(flag)
        if queue is not None and which != REFERENCE:
            extra, margin, origin = add_requests(texts, p, no, yes, est['w'], ask, ref['threshold'], q_option, q_model)
        scanned = ask | extra
        if scanned.any():
            weak = scan([s for s, k in zip(texts, scanned, strict=True) if k])
            old_scan = weak[ask[scanned]]
            known[ask] = old_scan >= 0
            flag[ask] = old_scan == 1
            p[known] = np.where(flag[known], yes[known], no[known])
        state = dict(p=p.astype(np.float32), known=known, flag=flag, no=no, yes=yes, w=est['w'])
        vp, _, _, nnz, _ = adjust(texts, values, state, ref['threshold'], variant, negative,
                                  section, info['length_cutoff'], section_head)
        eligible = flag & known
        mask, q, cash = np.zeros(len(raw), dtype=bool), np.empty(0), dict(rows=0, nnz=0, seconds=0.)
        if which != REFERENCE or queue is not None:
            vp, mask, q, cash = apply(texts, values, state, ref['threshold'], vp, option['online'], positive)
            if np.any(mask & ~eligible):
                raise ValueError('The ranker escaped its routing cohort')
        if queue is not None:
            q = np.empty(0)
            mask = extra
            if extra.any():
                vp, q = update(vp, extra, scanned, weak, no, yes, margin, origin, q_option)
        mt = canonical_mat(texts)
        elapsed = time.perf_counter()-start
        cost['wall_seconds'] += elapsed
        cost['forward_seconds'] += elapsed
        cost['analytic_gflops'] += (4*nnz+cash['nnz'])/1e9
        cost.update(positive_rows=cash['rows'], positive_seconds=cash['seconds'])
        if queue is not None:
            cost.update(extra_scans=int(extra.sum()))
        return np.column_stack((vp, mt)), mask, q, cost

    out.mkdir(exist_ok=True)
    keys = (REFERENCE, chosen)
    runs, final = {k: [] for k in keys}, {}
    for k in keys:
        forward(k, text[:2048])
    for i in range(5):
        for k in keys if i % 2 == 0 else keys[::-1]:
            bits, mask, q, cost = forward(k, text)
            if np.any(bits[~mask] != archive[~mask]) or not np.array_equal(bits[:, 1], archive[:, 1]):
                raise ValueError('Round 26 baseline or an unrelated prediction changed')
            if k in final and np.any(final[k][0] != bits):
                raise ValueError('Repeated predictions changed')
            runs[k].append(cost)
            final[k] = bits, mask, q
            print(i, k, cost, flush=True)
    records = {}
    for k, (bits, scope, score) in final.items():
        med = {q: median(v[q] for v in runs[k]) for q in runs[k][0]}
        metric = f.metric(y, bits)
        matched = archive[:, 0].copy()
        y0 = y[:, 0]
        if scope.any():
            fp = int((matched[scope] & (y0[scope] == 0)).sum())
            matched[scope], _ = f.exact_budget(score, y0[scope], fp)
        path = out/('reference.csv' if k == REFERENCE else 'candidate.csv')
        ranking_scores = np.zeros(len(y0))
        ranking_scores[scope] = score
        with path.open('w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(('row', 'label_vulnerability', 'label_mat', 'pred_vulnerability', 'pred_MAT',
                             'scoped', 'ranking_score', 'vulnerability_same_fp'))
            writer.writerows((i, *g, *b, int(m), s, int(v)) for i, (g, b, m, s, v) in
                             enumerate(zip(y, bits.astype(int), scope, ranking_scores, matched, strict=True)))
        records[k] = dict(score=f.competition_formula(metric['mean']['f1'], med['forward_seconds'], med['analytic_gflops']),
                          quality=metric, cost=med, trials=runs[k], frozen=f.comparison(y0, archive[:, 0], bits[:, 0]),
                          equal_fp=f.comparison(y0, archive[:, 0], matched), predictions_sha256=checksum(path))
    paired = [f.competition_formula(records[chosen]['quality']['mean']['f1'], c['forward_seconds'], c['analytic_gflops'])-
              f.competition_formula(records[REFERENCE]['quality']['mean']['f1'], r['forward_seconds'], r['analytic_gflops'])
              for r, c in zip(runs[REFERENCE], runs[chosen], strict=True)]
    source = OUT/'policy.json' if queue is None else HERE/'query_joint/policy.json'
    write_json(out/'results.json', dict(selected=chosen, results=records, paired_delta=paired,
                                       public_sha256=checksum(PUBLIC), policy_sha256=checksum(source),
                                       scanner_sha256=checksum(PROGRAM)))
    print({k: v['score'] for k, v in records.items()}, 'paired', paired, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--select-only', action='store_true')
    p.add_argument('--queue', action='store_true')
    args = p.parse_args()
    if args.queue:
        frozen = json_at(OUT/'policy.json')
        policy = json_at(HERE/'query_joint/policy.json')
        if policy['selected'] != REFERENCE:
            run(frozen, policy, HERE/'queue_public')
    else:
        frozen = freeze()
        if not args.select_only:
            run(frozen)
