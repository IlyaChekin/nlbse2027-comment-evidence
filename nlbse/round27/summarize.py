"""Audit the signed experiments, full validation predictions, and remaining score gap."""
import csv
from statistics import median

import numpy as np

from public import HERE, ROOT, OUT, OLD, checksum, json_at, write_json, f, PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes


def csv_predictions(path, expected):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r['row']) for r in rows] == list(range(len(expected)))
    fields = ('label_vulnerability', 'label_mat')
    assert np.array_equal([[int(r[k]) for k in fields] for r in rows], expected)
    pred = np.asarray([[int(r[k]) for k in ('pred_vulnerability', 'pred_MAT')] for r in rows], dtype=bool)
    selected = np.asarray([int(r['scoped']) for r in rows], dtype=bool)
    return pred, selected


def audit_public():
    report = json_at(OUT/'results.json')
    assert checksum(OUT/'policy.json') == report['policy_sha256']
    assert checksum(PUBLIC) == report['public_sha256']
    selected = report['selected']
    _, y, m = load(PUBLIC, audit=True)
    forecasts = {}
    for key, name in (('reference', 'reference'), ('candidate', selected)):
        model = report['results'][name]
        path = OUT/f'{key}.csv'
        assert checksum(path) == model['predictions_sha256']
        all_costs = model['trials']
        assert len(all_costs) == 5
        assert all(v == median(x[k] for x in all_costs) for k, v in model['cost'].items())
        forecasts[key], selected_rows = csv_predictions(path, y)
        actual = f.metric(y, forecasts[key])
        assert actual == model['quality']
        assert f.competition_formula(actual['mean']['f1'], model['cost']['forward_seconds'],
                                     model['cost']['analytic_gflops']) == model['score']
    control, live = forecasts['reference'], forecasts['candidate']
    old_archive = json_at(OLD/'results.json')['results']['canonical']['predictions_sha256']
    assert checksum(OLD/'canonical.csv') == old_archive
    with (OLD/'canonical.csv').open() as fp:
        assert all(np.array_equal(row, [int(l[k]) for k in ('pred_vulnerability', 'pred_MAT')])
                   for row, l in zip(control, csv.DictReader(fp), strict=True))
    assert np.array_equal(control[:, 1], live[:, 1]) and not np.any(control[~selected_rows] != live[~selected_rows])
    raw = ROOT/'round22/union_public/scanner_predictions.npz'
    signed_raw = json_at(ROOT/'round22/union_public/results.json')['scanner_predictions_sha256']
    assert checksum(raw) == signed_raw
    with np.load(raw) as z:
        gate, flag = z['gate'], z['flag']
        positive = gate.copy()
        negative = gate.copy()
        positive[gate] = flag == 1
        negative[gate] = flag == 0
    assert np.all(positive[selected_rows])
    independent = ROOT/'round20/conditional_audit.csv'
    with independent.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r['row']) for r in rows] == list(range(len(y)))
    assert np.array_equal([int(r['vulnerability_gold']) for r in rows], y[:, 0])
    full = np.asarray([int(r['any_rule']) for r in rows], dtype=bool)
    assert np.all(full[positive]) and not np.any(full[negative])
    missing = {k: f.comparison(y[x, 0], control[x, 0], live[x, 0]) for k, x in (
        ('positive_unobserved', full & ~positive), ('remaining', ~full & ~negative))}
    coh = {k: f.comparison(y[x, 0], control[x, 0], live[x, 0]) for k, x in (
        ('positive', positive), ('negative', negative), ('other', ~positive & ~negative))}
    groups = group_codes(zip(m['Projectname'], m['Filepath'], strict=True))
    interval = bootstrap_delta(y, control, live, groups)
    new = report['results'][selected]
    confusion = new['frozen']['after']
    def projected(tp, fp, fn):
        v = 2*tp/(2*tp+fp+fn)
        mean = (v+new['quality']['MAT']['f1'])/2
        return f.competition_formula(mean, new['cost']['forward_seconds'], new['cost']['analytic_gflops'])
    limits = {}
    limits['additional_fp_to_remove_with_fixed_recall'] = next(
        (x for x in range(confusion['fp']+1)
         if projected(confusion['tp'], confusion['fp']-x, confusion['fn']) >= .97), None)
    limits['score_if_no_fn_at_same_fp'] = projected(confusion['tp']+confusion['fn'], confusion['fp'], 0)
    limits['score_if_no_runtime_or_operations'] = .4+.6*new['quality']['mean']['f1']
    for k, v in coh.items():
        count = v['after']
        limits[f'perfect_{k}_at_same_cost'] = projected(confusion['tp']+count['fn'], confusion['fp']-count['fp'],
                                                       confusion['fn']-count['fn'])
    receipt = json_at(HERE/'release_verification/receipt.json')
    archive = ROOT/'submission/NLBSE2027_round27_positive_tool.zip'
    release = HERE/'release_verification/release_predictions.csv'
    assert receipt['archive_sha256'] == checksum(archive) and receipt['predictions_sha256'] == checksum(release)
    with release.open(newline='') as source:
        assert all(np.array_equal(row, [int(r[k]) for k in ('vulnerability', 'MAT')])
                   for row, r in zip(live, csv.DictReader(source), strict=True))
    return dict(interval_95=interval, cohorts=coh, scope_count=int(selected_rows.sum()), score=new['score'],
                unobserved_outcome_diagnostic=missing, offline_detector_sha256=checksum(independent),
                gap=.97-new['score'], limits=limits, profile_sha256=checksum(OUT/'results.json'))


def compare_studies():
    records = []
    best = {}
    for kind in ('joint', 'teacher'):
        directory = HERE/kind
        groups = [json_at(directory/p/'results.json') for p in ('test_a', 'test_b', 'official')]
        for p, table in zip(('test_a', 'test_b', 'official'), groups, strict=True):
            assert checksum(directory/p/'predictions.npz') == table['predictions_sha256']
            weights = 'weights.joblib' if kind == 'joint' else 'models.joblib'
            assert checksum(directory/p/weights) == table['weights_sha256']
        a, b, c = (t['options'] for t in groups)
        for k, v in c.items():
            tests = (a[k], b[k])
            for t in tests:
                assert all(x['delta_fp'] == 0 for x in t['project'].values())
                assert t['matched']['delta_tp'] == sum(x['delta_tp'] for x in t['project'].values())
            g = [t['matched']['delta_tp'] for t in tests]
            supported = min(g) >= 0 and max(g) > 0
            for criterion, d in v['cuts'].items():
                m, n = d['calibration'], d['frozen']
                row = dict(study=kind, method=k, criterion=criterion, project_a_tp=g[0], project_b_tp=g[1],
                         project_support=supported, internal_tp=m['delta_tp'], internal_fp=m['delta_fp'],
                         internal_review_tp=n['delta_tp'], internal_review_fp=n['delta_fp'],
                         paid_calls=v['cal_calls'], proxy_delta=d['proxy'])
                records.append(row)
                if supported and m['delta_tp'] >= 0 and m['delta_fp'] <= 0 and (
                        kind not in best or d['proxy'] > best[kind]['proxy_delta']):
                    best[kind] = row
    with (HERE/'comparison.csv').open('w', newline='') as fd:
        writer = csv.DictWriter(fd, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    return best


if __name__ == '__main__':
    metrics = audit_public()
    metrics['best_supported'] = compare_studies()
    write_json(HERE/'summary.json', metrics)
    print(metrics, flush=True)
