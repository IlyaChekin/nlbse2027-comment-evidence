"""Recompute MAT budgets, rank diagnostics, and the measured score gap."""
from collections import defaultdict
import csv
import json
import sys

import numpy as np

from neighbors import HERE, ROOT, adjustments, checksum, cut_at_budget, dataset, write_json

sys.path.append(str(ROOT))
from score_runtime import competition_formula


def recount(which, report):
    path = HERE / f'{which}_predictions.npz'
    assert checksum(path) == report['predictions_sha256']
    diagnostics = {}
    with np.load(path) as z:
        y, base, canonical, c, project = (z[k] for k in ('y', 'base', 'canonical', 'comment', 'project'))
        a = ~base
        assert list(z['active']) == list(np.flatnonzero(a))
        for view in ('ordered', 'anchors'):
            for key, scores in adjustments(z[view + '_scores']).items():
                name = f'{view}/{key}'
                status = report['options'][name]
                pred = z[f'{view}_{key}_matched']
                found = y & ~base & pred
                assert int(found.sum()) == status['matched']['recovered']
                assert int((found & ~c).sum()) == status['matched']['recovered_without_comment']
                assert all(int((~y & pred & (project == k)).sum()) <= int((~y & base & (project == k)).sum())
                           for k in np.unique(project))
                assert np.array_equal(pred[base], canonical[base])
                rank = scores[y[a] & ~c[a]]
                if len(rank) and rank.max() > .05:
                    false = int((scores[~y[a]] >= rank.max()).sum())
                else:
                    false = None
                without_floor = canonical.copy()
                for family in np.unique(project):
                    group = project == family
                    ask = project[a] == family
                    budget = int((base & ~y & group).sum()-(canonical & ~y & group).sum())
                    cutoff = cut_at_budget(scores[ask], y[a][ask], budget, -np.inf)
                    without_floor[np.flatnonzero(a)[ask]] |= scores[ask] > cutoff
                diagnostics[name] = dict(recovered=int(found.sum()), first_comment_free_tp_min_fp=false,
                                         no_score_floor_recovered=int((y & ~base & without_floor).sum()),
                                         no_score_floor_fp=int((~y & without_floor).sum()))
    return diagnostics


def public_audit(profile):
    report = profile['results']['canonical']
    rows = HERE/'public/canonical.csv'
    assert checksum(rows) == report['predictions_sha256']
    texts, y, projects, _ = dataset(public=True)
    y = y[:, 1].astype(bool)
    with rows.open() as src:
        forecasts = list(csv.DictReader(src))
    assert np.array_equal(y, [int(r['label_mat']) for r in forecasts])
    prediction = np.asarray([int(r['pred_MAT']) for r in forecasts], dtype=bool)
    parts = {'': range(len(y))}
    for p in np.unique(projects):
        parts[p] = np.flatnonzero(projects == p)
    represented = defaultdict(set)
    for text, gold in zip(texts, y, strict=True):
        represented[text].add(bool(gold))
    seen = np.asarray([len(represented[text]) > 1 for text in texts])
    from project_split import TOKEN
    comment = np.asarray([any(m['comment'] is not None for m in TOKEN.finditer(t)) for t in texts])
    states = {}
    for name, use in parts.items():
        yy, pp, cc, ss = (v[use] for v in (y, prediction, comment, seen))
        fn = yy & ~pp
        states[name or 'all'] = dict(tp=int((yy & pp).sum()), fp=int((~yy & pp).sum()), fn=int(fn.sum()),
                                     fn_with_comment=int((fn & cc).sum()), fn_without_comment=int((fn & ~cc).sum()),
                                     fn_with_identical_opposite_label=int((fn & ss).sum()))
    baseline = states['all']
    f1_v = report['quality']['vulnerability']['f1']
    cost = report['cost']
    def projected(k):
        tp = baseline['tp'] + k
        f1_m = 2*tp/(2*tp+baseline['fp']+baseline['fn']-k)
        return competition_formula((f1_v+f1_m)/2, cost['forward_seconds'], cost['analytic_gflops'])
    record = dict(groups=states, identical_mixed_labels_distinct_texts=sum(len(v)>1 for v in represented.values()),
                  score_if_all_visible_misses_fixed=projected(baseline['fn_with_comment']),
                  extra_mat_tp_for_097=next((k for k in range(baseline['fn']+1) if projected(k)>=.97), None))
    source = HERE/'visible_provenance.json'
    if source.exists():
        before = ROOT/'round18/mat_residual/source_audit_public.json'
        no_comments = {r['row'] for r in json.loads(before.read_text())['observations']
                       if r['mat_positive'] and r['observation'].startswith('marker_')}
        with_comments = {r['row'] for r in json.loads(source.read_text())['observations']
                         if r['split'] == 'public' and r['gold'] and not r['whole_word_tag_visible']
                         and r['observation'].startswith('marker_')}
        assert all(y[i] and not prediction[i] and not comment[i] for i in no_comments)
        assert all(y[i] and not prediction[i] and comment[i] for i in with_comments)
        found = len(no_comments | with_comments)
        record['provenance'] = dict(missed_with_confirmed_hidden_marker=found,
                                    with_visible_comments=len(with_comments), without_comments=len(no_comments),
                                    score_if_other_misses_fixed=projected(baseline['fn']-found),
                                    visible_source_sha256=checksum(source), original_source_sha256=checksum(before))
    return record


def main():
    results = json.loads((HERE/'results.json').read_text())
    audit = {k: recount(k, v) for k, v in results.items()}
    output = HERE/'public/results.json'
    if output.exists():
        audit['public'] = public_audit(json.loads(output.read_text()))
    write_json(HERE/'audit.json', audit)
    print(audit)


if __name__ == '__main__':
    main()
