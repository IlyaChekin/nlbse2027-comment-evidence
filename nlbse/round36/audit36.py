"""Recount all MAT decisions, verify thresholds, and join the public provenance audit."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARTS = ('dev', 'test_a', 'test_b')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def counts(truth, base, prediction):
    assert not (base & ~prediction).any()
    added = prediction & ~base
    return dict(recovered=int((truth & added).sum()), fp=int((~truth & added).sum()))


def audit(folder, source):
    policy = read(folder/'policy.json')
    assert policy['code_sha256'] == sha(HERE/source)
    assert policy['model_sha256'] == sha(folder/'heads.joblib')
    assert policy['partition_sha256'] == sha(ROOT/'round13/partition.npz')
    assert policy['data_manifest_sha256'] == sha(ROOT/'data/manifest.json')
    if 'provenance_sha256' in policy:
        assert policy['provenance_sha256'] == sha(ROOT/'round31/mat/fit_provenance.json')
        assert policy['dependencies_sha256'] == sha(HERE/'syntax_requirements.txt')
    table = read(folder/'results.json')
    summary = {}
    for part in PARTS:
        row = table[part]
        prediction_file = folder/f'{part}_predictions.npz'
        reference = ROOT/'round31/mat/paired_provenance'/f'{part}_predictions.npz'
        assert row['predictions_sha256'] == sha(prediction_file)
        assert row['reference_sha256'] == sha(reference)
        with np.load(prediction_file, allow_pickle=True) as z, np.load(reference, allow_pickle=True) as old:
            for name in ('ids','y','base','group','comment'):
                assert np.array_equal(z[name], old[name])
            truth, base, group, comment = [z[key] for key in ('y','base','group','comment')]
            miss = truth & ~base
            negative = ~truth & ~base
            assert not (~truth & base).any()
            metrics = {}
            for arm, threshold in policy['cutoffs'].items():
                score = z[arm+'_score']
                assert np.isfinite(score).all()
                if part == 'dev':
                    assert threshold == score[negative].max()
                frozen = base | (score > threshold)
                assert np.array_equal(frozen, z[arm+'_frozen'])
                diagnostic = base.copy()
                for family in np.unique(group):
                    member = group == family
                    t = score[member & negative].max()
                    diagnostic[member] |= score[member] > t
                    assert t == row['results'][arm]['projects'][family]['cut']
                assert np.array_equal(diagnostic, z[arm+'_matched'])
                metrics[arm] = dict(frozen=counts(truth,base,frozen), matched=counts(truth,base,diagnostic),
                                    matched_without_comment=int((miss & diagnostic & ~comment).sum()))
                assert metrics[arm]['matched']['fp'] == 0
                for name in ('frozen','matched'):
                    result = row['results'][arm][name]
                    assert result['recovered'] == metrics[arm][name]['recovered']
                    assert result['delta_fp'] == metrics[arm][name]['fp']
            summary[part] = dict(rows=len(truth), misses=int(miss.sum()),
                                 absent_comment=int((miss & ~comment).sum()), arms=metrics)
    return summary


def score_limits():
    release = read(ROOT/'round34/release_summary.json')
    with np.load(ROOT/'round34/public_approx/candidate.npz') as z:
        failed = set(np.flatnonzero(z['truth'][:,1].astype(bool) & ~z['pred'][:,1]))
    assert len(failed) == release['MAT']['fn']
    without_path = ROOT/'round18/mat_residual/source_audit_public.json'
    visible_path = ROOT/'round26/mat/visible_provenance.json'
    marker = ('marker_inside_matched_span','marker_in_adjacent_preceding_comment')
    without = {r['row'] for r in read(without_path)['observations']
               if r['mat_positive'] and r['observation'] in marker}
    visible = {r['row'] for r in read(visible_path)['observations']
               if r['split']=='public' and r['gold'] and r['observation'] in marker and not r['whole_word_tag_visible']}
    assert not without & visible and (without | visible) <= failed
    tp = release['MAT']['tp']
    total = tp + len(failed)
    score = release['score']
    def scenario(recover):
        mat = 2*(tp+recover)/(total+tp+recover)
        return score + .3*(mat - release['quality']['MAT']['f1'])
    need = next(n for n in range(len(failed)+1) if scenario(n) >= .97)
    assert need == release['changes_to_target_at_fixed_cost']['tp_m']
    return dict(released_score=score, baseline_mat=release['MAT'], public_misses=len(failed),
                source_marker_absent_without_other_comments=len(without),
                source_marker_absent_with_other_comments=len(visible),
                source_marker_absent_confirmed=len(without|visible),
                other_misses=len(failed-without-visible),
                tp_needed_with_fixed_cost_and_vulnerability=need,
                tp_needed_scenario=scenario(need), scenario_for_recovering_all_other_misses=scenario(len(failed-without-visible)),
                source_audit_sha256=[sha(without_path),sha(visible_path)], release_sha256=sha(ROOT/'round34/release_summary.json'))


if __name__ == '__main__':
    summary = dict(clauses=audit(HERE,'comment_signal.py'), syntax=audit(HERE/'syntax','syntax_signal.py'),
                   public=score_limits())
    (HERE/'audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['public'], indent=2))
