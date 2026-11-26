"""Verify frozen and equal-FP MAT predictions and account for hidden annotation context."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

from shape48 import AUDIT, DEST, HERE, ROOT, PARTS, canonical_mat, checksum, dataset, split, write_json

sys.path.insert(0, str(ROOT))
from score_runtime import competition_formula


def read(path):
    return json.loads(path.read_text())


def count(y, base, called):
    assert (called >= base).all()
    return dict(tp=int((y & called).sum()), fp=int((~y & called).sum()),
                recovered=int((y & ~base & called).sum()))


def load_shape():
    texts, labels, group, files = dataset()
    folds = split()
    policy, results = (read(DEST/n) for n in ('policy.json','results.json'))
    artifacts = {'code_sha256': HERE/'shape48.py', 'audit_sha256': AUDIT,
                 'partition_sha256': ROOT/'round13/partition.npz',
                 'data_sha256': ROOT/'data/manifest.json', 'weights_sha256': DEST/'weights.joblib'}
    assert all(policy[k] == checksum(p) for k,p in artifacts.items())
    out = {}
    receipt = {}
    for part in PARTS:
        path = DEST/f'{part}_predictions.npz'
        previous = ROOT/'round41/mat/text'/f'{part}_predictions.npz'
        assert results[part]['predictions_sha256'] == checksum(path)
        with np.load(path) as data, np.load(previous) as ref:
            ids,y,base,p,comment = (data[k] for k in ('ids','y','base','group','comment'))
            assert np.array_equal(ids, folds[part]) and np.array_equal(y, labels[ids,1])
            assert np.array_equal(p,group[ids])
            assert np.array_equal(base,ref['base']) and np.array_equal(y,ref['y'])
            assert np.array_equal(data['previous_joint_score'],ref['joint_score'])
            if part == 'dev':
                negative = ~y & ~base
                assert all(policy['cutoffs'][name] == data[name+'_score'][negative].max()
                           for name in policy['cutoffs'])
            summaries = {}
            for name,t in policy['cutoffs'].items():
                score = data[name+'_score']
                assert np.isfinite(score).all()
                freeze = base | (score>t)
                assert np.array_equal(data[name+'_frozen'], freeze)
                by_project = {}
                recovered = []
                for project in np.unique(p):
                    m = p == project
                    edge = score[m & ~y & ~base].max()
                    select = base[m] | (score[m]>edge)
                    assert np.array_equal(data[name+'_matched'][m],select)
                    summary = count(y[m],base[m],select)
                    assert summary['fp']==int((base[m] & ~y[m]).sum())
                    recovered.extend(ids[m][y[m] & ~base[m] & select].tolist())
                    by_project[project] = summary
                result = count(y,base,freeze)
                assert all(result[k] == results[part]['results'][name]['frozen'][k] for k in result)
                assert len(recovered)==results[part]['results'][name]['matched']['recovered']
                summaries[name] = dict(frozen=result, same_fp_recovered=len(recovered),
                                       recovered_rows=recovered, same_fp_projects=by_project)
            out[part] = {key: data[key] for key in data.files}
            receipt[part] = dict(predictions_sha256=checksum(path), rows=len(ids),
                                 baseline=count(y,base,base), misses=int((y & ~base).sum()),
                                 comment_free=int((y & ~base & ~comment).sum()), arms=summaries)
    return (texts,labels,group,files),out,receipt


def provenance(out):
    previous = ROOT/'round18/mat_residual/source_audit.json'
    comment = ROOT/'round26/mat/visible_provenance.json'
    checked = defaultdict(set)
    source_data = read(previous)
    assert source_data['partition_sha256']==checksum(ROOT/'round13/partition.npz')
    for item in source_data['observations']:
        if item['mat_positive'] and item['observation'].startswith('marker_'):
            for part in ('test_a','test_b'):
                checked[part].add(item['row'])
    for item in read(comment)['observations']:
        if (item['split'] in ('test_a','test_b') and item['gold'] and not item['whole_word_tag_visible']
                and item['observation'].startswith('marker_')):
            checked[item['split']].add(item['row'])
    groups = {}
    for part in ('test_a','test_b'):
        data = out[part]
        missed = data['y'] & ~data['base']
        m = missed & np.isin(data['ids'],list(checked[part]))
        # Only exact provenance matches on a pinned snapshot are counted here.
        groups[part] = dict(confirmed_omission=int(m.sum()), without_comment=int((m & ~data['comment']).sum()),
                            with_other_comment=int((m & data['comment']).sum()),
                            rows=data['ids'][m].tolist())
    return groups, {p.name:checksum(p) for p in (previous,comment)}


def public_projection():
    text,label,_,_ = dataset(public=True)
    public = ROOT/'round38/release_summary.json'
    record = read(public)
    m,a,c = (record[k] for k in ('MAT','quality','cost'))
    profile = ROOT/'round38/release_verification'
    receipt = read(profile/'receipt.json')
    prediction = profile/'release_predictions.csv'
    assert receipt['predictions_sha256'] == checksum(prediction)
    with prediction.open(newline='') as handle:
        picks = list(csv.DictReader(handle))
    assert len(picks) == receipt['rows'] == len(text)
    assert all(set(row) == {'vulnerability','MAT'} for row in picks)
    predicted = np.array([bool(int(row['MAT'])) for row in picks])
    assert np.array_equal(canonical_mat(text),predicted)
    gold = label[:,1].astype(bool)
    misses = gold & ~predicted
    assert int(misses.sum())==m['fn'] and int((gold & predicted).sum())==m['tp']
    assert int((~gold & predicted).sum())==m['fp']
    prior = ROOT/'round26/mat/audit.json'
    s = read(prior)['public']
    fixed = s['provenance']['missed_with_confirmed_hidden_marker']
    def at(k,fp=0):
        f1 = 2*(m['tp']+k)/(2*m['tp']+m['fp']+m['fn']+k+fp)
        return competition_formula(.5*(f1+a['vulnerability']['f1']),c['forward_seconds'],c['analytic_gflops'])
    assert abs(at(0)-record['score']) < 1e-10
    needed = next(k for k in range(m['fn']+1) if at(k)>=.97)
    assert needed==record['changes_to_target_at_fixed_cost']['tp_m']
    return dict(verified_score=record['score'], target=.97, gap=record['target_gap'],
                required_zero_fp_returns=needed, fn=m['fn'], confirmed_hidden=fixed,
                unconfirmed=m['fn']-fixed, score_if_all_unconfirmed_recovered=at(m['fn']-fixed),
                score_if_only_one_more_recovered=at(1), score_if_15_more_recovered=at(15),
                score_if_16_more_recovered=at(16), public_predictions_sha256=receipt['predictions_sha256'],
                archive_sha256=record['archive_sha256'], provenance_sha256=checksum(prior))


def main():
    _,pred,verified = load_shape()
    observed,inputs = provenance(pred)
    record = dict(verification=verified, held_source_provenance=observed, source_inputs=inputs,
                  public=public_projection(), code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',record)
    print({name:{'misses':entry['misses'],'without_comment':entry['comment_free'],
                  'best_same_fp':max(r['same_fp_recovered'] for r in entry['arms'].values())}
           for name,entry in verified.items()}, flush=True)
    print({name:{k:v for k,v in group.items() if k!='rows'} for name,group in observed.items()}, flush=True)
    print(record['public'], flush=True)


if __name__ == '__main__':
    main()
