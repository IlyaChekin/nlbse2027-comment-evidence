"""Check predictions against the signed comparator and expose the source precision frontier."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROLES = ('dev','test_a','test_b')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def confusion(y,pred):
    return dict(tp=int((y & pred).sum()),fp=int((~y & pred).sum()),fn=int((y & ~pred).sum()))


def verify(mode,config):
    home = HERE/mode
    results = read(home/'results.json')
    thresholds = config['cutoffs'] if mode=='text' else config['frozen']
    output = {}
    for role in ROLES:
        path = home/f'{role}_predictions.npz'
        current = results[role]
        assert current['predictions_sha256']==sha(path)
        old_path = ROOT/'round31/mat/paired_provenance'/path.name
        p = np.load(path)
        old = np.load(old_path)
        assert np.array_equal(p['ids'],old['ids'])
        assert np.array_equal(p['base'],old['base'])
        y,base,group = p['y'],p['base'],p['group']
        assert np.array_equal(y,old['y'])
        baseline = confusion(y,base)
        assert all(current['baseline'][k]==v for k,v in baseline.items())
        recovered = {}
        for k,x in thresholds.items():
            score = p[k+'_score']
            assert np.isfinite(score).all()
            frozen = base | (score>x)
            assert np.array_equal(frozen,p[k+'_frozen'])
            matched = p[k+'_matched']
            if role=='dev':
                assert x==max(score[~y & ~base])
            for g in np.unique(group):
                selected = group==g
                cut = max(score[selected & ~y & ~base])
                assert np.array_equal(matched[selected],(base | (score>cut))[selected])
                assert (~y & base & selected).sum()==(~y & matched & selected).sum()
            c,f = confusion(y,matched),confusion(y,frozen)
            summary = current['results'][k]
            assert all(summary['matched'][key]==v for key,v in c.items())
            assert all(summary['frozen'][key]==v for key,v in f.items())
            recovered[k] = dict(matched_tp=c['tp']-baseline['tp'],
                                frozen_tp=f['tp']-baseline['tp'],frozen_fp=f['fp']-baseline['fp'])
        output[role] = dict(rows=len(y),baseline=baseline,recovered=recovered)
        if mode=='source_context':
            rank = p['owner_aligned_score']
            points = []
            for threshold in sorted(set(rank),reverse=True):
                chosen = base | (rank>threshold)
                d = confusion(y,chosen)
                point = dict(tp=d['tp']-baseline['tp'],fp=d['fp']-baseline['fp'])
                if point not in points:
                    points.append(point)
            output[role]['source_frontier'] = points
    return output


def main():
    v = read(HERE/'text/policy.json')
    s = read(HERE/'source_context/receipt.json')
    for mode,stats in [('text41',v),('source_context41',s)]:
        assert stats['code_sha256']==sha(HERE/(mode+'.py'))
        assert stats['partition_sha256']==sha(ROOT/'round13/partition.npz')
    assert v['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
    assert v['weights_sha256']==sha(HERE/'text/heads.joblib')
    assert v['fit_source_sha256']==sha(ROOT/'round31/mat/fit_provenance.json')
    assert s['inventory_origin_sha256']==sha(ROOT/'round18/mat_residual/source_audit.json')
    score = read(ROOT/'round38/release_summary.json')
    m = score['MAT']
    def project(n):
        tp = m['tp']+n
        return score['score']+.3*(2*tp/(2*tp+m['fp']+m['fn']-n)-score['quality']['MAT']['f1'])
    need = next(n for n in range(m['fn']+1) if project(n)>=.97)
    assert need==score['changes_to_target_at_fixed_cost']['tp_m']
    report = dict(text=verify('text',v),source_context=verify('source_context',s),
                  target=dict(round38=score['score'],gap=score['target_gap'],needed_mat_tp_at_fixed_cost=need,
                              projected_with_previous=project(need-1),projected_at_minimum=project(need)))
    comparison = HERE.parent/'comparison'
    if comparison.exists():
        select = read(comparison/'selection.json')
        assert select['csv_sha256']==sha(comparison/'comparison.csv')
        assert select['code_sha256']==sha(HERE.parent/'evaluate41.py')
        for role,checksum in select['sources'].items():
            assert checksum==sha(comparison/role/'results.json')
            receipt = read(comparison/role/'results.json')
            assert receipt['predictions_sha256']==sha(comparison/role/'predictions.npz')
            assert receipt['head_receipt_sha256']==sha(HERE.parent/'heads'/role/'receipt.json')
        report['vulnerability_selection'] = select['groups']
    (HERE/'verified.json').write_text(json.dumps(report,indent=2)+'\n')
    print('verified project counts, thresholds and signed predictions')


if __name__=='__main__':
    main()
