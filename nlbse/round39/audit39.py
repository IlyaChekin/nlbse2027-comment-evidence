"""Recompute location-profile scores and isolate changes to the frozen scope."""
from pathlib import Path
from statistics import median

import numpy as np

from location39 import HERE, ROOT, f, PROGRAM, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes
from eval35 import cut


def check(folder,truth,groups,release):
    file=folder/'results.json'
    r=json_at(file)
    plan=HERE/'selection.json'
    changes=r['changes']
    assert r['choice']==json_at(plan)[r['arm']]
    source=ROOT/'round38/public_primary/candidate.npz'
    current=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    tests={'selection':plan,'public':PUBLIC,'anchor':source,'archive':current,'extended':PROGRAM,
           'first_hit':ROOT/'round32/first_hit_public.npz','readout':HERE/'comparison/official/models.joblib',
           'code':HERE/'live39.py','core':ROOT/'round35/live35.py','context':HERE/'context39.py'}
    receipt=r['receipt']
    for key,p in tests.items():
        assert checksum(p)==receipt[key+'_sha256']
    assert receipt['archive_sha256']==release['archive_sha256'] and receipt['scanner_comparisons']==7279
    forecast,scores={},{}
    for arm,sample in r['comparison'].items():
        path=folder/f'{arm}.npz'
        assert checksum(path)==sample['predictions_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['truth'],truth)
            forecast[arm]=z['pred'],z['scope'],z['margin']
        q=f.metric(truth,forecast[arm][0])
        assert q==sample['quality']
        for k,v in sample['cost'].items():
            assert median(t[k] for t in sample['trials'])==v
        scores[arm]=[f.competition_formula(q['mean']['f1'],a['forward_seconds'],a['analytic_gflops']) for a in sample['trials']]
        assert scores[arm]==[a['score'] for a in sample['trials']]
        assert len(scores[arm])==5 and median(scores[arm])==sample['score']
    before=forecast['reference'][0]
    after,scope,rank=forecast['candidate']
    with np.load(source) as z:
        assert np.array_equal(before,z['pred'])
    assert np.array_equal(before[:,1],after[:,1]) and np.array_equal(before[~scope],after[~scope])
    y=truth[:,0].astype(bool)
    previous=before[:,0]
    assert changes['frozen']==f.comparison(y,previous,after[:,0])
    for mode in ('fp','tp'):
        q=previous.copy()
        q[scope]=rank[scope]>cut(y[scope],previous[scope],rank[scope],mode)
        assert changes[mode]==f.comparison(y,previous,q)
    paired=[v-u for u,v in zip(scores['reference'],scores['candidate'],strict=True)]
    assert paired==r['paired']
    wins=sum(x>0 for x in paired)
    candidate=r['comparison']['candidate']
    ref=r['comparison']['reference']
    promoted=wins>=4 and candidate['score']>max(ref['score'],release['score']) and candidate['quality']['mean']['f1']>ref['quality']['mean']['f1']
    return dict(scope=int(scope.sum()),changes=int((before!=after).sum()),wins=wins,pairs=len(paired),
                result=promoted,score=candidate['score'],control=ref['score'],released_score=release['score'],
                ci=bootstrap_delta(truth,before,after,groups),profile_sha256=checksum(file))


def main():
    _,truth,meta=load(PUBLIC,audit=True)
    groups=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    release=json_at(ROOT/'round38/release_summary.json')
    outcomes={key:check(HERE/f'public_{key}',truth,groups,release) for key in ('chosen','best_local')}
    selected=max((k for k,row in outcomes.items() if row['result']),key=lambda k:outcomes[k]['score'],default=None)
    record=dict(selected=selected,outcomes=outcomes,code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',record)
    print(record,flush=True)


if __name__=='__main__':
    main()
