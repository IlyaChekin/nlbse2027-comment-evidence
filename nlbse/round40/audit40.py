"""Recompute the public timing scores and the fixed-scope confusion changes."""
from pathlib import Path
from statistics import median

import numpy as np

from live40 import HERE, ROOT, f, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes
from eval35 import cut


def check(arm,truth,groups,previous):
    loc=HERE/f'public_{arm}'
    path=loc/'results.json'
    r=json_at(path)
    plan=HERE/'selection.json'
    picked=json_at(plan)['picked']
    assert r['selection']==(picked['experts']['all'] if arm=='combined' else picked[arm])
    reference=ROOT/'round38/public_primary/candidate.npz'
    files={'selection':plan,'archive':ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip',
           'anchor':reference,'public':PUBLIC,'weights':HERE/'comparison/official/models.joblib',
           'code':Path(__file__).with_name('live40.py'),'runtime':ROOT/'round35/live35.py',
           'new_weights':HERE/'sparse/official/weights.joblib',
           'old_weights':ROOT/'round24/official/token.joblib'}
    for key,p in files.items():
        assert r['receipt'][key+'_sha256']==checksum(p)
    assert r['receipt']['archive_sha256']==previous['archive_sha256']
    y=truth[:,0].astype(bool)
    arms={}
    scores={}
    for key,part in r['comparison'].items():
        p=loc/f'{key}.npz'
        assert checksum(p)==part['predictions_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['truth'],truth)
            arms[key]=z['pred'],z['scope'],z['margin']
        q=f.metric(truth,arms[key][0])
        assert q==part['quality']
        scores[key]=[f.competition_formula(q['mean']['f1'],line['forward_seconds'],line['analytic_gflops'])
                     for line in part['trials']]
        assert scores[key]==[x['score'] for x in part['trials']]
        assert len(scores[key])==5 and median(scores[key])==part['score']
        for k,value in part['cost'].items():
            assert median(x[k] for x in part['trials'])==value
    base=arms['reference'][0]
    candidate,scope,rank=arms['candidate']
    with np.load(reference) as z:
        assert np.array_equal(base,z['pred'])
    assert np.array_equal(base[~scope],candidate[~scope])
    assert np.array_equal(base[:,1],candidate[:,1])
    initial=base[:,0]
    changes=r['changes']
    assert changes['frozen']==f.comparison(y,initial,candidate[:,0])
    for method in ('fp','tp'):
        offset=cut(y[scope],initial[scope],rank[scope],method)
        proposed=initial.copy()
        proposed[scope]=rank[scope]>offset
        assert changes[method]==f.comparison(y,initial,proposed)
    pairs=[b-a for a,b in zip(scores['reference'],scores['candidate'],strict=True)]
    assert pairs==r['paired']
    first=r['comparison']['reference']
    second=r['comparison']['candidate']
    wins=sum(q>0 for q in pairs)
    tested=second['score']
    release=first['score']
    quality=second['quality']['mean']['f1']-first['quality']['mean']['f1']
    promote=wins>=4 and tested>max(release,previous['score']) and quality>0.
    return dict(scope=int(scope.sum()),wins=wins,changes=int((base!=candidate).sum()),
              score=tested,control=release,promote=promote,ci=bootstrap_delta(truth,base,candidate,groups),
              profile_sha256=checksum(path))


def main():
    _,truth,meta=load(PUBLIC,audit=True)
    groups=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    previous=json_at(ROOT/'round38/release_summary.json')
    arms={name:check(name,truth,groups,previous) for name in ('chosen','combined')}
    selected=max((k for k,v in arms.items() if v['promote']),key=lambda k:arms[k]['score'],default=None)
    result=dict(selected=selected,profiles=arms,baseline=previous['score'],code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
