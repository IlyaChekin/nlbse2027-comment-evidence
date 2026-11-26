"""Verify the paired runtime and the exact preserved Round 34 comparator."""
from statistics import median

import numpy as np

from teacher35 import HERE, ROOT, f, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes


def check(stage):
    r34=ROOT/'round34'
    released=ROOT/'submission/NLBSE2027_round34_pooled_projection_tool.zip'
    reference=json_at(r34/'release_summary.json')
    assert reference['archive_sha256']==checksum(released)
    previous=r34/'public_approx/candidate.npz'
    path=HERE/stage/'results.json'
    record=json_at(path)
    receipt=record['receipt']
    assert receipt['public_sha256']==checksum(PUBLIC)
    assert receipt['reference_sha256']==checksum(previous)
    src=HERE/'source_snapshots/live_projection.py' if stage=='public' else HERE/'live35.py'
    assert receipt['code_sha256']==checksum(src)
    assert receipt['selection_sha256']==checksum(HERE/'selection.json')
    policy=json_at(HERE/'selection.json')
    assert record['selected']==(policy['by_stage']['sparse'] if stage=='public_sparse' else policy['selected'])
    _,truth,meta=load(PUBLIC,audit=True)
    marks={}
    for name,data in record['comparison'].items():
        file=path.parent/f'{name}.npz'
        assert checksum(file)==data['predictions_sha256']
        with np.load(file) as z:
            assert np.array_equal(z['truth'],truth)
            bits,mask=z['pred'],z['scope']
            marks[name]=bits,mask
        assert f.metric(truth,bits)==data['quality']
        pay=data['cost']
        assert all(median(r[k] for r in data['trials'])==v for k,v in pay.items())
        assert f.competition_formula(data['quality']['mean']['f1'],pay['forward_seconds'],pay['analytic_gflops'])==data['score']
    b,_=marks['reference']
    c,active=marks['candidate']
    with np.load(previous) as z:
        assert np.array_equal(b,z['pred'])
    assert np.array_equal(b[:,1],c[:,1]) and np.array_equal(b[~active],c[~active])
    assert f.comparison(truth[:,0],b[:,0],c[:,0])==record['quality']['frozen']
    assert record['comparison']['candidate']['cost']['scanner_calls']==record['comparison']['reference']['cost']['scanner_calls']
    wins=sum(q>0 for q in record['paired'])
    ctrl,new=[record['comparison'][k] for k in ('reference','candidate')]
    promoted=(wins>=4 and new['score']>max(reference['score'],ctrl['score']) and
              new['quality']['mean']['f1']>ctrl['quality']['mean']['f1'])
    clusters=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    diagnostics=dict(selected=promoted,candidate=new['score'],reference=ctrl['score'],release_score=reference['score'],
                     wins=wins,pairs=len(record['paired']),scope=int(active.sum()),changes=int((b!=c).sum()),
                     interval=bootstrap_delta(truth,b,c,clusters),source_sha256=checksum(path),
                     release_archive_sha256=reference['archive_sha256'])
    return diagnostics


if __name__=='__main__':
    stages={name:check(name) for name in ('public','public_sparse')}
    support=[name for name,v in stages.items() if v['selected']]
    selected=max(support,key=lambda s:stages[s]['candidate'],default=None)
    result=dict(selected=selected,stages=stages,code_sha256=checksum(HERE/'audit35.py'))
    write_json(HERE/'audit.json',result)
    print(result,flush=True)
