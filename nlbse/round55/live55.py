"""Profile the selected token readout inside the frozen Round 38 pipeline."""
from contextlib import nullcontext
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from pick55 import DEST as BOARD
from span55 import FIT,HERE,ROOT,WIDTH,checksum,f,json_at,load,readout,write_json
from eval55 import head_views,parts
from eval51 import geometry
from live35 import forward
from live40 import anchor, pipeline
from pipeline import PUBLIC,load as corpus_load

DEST=HERE/'live'


def session(choice,setup):
    _,runtime,original,args=setup
    name=choice['arm']
    receipt=FIT/'official/fit.json'
    head=readout(receipt.parent,name,args[0].encoder,json_at(receipt))
    keep=runtime.record

    def run(texts,enabled):
        p=np.empty((len(texts),2),np.float32)
        old=np.empty_like(p)
        bill=0
        def capture(ids,state,mask,side,base,aux,pool):
            nonlocal bill
            prior=torch.stack((base[:,0],aux[:,0]),1)
            q,_=head(state,mask,prior)
            p[ids]=q.float().cpu().numpy()
            old[ids]=prior.float().cpu().numpy()
            width=state.shape[-1]
            num=len(ids) if name=='pooled' else mask.numel()
            bill+=num*(2*width*WIDTH+12*WIDTH+20)
            return keep(ids,state,mask,side,base,aux,pool)

        def stage(row,rank,first,aux,ctx):
            baseline,_,prior=original(row,rank,first,aux,ctx)
            active=row['old']&(row['flag']>=0)
            branch=np.where(active,(row['flag']==1).astype(np.int8),2)
            data=row|{'ids':np.arange(len(baseline)), 'signed':baseline,'observed_regime':branch}
            use=geometry(data,choice['panel'],choice['width'])
            cast=head_views(p,old)[choice['view']][use]
            delta=choice['mix']
            result=baseline.copy()
            result[use]=(1.-delta)*baseline[use]+delta*cast-choice['cut']
            return result,use,prior+bill+5*use.sum()

        with patch.object(runtime,'record',capture) if enabled else nullcontext():
            result=forward(texts,*args,stage if enabled else original)
        started=perf_counter()
        cost=result[-1]
        runtime.finish(cost)
        elapsed=perf_counter()-started
        cost['wall_seconds']+=elapsed
        cost['forward_seconds']+=elapsed
        return result[0],cost

    return run,checksum(receipt.parent/(name+'.pt'))


def profile(selected,setup):
    run,w=session(selected,setup)
    archive,prior,expected,reference=anchor()
    texts,y,_=corpus_load(PUBLIC)
    assert np.array_equal(expected,y)
    for toggle in (False,True):
        run(texts[:4096],toggle)
    stored={}
    costs={'reference':[],'candidate':[]}
    for i in range(5):
        for cand in ((False,True) if i%2==0 else (True,False)):
            tag='candidate' if cand else 'reference'
            values,pay=run(texts,cand)
            if tag in stored:
                assert np.array_equal(values,stored[tag])
            assert np.array_equal(values[:,1],reference[:,1])
            if not cand:
                assert np.array_equal(values,reference)
            stored[tag]=values
            q=f.metric(y,values)
            score=f.competition_formula(q['mean']['f1'],pay['forward_seconds'],pay['analytic_gflops'])
            costs[tag].append(pay|{'score':score})
            print(i,tag,score,pay['forward_seconds'],flush=True)
    points={}
    for tag,a in stored.items():
        m=f.metric(y,a)
        t={k:median(v[k] for v in costs[tag]) for k in costs[tag][0] if k!='score'}
        value=f.competition_formula(m['mean']['f1'],t['forward_seconds'],t['analytic_gflops'])
        points[tag]=dict(score=value,quality=m,cost=t,trials=costs[tag])
    diff=[b['score']-a['score'] for a,b in zip(costs['reference'],costs['candidate'],strict=True)]
    wins=sum(d>0 for d in diff)
    release=json_at(ROOT/'round38/release_summary.json')
    usable=bool(wins>=4 and points['candidate']['score']>max(release['score'],points['reference']['score']))
    path=DEST/'public_predictions.npz'
    np.savez_compressed(path,truth=y,**stored)
    write_json(DEST/'public.json',dict(score=points,paired_differences=diff,paired_wins=int(wins),passed=usable,
               selected=selected,predictions_sha256=checksum(path),archive_sha256=checksum(archive),
               dataset_sha256=checksum(PUBLIC),base_predictions_sha256=checksum(prior),weights_sha256=w,
               code_sha256=checksum(Path(__file__))))


def main():
    DEST.mkdir(exist_ok=True)
    path=BOARD/'selection.json'
    plan=json_at(path)
    assert plan['csv_sha256']==checksum(path.with_name('comparison.csv'))
    assert plan['code_sha256']==checksum(HERE/'pick55.py')
    options=[v for v in plan['selected'].values() if v]
    table=[]
    if options:
        setup=pipeline()
        source,_,_=load('official')
        texts,labels,projects,files,_=f.inputs()
        seen={(projects[i],files[i]) for i in source['oof']['ids'] if files[i]}
        rows=parts(source)
        for choice in options:
            run,weights=session(choice,setup)
            test={}
            for phase,view in zip(('cal','held'),rows,strict=True):
                ids=view['ids']
                text=[texts[i] for i in ids]
                y=view['y']
                assert np.array_equal(y,labels[ids,0])
                before,base_cost=run(text,False)
                after,new_cost=run(text,True)
                assert np.array_equal(before[:,1],after[:,1]) and base_cost['scanner_calls']==new_cost['scanner_calls']
                unseen=np.asarray([bool(files[i]) and (projects[i],files[i]) not in seen for i in ids])
                test[phase]=dict(change=f.comparison(y,before[:,0],after[:,0]),
                    new_files=f.comparison(y[unseen],before[unseen,0],after[unseen,0]),unseen_file_rows=int(unseen.sum()),
                    reference_cost=base_cost,candidate_cost=new_cost)
            passed=all(r['change']['f1_after']>r['change']['f1_before'] and
                       r['new_files']['f1_after']>=r['new_files']['f1_before'] for r in test.values())
            table.append(dict(policy=choice,reproduced=passed,proof=test,weights_sha256=weights))
            print(choice['arm'],passed,{k:v['change'] for k,v in test.items()},flush=True)
    approved=[b for b in table if b['reproduced']]
    winner=max(approved,key=lambda b:b['proof']['cal']['change']['f1_after']) if approved else None
    write_json(DEST/'selection.json',dict(selected=winner,candidates=table,offline_sha256=checksum(path),
               code_sha256=checksum(Path(__file__))))
    if winner is not None:
        profile(winner['policy'],setup)


if __name__=='__main__':
    main()
