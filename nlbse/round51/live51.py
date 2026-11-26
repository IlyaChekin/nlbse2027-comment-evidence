"""Benchmark a selected continuation through the complete one-encoder pipeline."""
from contextlib import nullcontext
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from eval51 import HERE, ROOT, FIT, checksum, json_at, write_json, load, f, configured, geometry, view
from train51 import added
from source41 import parts
from live40 import anchor, pipeline
from live35 import forward
from pipeline import PUBLIC, load as corpus_load

DEST=HERE/'live'


def policies():
    result=[]
    for name,dir_ in [('multitask','comparison'),('external','probe_comparison')]:
        file=HERE/dir_/'selection.json'
        card=json_at(file)
        assert card['csv_sha256']==checksum(file.with_name('comparison.csv'))
        for k,choice in card['selected'].items():
            if choice is not None:
                result.append((name,k,choice,checksum(file)))
    return result


def build_adapter(family,kind,choice,tokenizer,proof):
    mode=kind if kind!='original' else None
    net,head=configured('official',mode,tokenizer,proof)
    receipt={}
    if family=='external':
        p=HERE/'external_probe/official'
        weights=p/'weights.pt'
        card=json_at(p/'results.json')
        assert checksum(weights)==card['weights_sha256']
        tensors=torch.load(weights,map_location='cpu',weights_only=True)[kind+'/'+choice['view']]
        head=nn.Linear(2*net.width,1).cuda()
        head.load_state_dict(tensors)
        head.eval()
        receipt['probe_weights_sha256']=checksum(weights)
    else:
        receipt['continuation_sha256']=proof['arms'][kind]['sha256']
    return SimpleNamespace(encoder=net,tok=tokenizer,mode='control'),head,receipt


def session(family,kind,choice,proof,setup):
    _,runtime,original_stage,arguments=setup
    original_model=arguments[0]
    model,head,receipt=build_adapter(family,kind,choice,original_model.tok,proof)
    updated=(model,*arguments[1:])
    layer=runtime.record
    def run(texts,use):
        logits=np.empty((len(texts),3 if family=='multitask' else 1),np.float32)
        z_ops=0.

        def capture(ids,states,mask,side,original,aux,pool):
            nonlocal z_ops
            q=model.encoder.rule_head[0](pool)
            if family=='multitask':
                x=added(head,torch.cat((original,aux),1).float(),q)
                z_ops+=len(ids)*(2*model.encoder.width*64*2+64*3*2+10*model.encoder.width)
            else:
                with torch.autocast('cuda',enabled=False):
                    x=head(q.half().float())
                z_ops+=len(ids)*(4*model.encoder.width+10*model.encoder.width)
            logits[ids]=x.float().cpu().numpy()
            return layer(ids,states,mask,side,original,aux,pool)

        def correct(rows,rank,first,aux,context):
            current,_,ops=original_stage(rows,rank,first,aux,context)
            active=rows['old']&(rows['flag']>=0)
            branch=np.where(active,(rows['flag']==1).astype(np.int8),2)
            data=rows|{'ids':np.arange(len(rows['old'])),'signed':current,'observed_regime':branch}
            area=geometry(data,choice['panel'],choice['width'])
            if family=='multitask':
                new=view(logits,kind)[choice['view']][area]
            else:
                new=np.maximum(current[area],logits[area,0])
            mix=choice['mix']
            result=current.copy()
            result[area]=(1.-mix)*result[area]+mix*new-choice['cut']
            return result,area,ops+z_ops+5*area.sum()

        with patch.object(runtime,'record',capture) if use else nullcontext():
            answer=forward(texts,*(updated if use else arguments),correct if use else original_stage)
        started=perf_counter()
        cost=answer[-1]
        runtime.finish(cost)
        elapsed=perf_counter()-started
        cost['wall_seconds']+=elapsed
        cost['forward_seconds']+=elapsed
        return answer

    return run,receipt


def internal(family,kind,choice,join,proof,setup):
    frames=parts(load('official')[0])
    texts,labels,_,_,_=f.inputs()
    run,weights=session(family,kind,choice,proof,setup)
    review={}
    for phase,r in zip(('cal','held'),frames,strict=True):
        actual=[texts[i] for i in r['ids']]
        baseline,_,_,old=run(actual,False)
        test,_,_,cost=run(actual,True)
        assert np.array_equal(baseline[:,1],test[:,1])
        y=r['y']
        assert np.array_equal(y,labels[r['ids'],0])
        compare=f.comparison(y,baseline[:,0],test[:,0])
        review[phase]=dict(rows=len(y),change=compare,base_cost=old,candidate_cost=cost,
                           reference_differences=int(((r['signed']>0.)!=baseline[:,0]).sum()))
    supported=all(x['change']['f1_after']>x['change']['f1_before'] and
                  x['candidate_cost']['scanner_calls']<=x['base_cost']['scanner_calls']
                  for x in review.values())
    return dict(kind=kind,family=family,choice=choice,supported=supported,validation=review,selection_sha256=join)|weights


def public(support,proof,setup):
    family,kind,choice=(support[k] for k in ('family','kind','choice'))
    run,_=session(family,kind,choice,proof,setup)
    archive,source,y,reference=anchor()
    text,truth,_=corpus_load(PUBLIC)
    assert np.array_equal(truth,y)
    for v in (False,True):
        run(text[:4096],v)
    series=dict(reference=[],candidate=[])
    saved={}
    scores={}
    for i in range(5):
        for toggle in ((False,True) if i%2==0 else (True,False)):
            tag='candidate' if toggle else 'reference'
            answer,_,_,cost=run(text,toggle)
            if tag in saved:
                assert np.array_equal(saved[tag],answer)
            assert np.array_equal(answer[:,1],reference[:,1])
            if not toggle:
                assert np.array_equal(answer,reference)
            saved[tag]=answer
            result=f.metric(y,answer)
            score=f.competition_formula(result['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            series[tag].append(cost|{'score':score})
            scores[tag]=result
            print(i,tag,score,cost['forward_seconds'],cost['scanner_calls'],flush=True)
    table={}
    fn=DEST/'public_predictions.npz'
    np.savez_compressed(fn,truth=y,**saved)
    for k,z in series.items():
        cost={n:median(r[n] for r in z) for n in z[0] if n!='score'}
        value=f.competition_formula(scores[k]['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        table[k]=dict(quality=scores[k],cost=cost,score=value,trials=z)
    difference=[v['score']-r['score'] for v,r in zip(series['candidate'],series['reference'],strict=True)]
    historic=json_at(ROOT/'round38/release_summary.json')['score']
    eligible=bool(sum(x>0 for x in difference)>=4 and table['candidate']['score']>max(historic,table['reference']['score']))
    write_json(DEST/'public.json',dict(eligible=eligible,comparison=table,differences=difference,selection=support,
               baseline_archive_sha256=checksum(archive),reference_sha256=checksum(source),predictions_sha256=checksum(fn),
               dataset_sha256=checksum(PUBLIC),model_sha256=checksum(FIT/'official/fit.json'),
               code_sha256=checksum(Path(__file__))))
    return eligible


def main():
    options=policies()
    if not options:
        return
    DEST.mkdir(exist_ok=True)
    proof=json_at(FIT/'official/fit.json')
    setup=pipeline()
    evidence=[]
    for family,kind,policy,stamp in options:
        result=internal(family,kind,policy,stamp,proof,setup)
        evidence.append(result)
        print('internal',result,flush=True)
    tests=[b for b in evidence if b['supported']]
    chosen=max(tests,key=lambda v:min(r['change']['f1_after']-r['change']['f1_before'] for r in v['validation'].values())) if tests else None
    write_json(DEST/'selection.json',dict(candidates=evidence,selected=chosen,code_sha256=checksum(Path(__file__))))
    if chosen is not None:
        public(chosen,proof,setup)


if __name__=='__main__':
    main()
