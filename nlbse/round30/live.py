"""Replay the verified tool and benchmark one preselected student on public validation."""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys
import time
from types import SimpleNamespace

import joblib
import numpy as np

from score_students import HERE, PARTS, f, checksum, json_at, write_json, load_heads
from reference29 import R28
from rank30 import rerank, student_bits
from student_runtime import predict
from evaluate_rules import infer
from union_scanner import scan, PROGRAM
from normalize_code import batch as canonicalize
from length_eval import R24, recipe
from pipeline import PUBLIC, load
from positive_runtime import apply as correct_positive
sys.path[:0]=[str(R28.parent/'round26/mat'),str(R28.parent/'round21/mat')]
from mat_runtime import canonical_mat

ROOT=R28.parent
OUT=HERE/'public'


def freeze():
    eligible={}
    table={}
    for stage in ('all','delta'):
        directory=HERE/'comparisons'/stage
        paths=[directory/name/'results.json' for name in PARTS]
        a,b,c=[json_at(k) for k in paths]
        table[stage]={k.parent.name:checksum(k) for k in paths}
        for key,row in c['options'].items():
            gains=[x['options'][key]['matched']['delta_tp'] for x in (a,b)]
            if min(gains)<=0:
                continue
            for arm,v in row['cuts'].items():
                x,y=v['cal'],v['frozen']
                if v['proxy']<=0 or x['delta_tp']<0 or x['delta_fp']>0 or y['f1_after']<y['f1_before']:
                    continue
                candidate=f'{stage}:{key}:{arm}'
                eligible[candidate]=dict(v,project_tp=gains,mode=row['mode'],stage=stage,width=row['width'],
                                           mix=row['mix'],calls=row['cal_calls'],head_sha256=c['heads'][row['mode']],
                                           readout_sha256=c['readout_sha256'],cost=c['timing'])
    mode=max(eligible,key=lambda k:(eligible[k]['proxy'],-eligible[k]['calls']),default='reference')
    result=dict(selected=mode,options=eligible,table_sha256=table)
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'policy.json',result)
    print('selected',mode,eligible.get(mode),flush=True)
    return result


def benchmark(signed):
    selected=signed['selected']
    if selected=='reference':
        return
    cfg=signed['options'][selected]
    record=R28/'recover_public'
    current=json_at(record/'release_selection.json')['selected']
    report=json_at(record/'results.json')
    data=record/report['results'][current]['csv']
    if checksum(data)!=report['results'][current]['predictions_sha256'] or checksum(PUBLIC)!=report['public_sha256']:
        raise ValueError('Source evaluation data changed')
    archive=np.array([[int(r['pred_vulnerability']),int(r['pred_MAT'])]
                      for r in csv.DictReader(data.open())],dtype=bool)
    p29=json_at(record/'policy.json')
    variant29=p29['variants'][current]
    model29=R28/'recover_calibrated/official/heads.joblib'
    assert checksum(model29)==variant29['source_sha256']
    recall=joblib.load(model29)['compact']
    r27=ROOT/'round27/public'
    p27=json_at(r27/'policy.json')
    correction=p27['supported'][p27['selected']]
    head=ROOT/'round27/joint/official/weights.joblib'
    assert checksum(head)==p27['weights_sha256']
    positive=joblib.load(head)[correction['weights_key']]
    cfg24=json_at(R24/'official/tuned/policy.json')
    package=R24/'official/tuned/models.joblib'
    assert checksum(package)==cfg24['models_sha256']
    weights=joblib.load(package)['token']
    variant=cfg24['candidates'][cfg24['selected']]
    r25=ROOT/'round25/views'
    p25=json_at(r25/'public/policy.json')
    section=p25['support'][p25['selected']]
    stored=r25/'official/calibrator.joblib'
    assert checksum(stored)==p25['calibrator_sha256']
    h=joblib.load(stored)[section['region']]
    path=HERE/'comparisons'/cfg['stage']/'official/readouts.joblib'
    assert checksum(path)==cfg['readout_sha256']
    classifier=joblib.load(path)[cfg['mode']]
    model,tok,arms,received=load_heads('official',(cfg['mode'],))
    assert received[cfg['mode']]==cfg['head_sha256']
    pipe=SimpleNamespace(encoder=model,tok=tok,mode='control')
    canonical_pipe=SimpleNamespace(encoder=model.encoder,tok=tok,mode='control')
    core,gate,regression=recipe()
    texts,y,_=load(PUBLIC)
    overlay=variant29['setting']
    normalize=f.pinned()[0]

    def forward(name,text):
        candidate=name!='reference'
        if candidate:
            base_start=time.perf_counter()
            raw,aux,extras,cost=predict(model,tok,text,tuple(arms.values()))
            view=(raw[:,0]>normalize['lower'])&(raw[:,0]<normalize['upper'])
            if view.any():
                t=time.perf_counter()
                masked=canonicalize([s for s,on in zip(text,view,strict=True) if on])
                cost['forward_seconds']+=time.perf_counter()-t
                q,payment=infer(canonical_pipe,masked,encoding=f.tokenize)
                raw[view,0]=.5*(raw[view,0]+q[:,0])
                for field in ('forward_seconds','wall_seconds','analytic_gflops'):
                    cost[field]+=payment[field]
            cost['routed']=int(view.sum())
            cost['wall_seconds']=time.perf_counter()-base_start
        else:
            raw,aux,cost=f.infer_student(pipe,text,normalize,columns=22,return_all=True)
            cost['extra_gflops']=0.
        now=time.perf_counter()
        values=np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
        est=f.estimate(regression,values)
        v,no,yes,asked=f.choose(values,est,gate,core['gate'])
        known=np.zeros(len(raw),dtype=bool)
        flag=np.zeros_like(known)
        if asked.any():
            ids=scan([s for s,on in zip(text,asked,strict=True) if on])
            known[asked]=ids>=0
            flag[asked]=ids==1
            v[known]=np.where(flag[known],yes[known],no[known])
        state=dict(p=v.astype(np.float32),w=est['w'],no=no,yes=yes,known=known,flag=flag)
        margin=np.full(len(raw),np.nan) if candidate else None
        before,_,_,ops=rerank(text,values,state,core['threshold'],variant,weights,section,h,overlay,recall,margin)
        result,_,_,cash=correct_positive(text,values,state,core['threshold'],before,correction['online'],positive)
        scope=np.zeros(len(raw),dtype=bool)
        ranking=np.empty(0)
        if candidate:
            result,scope,ranking=student_bits(result,margin,extras[:,0],aux,classifier,cfg)
        bit=np.column_stack((result,canonical_mat(text)))
        step=time.perf_counter()-now
        cost['forward_seconds']+=step
        cost['wall_seconds']+=step
        cost['analytic_gflops']+=(ops['base_ops']+ops['extra_ops']+cash['nnz']+8*int(scope.sum()))/1e9
        return bit,scope,ranking,cost|{'new_rows':int(scope.sum())}

    keys=('reference',selected)
    batches={k:[] for k in keys}
    result={}
    for k in keys:
        forward(k,texts[:2048])
    for trial in range(5):
        for k in keys if trial%2==0 else keys[::-1]:
            b,s,r,c=forward(k,texts)
            if not np.array_equal(b[:,1],archive[:,1]) or np.any(b[~s]!=archive[~s]):
                raise ValueError('Compared outputs differ outside the exact routed negative scope')
            if k in result and not np.array_equal(result[k][0],b):
                raise ValueError('Predictions differ between repeated calls')
            result[k]=b,s,r
            batches[k].append(c)
            print(trial,k,c,flush=True)
    reports={}
    current=f.metric(y,archive)
    score=lambda m,c:f.competition_formula(m['mean']['f1'],c['forward_seconds'],c['analytic_gflops'])
    pairs={}
    for k,(b,s,r) in result.items():
        c={key:median(v[key] for v in batches[k]) for key in batches[k][0]}
        p=archive[:,0].copy()
        if s.any():
            budget=int((p[s]&(y[s,0]==0)).sum())
            p[s],_=f.exact_budget(r,y[s,0],budget)
        file=OUT/('reference.csv' if k=='reference' else 'candidate.csv')
        metric=f.metric(y,b)
        with file.open('w',newline='') as buf:
            w=csv.writer(buf)
            w.writerow(('row','label_vulnerability','label_mat','pred_vulnerability','pred_MAT','scoped','matched'))
            w.writerows((i,*g,*v,int(e),int(t)) for i,(g,v,e,t) in enumerate(zip(y,b.astype(int),s,p,strict=True)))
        reports[k]=dict(score=score(metric,c),quality=metric,timing=c,trials=batches[k],
                        frozen=f.comparison(y[:,0],archive[:,0],b[:,0]),matched=f.comparison(y[:,0],archive[:,0],p),
                        predictions_sha256=checksum(file))
        pairs[k]=[score(metric,x)-score(current,z) for x,z in zip(batches[k],batches['reference'],strict=True)]
    write_json(OUT/'results.json',dict(results=reports,pairs=pairs,selected=selected,policy_sha256=checksum(OUT/'policy.json'),
              public_sha256=checksum(PUBLIC),scanner_sha256=checksum(PROGRAM)))
    print({k:v['score'] for k,v in reports.items()},pairs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--select-only',action='store_true')
    a=p.parse_args()
    policy=freeze()
    if not a.select_only:
        benchmark(policy)
