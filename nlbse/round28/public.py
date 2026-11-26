"""Select untuned candidates and measure the real cost on full public validation."""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np

from pairs import HERE, ROOT, f, checksum, json_at, write_json
from paired_context import R27
from negative_runtime import rerank
from positive_runtime import apply as adjust_positive
from union_scanner import scan, PROGRAM
from length_eval import R24,recipe
from pipeline import load, PUBLIC
sys.path[:0]=[str(ROOT/'round26/mat'),str(ROOT/'round21/mat')]
from mat_runtime import canonical_mat

OUT=HERE/'public'
R25=ROOT/'round25/views'


def freeze():
    all_rows={}
    receipts={}
    for backend in ('head','sparse','canonical'):
        d=HERE/'studies'/backend
        paths=[d/q/'results.json' for q in ('test_a','test_b','official')]
        a,b,c=[json_at(p) for p in paths]
        receipts[backend]={p.parent.name:checksum(p) for p in paths}
        for key,row in c['options'].items():
            g=[x['options'][key]['matched']['delta_tp'] for x in (a,b)]
            if min(g)<0 or not any(g):
                continue
            for name,stat in row['cuts'].items():
                review=stat['frozen']
                cal=stat['cal']
                if stat['proxy']>0 and review['f1_after']>=review['f1_before']:
                    tag=f'{backend}:{key}:{name}'
                    all_rows[tag]=dict(stat,project_tp=g,source_sha256=c['weights_sha256'],mode=row['mode'],
                                        cal_calls=row['cal_calls'],setting=dict(kind=backend,width=row['width'],
                                                                              mix=row['mix'],cut=stat['cut']))
    variants={}
    for name, restrict in (('score',False),('retain_tp',True)):
        possible=[k for k,row in all_rows.items() if not restrict or row['cal']['delta_tp'] >= 0]
        variants[name]=max(possible,key=lambda k:all_rows[k]['proxy'],default='reference')
    options=list(dict.fromkeys(k for k in variants.values() if k!='reference'))
    record=dict(family=variants,options=options,variants=all_rows,tables_sha256=receipts,
                current_sha256=checksum(R27/'public/candidate.csv'))
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'policy.json',record)
    print('Options:',[(k,all_rows[k]) for k in options],flush=True)
    return record


def run(frozen, out=OUT, recovery=False):
    picked=frozen['options']
    if not picked:
        return
    previous=json_at(R27/'public/results.json')
    current=R27/'public/candidate.csv'
    nv=previous['selected']
    if checksum(current)!=previous['results'][nv]['predictions_sha256'] or checksum(PUBLIC)!=previous['public_sha256']:
        raise ValueError('The release comparison source changed')
    if recovery:
        target=HERE/'threshold'
        study=json_at(target/'results.json')
        winner=study['results']['calibration_only']
        current=target/winner['csv']
        if checksum(current)!=winner['predictions_sha256']:
            raise ValueError('The calibrated public baseline changed')
    with current.open(newline='') as src:
        archive=np.asarray([[int(r['pred_vulnerability']),int(r['pred_MAT'])] for r in csv.DictReader(src)],dtype=bool)
    p27=json_at(R27/'public/policy.json')
    positive=p27['supported'][p27['selected']]
    p27m=R27/'joint/official/weights.joblib'
    assert checksum(p27m)==p27['weights_sha256']
    positive_head=joblib.load(p27m)[positive['weights_key']]
    config=json_at(R24/'official/tuned/policy.json')
    base=R24/'official/tuned/models.joblib'
    p25=json_at(R25/'public/policy.json')
    part=R25/'official/calibrator.joblib'
    assert checksum(base)==config['models_sha256'] and checksum(part)==p25['calibrator_sha256']
    package=joblib.load(base)['token']
    variant=config['candidates'][config['selected']]
    section=p25['support'][p25['selected']]
    section_head=joblib.load(part)[section['region']]
    opts={}
    for mode in picked:
        row=frozen['variants'][mode]
        if row['setting']['kind']=='threshold':
            expert=None
            if 'recover' in row['setting']:
                weights=HERE/'recover_calibrated/official/heads.joblib'
                if checksum(weights)!=row['source_sha256']:
                    raise ValueError('The recall ranker changed')
                expert=joblib.load(weights)['compact']
            opts[mode]=row['setting'],expert
        else:
            weights=HERE/'studies'/row['setting']['kind']/'official/readouts.joblib'
            assert checksum(weights)==row['source_sha256']
            opts[mode]=row['setting'],joblib.load(weights)[row['mode']]
    if recovery:
        opts['reference']=frozen['baseline_setting'],None
    ref, routing, heads=recipe()
    texts,label,_=load(PUBLIC)
    encoder=f.pipe('official',f.small.tokenizer())

    def forward(name,sample):
        p,aux,cost=f.infer_student(encoder,sample,f.pinned()[0],columns=22,return_all=True)
        at=time.perf_counter()
        values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
        est=f.estimate(heads,values)
        score,no,yes,ask=f.choose(values,est,routing,ref['gate'])
        known=np.zeros(len(p),dtype=bool)
        flag=np.zeros_like(known)
        if ask.any():
            outcome=scan([s for s,k in zip(sample,ask,strict=True) if k])
            known[ask]=outcome>=0
            flag[ask]=outcome==1
            score[known]=np.where(flag[known],yes[known],no[known])
        state=dict(p=score.astype(np.float32),w=est['w'],no=no,yes=yes,known=known,flag=flag)
        cut,data=opts.get(name,(None,None))
        det,scope,ranking,ops=rerank(sample,values,state,ref['threshold'],variant,package,section,section_head,cut,data)
        vp,active,_,bill=adjust_positive(sample,values,state,ref['threshold'],det,positive['online'],positive_head)
        if np.any(active & scope):
            raise ValueError('The candidate changed the positive-rule cohort')
        predictions=np.column_stack((vp,canonical_mat(sample)))
        elapsed=time.perf_counter()-at
        cost['wall_seconds']+=elapsed
        cost['forward_seconds']+=elapsed
        cost['analytic_gflops']+=(ops['base_ops']+ops['extra_ops']+bill['nnz'])/1e9
        bill2=cost|{'new_rows':int(scope.sum()),'new_seconds':ops['added']}
        if recovery and name=='reference':
            return predictions,np.zeros(len(p),dtype=bool),np.empty(0),bill2
        return predictions,scope,ranking,bill2

    modes=['reference',*picked]
    for name in modes:
        forward(name,texts[:2048])
    costs={n:[] for n in modes}
    result={}
    for attempt in range(5):
        for name in modes if attempt%2==0 else modes[::-1]:
            bits,scope,r,trial=forward(name,texts)
            if np.any(bits[~scope]!=archive[~scope]) or np.any(bits[:,1]!=archive[:,1]):
                raise ValueError('The full baseline or an unrelated output changed')
            if name in result and not np.array_equal(bits,result[name][0]):
                raise ValueError('Repeated predictions differ')
            costs[name].append(trial)
            result[name]=(bits,scope,r)
            print(attempt,name,trial,flush=True)
    report={}
    quality=f.metric(label,archive)
    base_y=label[:,0]
    paired={}
    for j,(name,(bits,scope,r)) in enumerate(result.items()):
        metric=f.metric(label,bits)
        med={k:median(x[k] for x in costs[name]) for k in costs[name][0]}
        local=lambda m,tm:f.competition_formula(m['mean']['f1'],tm['forward_seconds'],tm['analytic_gflops'])
        relative=archive[:,0].copy()
        if scope.any():
            budget=int((relative[scope]&(base_y[scope]==0)).sum())
            relative[scope],_=f.exact_budget(r,base_y[scope],budget)
        path=out/f'{j}_{name.replace(":","_")}.csv'
        score=np.zeros(len(bits))
        score[scope]=r
        with path.open('w',newline='') as fd:
            w=csv.writer(fd)
            w.writerow(('row','label_vulnerability','label_mat','pred_vulnerability','pred_MAT','scoped','rank','matched'))
            w.writerows((i,*y,*q,on,r,int(m)) for i,(y,q,on,r,m) in enumerate(
                zip(label,bits.astype(int),scope.astype(int),score,relative,strict=True)))
        report[name]=dict(score=local(metric,med),quality=metric,cost=med,trials=costs[name],csv=path.name,
                          frozen=f.comparison(base_y,archive[:,0],bits[:,0]),
                          equal_fp=f.comparison(base_y,archive[:,0],relative),predictions_sha256=checksum(path))
        paired[name]=[local(metric,c)-local(quality,b) for c,b in zip(costs[name],costs['reference'],strict=True)]
    write_json(out/'results.json',dict(results=report,paired=paired,policy_sha256=checksum(out/'policy.json'),
                                      public_sha256=checksum(PUBLIC),rows=len(label),scanner_sha256=checksum(PROGRAM)))
    print({k:v['score'] for k,v in report.items()},'paired',paired,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--select-only',action='store_true')
    p.add_argument('--threshold',action='store_true')
    p.add_argument('--recover',action='store_true')
    args=p.parse_args()
    if args.recover:
        from select_recover import select
        plan=select()
        run(plan,HERE/'recover_public',True)
    elif args.threshold:
        d=HERE/'threshold'
        source=json_at(OUT/'policy.json')
        receipt=json_at(OUT/'release_selection.json')
        name=receipt['selected']
        if name=='reference':
            raise ValueError('The matched threshold needs a live learned model')
        comparison=json_at(OUT/'audit.json')
        evidence=comparison['options'][name]
        cut=evidence['recalibration_cut']
        option=source['variants'][name]
        calibrated=dict(width=option['setting']['width'],mix=1.,cut=cut,kind='threshold')
        plan=dict(options=[name,'calibration_only'],family={'score':name},variants={
            name:option,'calibration_only':dict(setting=calibrated)},source_sha256=checksum(OUT/'audit.json'),
            calibration_tp=evidence['calibration_tp'])
        d.mkdir(exist_ok=True)
        write_json(d/'policy.json',plan)
        run(plan,d)
    else:
        plan=freeze()
        if not args.select_only:
            run(plan)
