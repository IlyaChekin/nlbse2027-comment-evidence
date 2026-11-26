"""Recover precision-first rejections using the previous score or an independent teacher."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import sys

import joblib
import numpy as np

from pairs import HERE, ROOT, PARTS, f, checksum, json_at, write_json, clip_logit
from paired_context import context, match, strict_cut, R27
from sklearn.linear_model import LogisticRegression
from sparse_model import vectorize

sys.path.insert(0,str(R27))
from teacher import archive, MODEL

DEST=HERE/'recover'
RUNTIME=HERE/'public'
MODES=('compact','teacher')
WIDTHS=(.3,.75,1.5)


def updated(d,old,margin,option,package):
    eligible=d.active & (abs(margin)<option['width'])
    texts=[t for t,k in zip(d.texts,eligible,strict=True) if k]
    x=vectorize(texts,'token')
    features=np.column_stack((margin[eligible],*[h.decision_function(x) for h in package['rankers']]))
    score=package['readout'].decision_function(features)
    bits=old.copy()
    adjusted=margin.copy()
    w=option['mix']
    adjusted[eligible]=(1.-w)*margin[eligible]+w*score-option['cut']
    bits[eligible]=adjusted[eligible]>0
    return bits,adjusted


def run(part, calibrated):
    source=json_at(RUNTIME/'policy.json')
    release=json_at(RUNTIME/'release_selection.json')['selected']
    if release=='reference':
        raise ValueError('The pilot requires the frozen precision-first release')
    method,variant,kind=release.split(':')
    d,t,fit,cal,review,old,other,dev_before,test_before=context(part)
    branch=HERE/'studies'/method/part
    study=json_at(branch/'results.json')
    weight=branch/'readouts.joblib'
    if checksum(weight)!=study['weights_sha256']:
        raise ValueError('The precision-first ranker changed')
    row=study['options'][variant]
    cfg=dict(width=row['width'],mix=row['mix'],cut=row['cuts'][kind]['cut'])
    package=joblib.load(weight)[row['mode']]
    if calibrated:
        select=d.active&cal&(abs(dev_before)<cfg['width'])
        gold=d.frame['y'][select]
        n=int((gold&old[select]).sum())+row['cuts'][kind]['cal']['delta_tp']
        if n>0:
            ordered=np.sort(dev_before[select][gold>0])[::-1]
            cut=float(np.nextafter(ordered[n-1],-np.inf))
        else:
            cut=float(max(dev_before[select]))
        if not abs(cut)<=cfg['width']:
            raise ValueError('The unchanged decisions outside the margin band cannot be inferred from the new cutoff')
        def update(c,b,p):
            changed=b.copy()
            m=p.copy()
            m[c.active]-=cut
            changed[c.active]=m[c.active]>0.
            return changed,m
        ref,ra=update(d,old,dev_before)
        op,rb=(ref,ra) if t is d else update(t,other,test_before)
        assert int((ref[select]&gold).sum())==n
        if part=='official':
            official=json_at(HERE/'threshold/policy.json')
            assert cut==official['variants']['calibration_only']['setting']['cut']
    else:
        ref,ra=updated(d,old,dev_before,cfg,package)
        op,rb=(ref,ra) if t is d else updated(t,other,test_before,cfg,package)
        archive_file=branch/'predictions.npz'
        if checksum(archive_file)!=study['predictions_sha256']:
            raise ValueError('Frozen precision-first decisions changed')
        with np.load(archive_file) as z:
            scope=review if review is not None else np.ones(len(op),dtype=bool)
            if not np.array_equal(z[f'{variant}_{kind}'],op[scope]):
                raise ValueError('Live precision-first decisions did not reproduce in the study')
    held=review if review is not None else np.ones(len(op),dtype=bool)
    a,b,receipts=archive(part,d,t)
    z=lambda c,p,q: np.column_stack((p,q,c.inputs[:,:3]))
    c,d2=z(d,dev_before,ra), z(t,test_before,rb)
    da=np.column_stack((c,clip_logit(a)))
    tb=np.column_stack((d2,clip_logit(b)))
    yd,yt=d.frame['y'],t.frame['y']
    penalty=.4+2.0*(old!=ref)+np.exp(-abs(ra))
    destination=(HERE/'recover_calibrated' if calibrated else DEST)/part
    destination.mkdir(parents=True,exist_ok=True)
    evidence={}
    models={}
    predictions=dict(ids=t.frame['ids'][held],label=yt[held],old=op[held])
    if review is not None:
        from students import load_selected,scores
        remodel,tok,_=load_selected(MODEL)
    for model in MODES:
        x,y=(da,tb) if model=='teacher' else (c,d2)
        use=fit&d.active
        f1=LogisticRegression(C=.6,solver='liblinear',max_iter=200,random_state=42).fit(
                x[use],yd[use],sample_weight=penalty[use])
        qa,qb=f1.decision_function(x),f1.decision_function(y)
        models[model]=f1
        for width in WIDTHS:
            inside=d.active&(abs(ra)<width)
            query=t.active&(abs(rb)<width)
            if (inside&cal).sum()==0:
                continue
            cost=dict(seconds=0.,gflops=0.)
            if model=='teacher' and review is not None:
                subset=[s for s,k in zip(d.texts,inside&cal,strict=True) if k]
                values,time0=scores(remodel,tok,subset,MODEL)
                if not np.allclose(values[:,0],a[inside&cal],atol=.025,rtol=0):
                    raise ValueError('Live teacher scores differ from its cached scores')
                runs=[scores(remodel,tok,subset,MODEL)[1] for _ in range(3)]
                cost=dict(seconds=median(x['forward_seconds'] for x in runs),
                          gflops=median(x['analytic_gflops'] for x in runs))
            for mix,direction in product((.35,.7),('exchange','recover')):
                xd=(1.-mix)*ra[inside]+mix*qa[inside]
                xt=(1.-mix)*rb[query]+mix*qb[query]
                if direction=='recover':
                    xd=np.where(ref[inside],ra[inside],np.maximum(ra[inside],xd))
                    xt=np.where(op[query],rb[query],np.maximum(rb[query],xt))
                key=f'{model}_{width:g}_{mix:g}_{direction}'
                row=dict(cost=cost,cal_calls=int((inside&cal).sum()),query_calls=int((query&held).sum()),cuts={})
                if review is None:
                    votes,group=match(yt,t.frame['project'],op,query,xt)
                    row.update(matched=f.comparison(yt,op,votes),projects=group)
                    predictions[key+'_matched']=votes
                for ch in ('fp','tp','fixed'):
                    cut=strict_cut(yd[inside&cal],ref[inside&cal],xd[cal[inside]],ch)
                    before,after=ref.copy(),op.copy()
                    before[inside],after[query]=xd>cut,xt>cut
                    local=f.comparison(yd[cal],ref[cal],before[cal])
                    row['cuts'][ch]=dict(cut=cut,cal=local,frozen=f.comparison(yt[held],op[held],after[held]))
                    if review is not None:
                        tax=.2*f.PUBLIC_N/cal.sum()*(cost['seconds']/1000+cost['gflops']/3e6)
                        row['cuts'][ch]['proxy']=.3*(local['f1_after']-local['f1_before'])-tax
                    predictions[key+'_'+ch]=after[held]
                evidence[key]=row
    out=destination/'predictions.npz'
    np.savez_compressed(out,**predictions)
    weights=destination/'heads.joblib'
    joblib.dump(models,weights,compress=3)
    write_json(destination/'results.json',dict(options=evidence,teacher=receipts,threshold_baseline='calibration_only' if calibrated else release,
                    predictions_sha256=checksum(out),weights_sha256=checksum(weights)))
    print(part,len(evidence),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    p.add_argument('--calibrated',action='store_true')
    a=p.parse_args()
    run(a.part,a.calibrated)
