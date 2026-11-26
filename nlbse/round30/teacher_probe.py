"""Fit a file-disjoint teacher and measure its complementarity to the compact tool."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from embed import HERE, ROOT, CACHE, f, checksum, json_at, write_json
from compact_oof import signed_source
from disagreement import clip_logit, trained_logistic

DEST=HERE/'teachers'
PARTS=('test_a','test_b','official')
SEED=96


def signed_inputs(rotation):
    scores,signature=signed_source(rotation)
    area=ROOT/'round26/compact'/rotation
    file=area/'scores.npz'
    report=json_at(area/'manifest.json')
    if checksum(file)!=report['sha256'] or signature!=report['sparse_oof_sha256']:
        raise ValueError('Independent compact scores changed')
    with np.load(file) as old:
        if not all(np.array_equal(scores[k],old[k]) for k in ('ids','y','fold')):
            raise ValueError('Compact and sparse assignments disagree')
        active=scores['weak']==0
        p=old['values'][active,2]
    return {k:w[active] for k,w in scores.items()},p,dict(sparse=signature,compact=checksum(file))


def store():
    cfg=json_at(CACHE/'manifest.json')
    source=CACHE/'features.npy'
    rows=CACHE/'rows.npz'
    if checksum(source)!=cfg['data_sha256'] or checksum(rows)!=cfg['rows_sha256']:
        raise ValueError('Frozen embeddings changed')
    with np.load(rows) as z:
        ix,y=z['ids'],z['y']
    x=np.load(source,mmap_mode='r')
    return x,ix,y,cfg


def probe(x,y,keep):
    torch.manual_seed(SEED)
    start=time.perf_counter()
    model=nn.Sequential(nn.LayerNorm(x.shape[1]),nn.Linear(x.shape[1],128),nn.GELU(),
                        nn.Dropout(.18),nn.Linear(128,1)).cuda()
    amount=min(4.,float(((len(keep)-y[keep].sum())/y[keep].sum()).sqrt()))
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.025)
    batch_size=512
    optim=torch.optim.lr_scheduler.CosineAnnealingLR(opt,3*((len(keep)+batch_size-1)//batch_size),eta_min=.00012)
    values=torch.as_tensor(keep,dtype=torch.long,device='cuda')
    for e in range(3):
        model.train()
        positions=values[torch.randperm(len(values),device='cuda')]
        for batch in positions.split(batch_size):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                z=model(x[batch])[:,0].float()
                loss=F.binary_cross_entropy_with_logits(z,y[batch],pos_weight=y.new_tensor(amount))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            optim.step()
    return model.eval(),amount,time.perf_counter()-start


@torch.inference_mode()
def logits(model,pool,amount):
    with torch.autocast('cuda',dtype=torch.float16):
        output=torch.cat([model(chunk)[:,0].float() for chunk in pool.split(2048)])
    return output.cpu().numpy()-np.log(amount)


def calibrated_oof(values,y,fold):
    p=np.empty_like(values,dtype=np.float64)
    for k in np.unique(fold):
        held=k==fold
        for j,column in enumerate(values.T):
            model=trained_logistic(column[~held,None],y[~held],c=.8)
            p[held,j]=model.predict_proba(column[held,None])[:,1]
    return p


def census(y,score,fold,old15):
    p=calibrated_oof(score,y,fold)
    plain=p[:,:2].mean(1)
    teacher=p[:,2]
    mixture=.65*plain+.35*teacher
    cutoff=.18
    initial=plain>=cutoff
    target=int((initial&~y).sum())
    bits,_=f.exact_budget(mixture,y,target)
    old=plain>=cutoff
    tp=np.flatnonzero(y&~old)
    fp=np.flatnonzero(~y&old)
    misses=np.flatnonzero(y&(p[:,:2].max(1)<cutoff))
    alarms=np.flatnonzero(~y&(p[:,:2].min(1)>=cutoff))
    def errors(part, prediction, want):
        return int(np.sum(prediction[part]==want))
    available=old15['match']
    result=dict(threshold=cutoff,positives=int(y.sum()),baseline_comparison=f.comparison(y,initial,bits),
                baseline_missed=len(tp),baseline_false_alarm=len(fp),both_low_misses=len(misses),both_high_alarms=len(alarms),
                teacher_detects_both_low=errors(misses,teacher>=cutoff,True),
                teacher_rejects_both_high=errors(alarms,teacher>=cutoff,False),
                ap={k:float(average_precision_score(y,c)) for k,c in zip(
                    ('compact','sparse','frozen_unix_probe','round15_unix','round15_compact'),
                    (*p.T, old15['unix'],old15['compact']),strict=True) if not k.startswith('round15')},
                earlier_overlap=int(available.sum()),earlier_overlap_ap={
                    k:float(average_precision_score(y[available],old15[k][available])) for k in ('unix','compact')})
    return np.column_stack((p,plain,mixture)),result


def earlier_scores(ids,y):
    dirname=ROOT/'round15/mining'
    old=dirname/'scores.npz'
    receipt=json_at(dirname/'manifest.json')
    if checksum(old)!=receipt['sha256'] or checksum(ROOT/'round15/folds.npz')!=receipt['partition_sha256']:
        raise ValueError('Earlier fold evidence changed')
    with np.load(old) as z:
        _, here, there=np.intersect1d(ids,z['ids'],return_indices=True)
        assert np.array_equal(y[here],z['y'][there,0])
        r={'match':np.isin(ids,z['ids'])}
        for name in ('unix','compact'):
            signal=np.full(len(y),np.nan)
            signal[here]=z[name][there,0]
            r[name]=signal
    return r


def fit(rotation):
    ticket,p,sources=signed_inputs(rotation)
    raw,ids,y,cfg=store()
    selected=ticket['ids']
    index=np.searchsorted(ids,selected)
    if not np.array_equal(ids[index],selected) or not np.array_equal(y[index],ticket['y']):
        raise ValueError('UniXcoder features and signed compact labels do not align')
    x=torch.as_tensor(raw[index],dtype=torch.float16,device='cuda')
    gold=torch.as_tensor(ticket['y'],dtype=torch.float32,device='cuda')
    fold=ticket['fold']
    guess=np.empty(len(index),dtype=np.float32)
    out=DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    log={}
    for k in np.unique(fold):
        use=(ticket['dup']&(1<<int(k)))==0
        target=fold==k
        if np.any(use&target):
            raise ValueError('Fold leaks known normalized copies')
        model,amount,clock=probe(x,gold,np.flatnonzero(use))
        guess[target]=logits(model,x[np.flatnonzero(target)],amount)
        path=out/f'probe_{k}.pt'
        torch.save(model.cpu().state_dict(),path)
        log[int(k)]=dict(fit=int(use.sum()),held=int(target.sum()),seconds=clock,sha256=checksum(path))
        print(rotation,int(k),log[int(k)],flush=True)
    raw=np.column_stack((clip_logit(p),ticket['logits'][:,0],guess))
    old=earlier_scores(selected,ticket['y'])
    calibrated,report=census(ticket['y'].astype(bool),raw,fold,old)
    path=out/'oof.npz'
    np.savez_compressed(path,ids=selected,y=ticket['y'],fold=fold,raw=raw,calibrated=calibrated,
                        earlier_unix=old['unix'],earlier_compact=old['compact'])
    results=dict(scope=report,sources=sources|{'embeddings':cfg['data_sha256']},folds=log,
                 predictions_sha256=checksum(path),code_sha256=checksum(Path(__file__)),rows=len(selected))
    write_json(out/'fit.json',results)
    print(rotation,report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    fit(p.parse_args().rotation)
