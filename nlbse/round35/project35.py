"""Transfer matched teacher differences into a single cheap pooled linear projection."""
import argparse

import joblib
import numpy as np
import torch

from teacher35 import HERE, PARTS, DEST as TEACHERS, f, checksum, json_at, write_json
from eval35 import load_all, signatures, crop, design, comparisons, point_classifier
from compact34 import cache

DEST=HERE/'projection'
RIDGE=.36
PAIR_WEIGHT=.65
MODES=('prefix_point','prefix_pair','body_point','body_pair','labels')


def project(x,y,w,dx,dy,pair_w,penalty):
    total=w.sum()
    offset=(w@x)/total
    center=(w@y)/total
    state=x-offset
    cov=(state.T@(w[:,None]*state))/total
    cov.diagonal().add_(RIDGE)
    rhs=(state.T@(w[:,None]*(y-center)))/total
    if penalty:
        adjust=pair_w*(penalty/pair_w.sum())
        cov+=dx.T@(adjust[:,None]*dx)
        rhs+=dx.T@(adjust[:,None]*dy)
    weights=torch.linalg.solve(cov,rhs)
    return weights,center-offset@weights


def forecast_features(data,t,score):
    return design(data,t,score)


def evaluate(rotation):
    data,ref,old,train=load_all(rotation)
    d=data['dev']
    ex=data.get('test',d)
    fit,cal,held=crop(d,d['fit']),crop(d,d['cal']),crop(ex,ex['held'])
    oof=data['oof']
    flat=f.inputs()[0]
    model=f.pipe(rotation,f.small.tokenizer())
    views=(oof,fit,cal,held)
    matrix=[]
    for q in views:
        text=[flat[i] for i in q['ids'][q['support']]]
        features=cache(model.encoder,model.tok,text)
        matrix.append(features[:,:2*model.encoder.width].float().contiguous())
    ref_rank=ref['cutoff']
    valid=oof['support']
    ix=np.flatnonzero(valid)
    a,b=signatures(rotation,oof)
    u,v=np.searchsorted(ix,a),np.searchsorted(ix,b)
    assert np.array_equal(ix[u],a) and np.array_equal(ix[v],b)
    _,owner,weight=np.unique(u,return_inverse=True,return_counts=True)
    pair_w=torch.as_tensor(1./weight[owner],device='cuda',dtype=torch.float32)
    x=matrix[0]
    dx=x[u]-x[v]
    cur=torch.as_tensor(oof['queried'][valid],device='cuda',dtype=torch.float32)
    conf=.3+1.50*(-.65*cur.abs()).exp()
    preds={}
    models={}
    stats={}
    out=DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    forecast=dict(oof_ids=oof['ids'][valid],cal_ids=cal['ids'][cal['support']],held_ids=held['ids'][held['support']])
    for mode in MODES:
        if mode=='labels':
            numbers=np.column_stack((2.*oof['y'][valid]-1.,oof['prefix_pair'][valid]-oof['prefix_initial'][valid]))
        else:
            stem=mode.split('_')[0]
            value=oof[mode][valid]
            numbers=np.column_stack((value,value-oof[stem+'_initial'][valid]))
        target=torch.as_tensor(numbers,device='cuda',dtype=torch.float32).clamp(-8.,8.)
        dy=target[u]-target[v]
        for loss,penalty in [('point',0.),('pair',PAIR_WEIGHT)]:
            name=f'{mode}_{loss}'
            beta,offset=project(x,target,conf,dx,dy,pair_w,penalty)
            weight_file=out/f'{name}.pt'
            torch.save(dict(weight=beta.T.contiguous().cpu(),bias=offset.cpu()),weight_file)
            study=[]
            for label,ds,features in zip(('oof','fit','cal','held'),views,matrix,strict=True):
                result=(features@beta+offset).cpu().numpy()
                forecast[label+'_'+name]=result
                full=np.zeros((len(ds['ids']),2),np.float32)
                full[ds['support']]=result
                study.append(forecast_features(ds,ref_rank,full))
            stats[name]=dict(sha256=checksum(weight_file),mode=mode,loss=loss)
            for i,(family,ds) in enumerate(zip(('oof','fit'),views[:2],strict=True)):
                r=point_classifier(study[i],ds['y'],ds['latest'],ds['support'])
                scores=[r.decision_function(q) for q in study[2:]]
                preds[family,name]=scores
                models[family,name]=r
    table,results=comparisons(rotation,cal,held,preds)
    embeddings=out/'scores.npz'
    np.savez_compressed(embeddings,**forecast)
    prediction=out/'predictions.npz'
    np.savez_compressed(prediction,**results)
    rec=out/'readouts.joblib'
    joblib.dump(models,rec,compress=3)
    source=TEACHERS/rotation/'receipt.json'
    assert checksum(source.with_name('scores.npz'))==train['predictions_sha256']
    write_json(out/'results.json',dict(rotation=rotation,options=table,heads=stats,baseline_sha256=old,
              teacher_sha256=checksum(source),ridge=RIDGE,pair_weight=PAIR_WEIGHT,
              code_sha256=checksum(HERE/'project35.py'),predictions_sha256=checksum(prediction),
              weights_sha256=checksum(rec),projection_scores_sha256=checksum(embeddings)))
    print(rotation,'projection',len(table),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    evaluate(p.parse_args().rotation)
