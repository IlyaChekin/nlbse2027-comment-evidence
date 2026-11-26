"""Check an independent, frozen UniXcoder view of functions flagged by the scanner."""
import argparse
from itertools import product

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from context32 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, signed_source, frames
from collect_dense import DEST
from eval32 import cut_for, compare, per_project

WIDTHS=(.4,.8,1.8,3.8,None)
STRENGTHS=(.35,.65,1.)
OUT=HERE/'dense_evaluation'


@torch.inference_mode()
def train_projection(x, y, sample_weight):
    centered = x.mean(0)
    spread = x.std(0).clamp_min(.08)
    u = (x-centered)/spread
    mass = sample_weight.sum()
    # Shrink the embedding readout, using the weighted loss in the compact decision region.
    mu = sample_weight @ u / mass
    rate = sample_weight @ y / mass
    v = (u-mu)*sample_weight.sqrt().unsqueeze(1)
    cov = v.T @ v
    cov.diagonal().add_(mass*.085)
    rhs = (u-mu).T @ ((y-rate)*sample_weight)
    coef = torch.linalg.solve(cov,rhs)/spread
    intercept = rate-(centered+mu*spread)@coef
    return coef,intercept


def logistic(x,y,weight=None):
    model = LogisticRegression(C=.5,solver='liblinear',random_state=42,max_iter=200)
    return model.fit(x,y,sample_weight=weight)


def compact_features(values):
    p = np.clip(values,1e-5,1.-1e-5).astype(np.float64)
    logits = np.log(p/(1.-p))
    return logits[:,[0,3,1,2]]


def learn(rotation, matrix, row_map, threshold):
    source, sha = signed_source(rotation)
    signed = ROOT/'round26/compact'/rotation
    config = json_at(signed/'manifest.json')
    path = signed/'scores.npz'
    if config['sparse_oof_sha256']!=sha or checksum(path)!=config['sha256']:
        raise ValueError('Compact source changed')
    mask = source['weak']==1
    ids, y, folds, duplicates = (source[k][mask] for k in ('ids','y','fold','dup'))
    x = torch.as_tensor(np.asarray(matrix[row_map[ids]],dtype=np.float32),device='cuda')
    g = torch.as_tensor(y,dtype=torch.float32,device='cuda')
    with np.load(path) as scores:
        assert all(np.array_equal(scores[k],source[k]) for k in ('ids','y','fold'))
        values = compact_features(scores['values'][mask])
    rank = np.empty(len(ids),np.float64)
    raw = np.empty(len(ids),np.float64)
    for k in np.unique(folds):
        hold = folds==k
        keep = (duplicates & (1<<int(k)))==0
        assert not (hold & keep).any()
        local = logistic(values[keep],y[keep])
        margin = local.decision_function(values)
        rank[hold] = margin[hold]
        w = .30+2.*np.exp(-abs(margin-np.log(threshold/(1.-threshold))))
        index = torch.as_tensor(keep,device='cuda')
        coef, midpoint = train_projection(x[index],g[index],torch.as_tensor(w[keep],device='cuda',dtype=torch.float32))
        raw[hold] = (x[hold]@coef+midpoint).cpu().numpy()
    weight = .30+2.*np.exp(-abs(rank-np.log(threshold/(1.-threshold))))
    coef,midpoint = train_projection(x,g,torch.as_tensor(weight,device='cuda',dtype=torch.float32))
    residual = logistic(np.column_stack((values,raw)),y,weight)
    calibrator = logistic(raw[:,None],y,weight)
    core = logistic(values,y,weight)
    return dict(coef=coef.cpu().numpy(),offset=float(midpoint),models={'dense_only':calibrator,'dense_joint':residual,
               'compact':core},source=sha,compact=config['sha256'],training=len(y))


def predict(ctx, row_map, features, models):
    mask = ctx.state['known'] & ctx.state['flag']
    ids = ctx.frame['ids'][mask]
    pos = row_map[ids]
    if (pos<0).any():
        raise ValueError('A signed scanner-positive function is absent from the embedding cache')
    v = compact_features(ctx.frame['reference'][mask])
    x = np.asarray(features[pos],dtype=np.float32)
    score = x@models['coef']+models['offset']
    predictors = models['models']
    return mask,dict(dense_only=predictors['dense_only'].decision_function(score[:,None]),
                     dense_joint=predictors['dense_joint'].decision_function(np.column_stack((v,score))),
                     compact=predictors['compact'].decision_function(v))


def evaluate(rotation):
    dev,test,cal,review,(p,q),threshold = frames(rotation)
    old,base = p>0.,q>0.
    y0,y1 = dev.frame['y'].astype(bool),test.frame['y'].astype(bool)
    source = DEST/'features.npy'
    rows = DEST/'rows.npz'
    meta = json_at(DEST/'receipt.json')
    if checksum(source)!=meta['archive_sha256'] or checksum(rows)!=meta['rows_sha256']:
        raise ValueError('The signed UniXcoder features changed')
    feat = np.load(source,mmap_mode='r')
    mapping = np.full(len(f.inputs()[0]),-1,dtype=np.int32)
    with np.load(rows) as z:
        mapping[z['ids']] = np.arange(len(z['ids']))
    train = learn(rotation,feat,mapping,threshold)
    scope,a = predict(dev,mapping,feat,train)
    active,b = (scope,a) if test is dev else predict(test,mapping,feat,train)
    records, snapshots = {}, dict(ids=test.frame['ids'][review],truth=y1[review],baseline=base[review],
                                  available=active[review],original_rank=q[review])
    for width in WIDTHS:
        width_key = 'all' if width is None else f'{width:g}'
        u,v = scope.copy(),active.copy()
        if width is not None:
            u &= abs(p)<width
            v &= abs(q)<width
        if not (u & cal).any():
            continue
        for name,mix in [('threshold',0.),*product(a,STRENGTHS)]:
            key = f'{name}_{width_key}_{mix:g}'
            d,t = p.copy(),q.copy()
            if mix:
                d[u] = (1.-mix)*d[u]+mix*a[name][u[scope]]
                t[v] = (1.-mix)*t[v]+mix*b[name][v[active]]
            row = dict(setting=dict(style=name,width=width,strength=mix,publishable=True),
                       calls=dict(cal=int((u & cal).sum()),held=int((v & review).sum())))
            if test is not dev:
                row['matched'], result = per_project(y1,test.frame['project'],base,v,t[v],('fp','tp'))
                for protocol, forecast in result.items():
                    snapshots[key+'_matched_'+protocol] = forecast
            for protocol in ('tp','fp','fixed'):
                cutoff = cut_for(y0[u & cal],old[u & cal],d[u & cal],protocol)
                c,_ = compare(y0[cal],old[cal],u[cal],d[u & cal],cutoff)
                h,z = compare(y1[review],base[review],v[review],t[v & review],cutoff)
                row[protocol] = dict(cut=cutoff,cal=c,frozen=h)
                snapshots[key+'_'+protocol] = z
            records[key] = row
            snapshots[key+'_rank'] = t[review]
    folder = OUT/rotation
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'predictions.npz'
    np.savez_compressed(path,**snapshots)
    weights = folder/'rankers.joblib'
    joblib.dump(train,weights,compress=3)
    write_json(folder/'results.json',dict(rotation=rotation,options=records,feature_source=meta['archive_sha256'],
               compact_oof_sha256=train['compact'],weights_sha256=checksum(weights),fit_rows=train['training'],
               predictions_sha256=checksum(path),code_sha256=checksum(HERE/'dense_rank.py')))
    print(rotation,'fit',train['training'],'hold',int((active & review).sum()),'options',len(records),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    evaluate(parser.parse_args().rotation)
