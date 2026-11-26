"""Fit an additive correction to within-file order using frozen representations."""
import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch

from fit46 import (HERE,ROOT,PARTS,BANK,PHASES,f,checksum,json_at,load,parts,write_json,negative,crop,
                    fingerprint,pairs,prepare,point_matrix,join,compact,inspect)

DEST = HERE/'paired'


def fit(x,rank,allow,a,b):
    keep = allow[a]&allow[b]
    p,n = a[keep],b[keep]
    selected = x[allow]
    mean = selected.mean(0)
    spread = selected.std(0).clamp_min(.045)
    if not p.size:
        raise ValueError('Every fold needs at least one eligible pair')
    pos = ((x[p]-mean)/spread).clamp(-8.,8.)
    neg = ((x[n]-mean)/spread).clamp(-8.,8.)
    difference = pos-neg
    pair_margin = rank[p]-rank[n]
    delta = (.9-pair_margin).clamp(-2.8,2.8)
    _,inverse,counts=np.unique(p,return_inverse=True,return_counts=True)
    weight = torch.as_tensor(1./counts[inverse],dtype=torch.float32,device='cuda')
    weight *= .6+torch.exp(-.23*(rank[p].abs()+rank[n].abs()))
    weight /= weight.sum()
    # Penalize movement on the broad point distribution and add a ridge prior.
    reserve = selected[::max(1,len(selected)//12000)]
    reserve = ((reserve-mean)/spread).clamp(-8.,8.)
    square = difference.T@(difference*weight[:,None])+.35*(reserve.T@reserve/len(reserve))
    square += .16*torch.eye(x.shape[1],device='cuda')
    slope = torch.linalg.solve(square,difference.T@(weight*delta))
    return dict(mean=mean.cpu().numpy(),scale=spread.cpu().numpy(),coef=slope.cpu().numpy(),
                selected=int(np.count_nonzero(allow)),pairs=len(p))


def predict(x,rank,head):
    return (rank+((x-head['mean'])/head['scale']).clip(-8.,8.)@head['coef']).astype(np.float32)


def main(rotation):
    start = perf_counter()
    data,t,stamp = load(rotation)
    phases = [crop(z,negative(z)) for z in (data['oof'],*parts(data))]
    train = phases[0]
    held = data.get('test',data['dev'])
    corpus = f.inputs()
    outer = np.unique(np.r_[data['dev']['ids'],held['ids']])
    assert not np.isin(train['ids'],outer).any()
    if 'test' in data:
        assert not set(corpus[2][train['ids']])&set(corpus[2][held['ids']])
    blocked = {fingerprint(corpus[0][i]) for i in outer}
    allowed = np.fromiter((fingerprint(corpus[0][i]) not in blocked for i in train['ids']),bool)
    pair_groups,proof = pairs(rotation,train,allowed,corpus)
    p,n = np.unique(np.concatenate([np.column_stack(pair) for pair in pair_groups.values()]),axis=0).T
    index,encodings,certificates = prepare(rotation,stamp)
    loc = [np.searchsorted(index,r['ids']) for r in phases]
    assert all(np.array_equal(index[ix],r['ids']) for ix,r in zip(loc,phases,strict=True))
    numerics = point_matrix(phases,t)
    latent,compact_proof = compact(rotation,phases,data,stamp)
    rank = torch.as_tensor(train['signed'].clip(-12.,12.),device='cuda',dtype=torch.float32)
    features,estimates,counts = {},{},{}
    tags = ('numeric','compact',*BANK)
    for kind in tags:
        if kind=='numeric':
            x = numerics
        elif kind=='compact':
            x = join(numerics,latent)
        else:
            x = join(numerics,[encodings[kind][v] for v in loc])
        backed = torch.as_tensor(x[0],device='cuda')
        outside = np.full(len(train['ids']),np.nan,np.float32)
        counts[kind] = {}
        for fold in np.unique(train['fold']):
            found = train['fold']==fold
            used = allowed & ((train['duplicates']&(1<<int(fold)))==0)
            assert not (found&used).any()
            estimate = fit(backed,rank,used,p,n)
            outside[found] = predict(x[0][found],train['signed'][found],estimate)
            counts[kind][int(fold)] = estimate['pairs']
        assert np.isfinite(outside).all()
        h = fit(backed,rank,allowed,p,n)
        estimates[kind] = outside,*(predict(a0,phase['signed'],h) for a0,phase in zip(x[1:],phases[1:],strict=True))
        features[kind] = h
        del x,backed
        print(rotation,kind,round(perf_counter()-start,2),flush=True)
    for cap in (96,384):
        estimates[f'joint{cap}'] = tuple(.5*(p+q) for p,q in zip(estimates[f'qwen{cap}'],estimates[f'unix{cap}'],strict=True))
    summary = inspect(estimates,train,pair_groups,corpus)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    packed = {name+'_ids':z['ids'] for name,z in zip(PHASES,phases,strict=True)}
    for kind,scores in estimates.items():
        packed.update({name+'_'+kind:y for name,y in zip(PHASES,scores,strict=True)})
    np.savez_compressed(out/'scores.npz',**packed,fold=train['fold'],permitted=allowed)
    joblib.dump(features,out/'weights.joblib',compress=3)
    receipt = dict(matrix_sha256=stamp,source_features=certificates,compact_sha256=compact_proof,
                   fit_rows=len(allowed),copies_excluded=int((~allowed).sum()),pair_sha256=proof,
                   diagnostics=summary,fold_pair_counts=counts,seconds=perf_counter()-start,
                   model_sha256=checksum(out/'weights.joblib'),predictions_sha256=checksum(out/'scores.npz'),
                   code_sha256=checksum(Path(__file__)))
    write_json(out/'receipt.json',receipt)
    print(rotation,'paired',
          {k:{n:round(v['pair_win'],5) for n,v in d['ranks'].items()} for k,d in summary.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
