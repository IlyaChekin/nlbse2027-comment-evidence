"""Cross-fit strong pooled teachers and train matched compact distillation controls."""
import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch
from torch import nn

from source41 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, load, supported, crop, parts
from collect41 import DEST as GRAPH
from compact41 import DEST as LATENT
from data40 import features
from project_split import fingerprint

DEST=HERE/'heads'
BASE=.70
EPOCHS=12
BATCH=4096
SEED=4100
T=1.7
TRANSFER=.52
DENSE=('graph','unix')


def columns(case,t,extra=None):
    a,b,c,_,s=features(case,t)
    original=np.column_stack((a,b,c)).astype(np.float32)
    return original if extra is None else np.column_stack((original,extra)).astype(np.float32)


def dense_sources(rotation,ticket):
    graph=GRAPH
    manifest=json_at(graph/'receipt.json')
    assert manifest['phases'][rotation]==ticket
    assert manifest['code_sha256']==checksum(HERE/'collect41.py')
    assert manifest['features_sha256']==checksum(graph/'features.npy')
    assert manifest['rows_sha256']==checksum(graph/'rows.npz')
    with np.load(graph/'rows.npz') as z:
        ids=z['ids']
    bands={}
    for home,stamp,field in [('round30/embeddings','manifest.json','data_sha256'),
                             ('round32/dense','receipt.json','archive_sha256')]:
        folder=ROOT/home
        ref=json_at(folder/stamp)
        file=folder/'features.npy'
        roster=folder/'rows.npz'
        assert checksum(file)==ref[field]
        assert checksum(roster)==ref['rows_sha256']
        with np.load(roster) as z:
            selected=z['ids']
        bands[home]=(selected,np.load(file,mmap_mode='r'),checksum(folder/stamp))
    return (ids,np.load(graph/'features.npy',mmap_mode='r')),bands


def feature_rows(query,store):
    hint,source=store
    selected=np.searchsorted(hint,query)
    assert np.array_equal(hint[selected],query)
    return source[selected]


def unix_rows(ids,parts):
    candidates=[(row,v) for row,v,_ in parts.values()]
    width=candidates[0][1].shape[1]
    arr=np.empty((len(ids),width),np.float16)
    ordinary=np.zeros(len(ids),bool)
    for row,v in candidates:
        common,i,j=np.intersect1d(ids,row,return_indices=True)
        assert not ordinary[i].any()
        ordinary[i]=True
        arr[i]=v[j]
    assert ordinary.all()
    return arr


def set_fit(x,y,rank,train,teacher=None,salt=0):
    # Compare identical feature layouts, schedules and label weights.
    torch.manual_seed(SEED+int(salt))
    g=torch.Generator(device='cuda').manual_seed(SEED+int(salt))
    used=torch.as_tensor(np.flatnonzero(train),device='cuda')
    mean=x[used].mean(0)
    spread=x[used].std(0).clamp_min(.045)
    residual=nn.Linear(x.shape[1],1).cuda()
    opt=torch.optim.AdamW(residual.parameters(),lr=.004,weight_decay=.045,fused=True)
    y0=y[used]
    baseline=rank[used]
    sample=.45+1.7*(-.26*baseline.abs()).exp()
    data=((x[used]-mean)/spread).clamp(-8.,8.)
    yself=None if teacher is None else (teacher[used]/T).sigmoid()
    stats=[]
    for epoch in range(EPOCHS):
        opt.param_groups[0]['lr']=.0045*(1.-.77*epoch/max(EPOCHS-1,1))
        epoch_cost=0.
        batches=torch.randperm(len(used),generator=g,device='cuda').split(BATCH)
        for idx in batches:
            opt.zero_grad(set_to_none=True)
            q=BASE*baseline[idx]+residual(data[idx])[:,0]
            a=nn.functional.binary_cross_entropy_with_logits(q,y0[idx],reduction='none')
            if yself is not None:
                b=nn.functional.binary_cross_entropy_with_logits(q/T,yself[idx],reduction='none')
                a=(1.-TRANSFER)*a+TRANSFER*T*T*b
            loss=(sample[idx]*a).mean()+.0012*residual.weight.square().mean()
            loss.backward()
            opt.step()
            epoch_cost+=loss.detach()
        stats.append(float(epoch_cost/len(batches)))
    return dict(mean=mean.cpu().numpy(),scale=spread.cpu().numpy(),
                coef=residual.weight[0].detach().cpu().numpy(),intercept=float(residual.bias.detach().item()),
                loss=stats)


def predict(x,margin,head):
    design=((x-head['mean'])/head['scale']).clip(-8.,8.)
    return (BASE*margin+design@head['coef']+head['intercept']).astype(np.float32)


def fit_scopes(kind,mats,ranking,cohorts,eligible,soft=None,crossfit=False):
    x=torch.as_tensor(mats[0],device='cuda',dtype=torch.float32)
    y=torch.as_tensor(cohorts[0]['y'],device='cuda',dtype=torch.float32)
    rank=torch.as_tensor(ranking[0],device='cuda',dtype=torch.float32)
    targets=None if soft is None else torch.as_tensor(soft,device='cuda',dtype=torch.float32)
    oof=np.empty(len(y),np.float32) if crossfit else None
    dev,held=(np.empty(len(t),np.float32) for t in ranking[1:])
    store={}
    folds=cohorts[0]['fold']
    copies=cohorts[0]['duplicates']
    for side in (0,2):
        lane=cohorts[0]['observed_regime']==side
        if crossfit:
            for k in np.unique(folds):
                include=lane&eligible&((copies&(1<<int(k)))==0)
                outside=lane&(folds==k)
                assert not (include&outside).any()
                est=set_fit(x,y,rank,include,salt=int(k)+side)
                oof[outside]=predict(mats[0][outside],ranking[0][outside],est)
        final=set_fit(x,y,rank,lane&eligible,teacher=targets,salt=side)
        store[side]=final
        for i,out in enumerate((dev,held),1):
            there=cohorts[i]['observed_regime']==side
            out[there]=predict(mats[i][there],ranking[i][there],final)
    assert np.isfinite(dev).all() and np.isfinite(held).all()
    if crossfit:
        assert np.isfinite(oof).all()
    return store,(oof,dev,held)


def main(rotation):
    started=perf_counter()
    frames,threshold,ticket=load(rotation)
    history=json_at(LATENT/rotation/'receipt.json')
    assert history['matrix_sha256']==ticket
    assert history['code_sha256']==checksum(HERE/'compact41.py')
    assert history['alignment_sha256']==checksum(LATENT/rotation/'alignment.npz')
    state=LATENT/rotation/'states.npz'
    assert checksum(state)==history['state_sha256']
    graph,unix=dense_sources(rotation,ticket)
    anchor=frames['dev']
    tail=frames.get('test',anchor)
    role=[frames['oof'],crop(anchor,anchor['cal']),crop(tail,tail['held'])]
    names,_,projects,paths,_=f.inputs()
    covered=[crop(x,supported(x)) for x in role]
    y=covered[0]
    held_ids=np.unique(np.r_[anchor['ids'],tail['ids']])
    assert not np.intersect1d(y['ids'],held_ids).size
    if 'test' in frames:
        assert not set(projects[y['ids']])&set(projects[tail['ids']])
    token_copies={fingerprint(names[i]) for i in held_ids}
    eligible=np.fromiter((fingerprint(names[i]) not in token_copies for i in y['ids']),bool)
    ranks=[x['signed'].clip(-12.,12.).astype(np.float32) for x in covered]
    plans={}
    cash={}
    sources={'oof':frames['oof'],'dev':anchor}
    if 'test' in frames:
        sources['test']=tail
    with np.load(state) as cache:
        compact=[]
        for name,frame in zip(('oof','dev','test' if 'test' in frames else 'dev'),covered,strict=True):
            baseline=sources[name]
            valid=supported(baseline)
            assert np.array_equal(cache[name+'_ids'],baseline['ids'][valid])
            # Selected phases preserve the ordering of their parent frames.
            retrieve={i:j for j,i in enumerate(cache[name+'_ids'])}
            ix=np.fromiter((retrieve[key] for key in frame['ids']),np.int64)
            compact.append(cache[name][ix])
    for name,fn in [('graph',lambda r:feature_rows(r['ids'],graph)),('unix',lambda r:unix_rows(r['ids'],unix))]:
        features_=[columns(z,threshold,fn(z)) for z in covered]
        plans[name],cash[name]=fit_scopes(name,features_,ranks,covered,eligible,crossfit=True)
        print(rotation,name,'teacher',round(perf_counter()-started,2),flush=True)
        del features_
    cash['average']=tuple((a+b)*.5 for a,b in zip(cash['graph'],cash['unix'],strict=True))
    compact=[columns(z,threshold,item) for z,item in zip(covered,compact,strict=True)]
    recipes=[('score',None),('compact',None),*((f'distill_{k}',cash[k][0]) for k in ('graph','unix','average'))]
    for label,logit in recipes:
        x=compact if label!='score' else [columns(z,threshold) for z in covered]
        plans[label],cash[label]=fit_scopes(label,x,ranks,covered,eligible,soft=logit)
        print(rotation,label,'readout',round(perf_counter()-started,2),flush=True)
    names=('oof','cal','held')
    output={k+'_ids':value['ids'] for k,value in zip(names,covered,strict=True)}
    for key,v in cash.items():
        for role,pred in zip(names,v,strict=True):
            if pred is not None:
                output[role+'_'+key]=pred
    d=DEST/rotation
    d.mkdir(exist_ok=True,parents=True)
    predicted=d/'scores.npz'
    models=d/'weights.joblib'
    np.savez_compressed(predicted,**output)
    joblib.dump(plans,models,compress=3)
    write_json(d/'receipt.json',dict(rotation=rotation,matrix_sha256=ticket,model_sha256=checksum(models),
             predictions_sha256=checksum(predicted),compact_sha256=checksum(state),graph_sha256=checksum(GRAPH/'receipt.json'),
             unix={k:v[2] for k,v in unix.items()},removed_validation_copies=int((~eligible).sum()),
             rows={k:len(z['ids']) for k,z in zip(names,covered,strict=True)},
             temperature=T,teacher_weight=TRANSFER,code_sha256=checksum(Path(__file__))))
    print(rotation,'done',round(perf_counter()-started,2),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
