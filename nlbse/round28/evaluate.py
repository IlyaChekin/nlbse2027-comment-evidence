"""Compare matched-rank heads at the prior budget on disjoint projects."""
import argparse
from itertools import product
from statistics import median
import time

import joblib
import numpy as np
from scipy.sparse import load_npz, vstack
from sklearn.linear_model import LogisticRegression
import torch

from pairs import HERE, ROOT, OUT, PARTS, f, checksum, json_at, write_json
from paired_context import context, match, strict_cut
from head_runtime import collect, head, infer_extra

WIDTHS = (.35, .8, 1.7, 3.2, None)
WEIGHTS = (.35, .7, 1.)
CUTS = ('fp', 'tp', 'fixed')
STUDY = HERE/'studies'


def linear(x,y,weights=None, c=.5, intercept=True):
    return LogisticRegression(C=c,solver='liblinear',max_iter=200,random_state=42,
                              fit_intercept=intercept).fit(x,y,sample_weight=weights)


def heads(rotation, d, t, cal):
    directory=HERE/'heads'/rotation
    metadata=json_at(directory/'fit.json')
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    rankers={}
    out_d={}
    out_t={}
    batch_d = [s for s,k in zip(d.texts,d.active,strict=True) if k]
    batch_t = batch_d if t is d else [s for s,k in zip(t.texts,t.active,strict=True) if k]
    pd, bd = collect(model,tok,batch_d)
    pt, bt = (pd,bd) if t is d else collect(model,tok,batch_t)
    source = {}
    for name in ('point','pair'):
        path=directory/f'{name}.pt'
        if checksum(path) != metadata['modes'][name]['sha256']:
            raise ValueError('Head checkpoint changed')
        classifier=head(model.width).cuda().eval()
        classifier.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            out_d[name]=(bd+classifier(pd)[:,0]).float().cpu().numpy()
            out_t[name]=out_d[name] if t is d else (bt+classifier(pt)[:,0]).float().cpu().numpy()
        source[name]=metadata['modes'][name]['sha256']
        rankers[name]=classifier
    costs={}
    if t is d:
        local=[s for s,k in zip(d.texts,cal,strict=True) if k]
        for name, classifier in rankers.items():
            for opt in ((),(classifier,)):
                infer_extra(model,tok,local,opt)
            differences=[]
            base_sec=[]
            extra_sec=[]
            for i in range(5):
                samples=(('reference',()),('candidate',(classifier,)))
                ticks={}
                for label,opt in (samples if i%2==0 else samples[::-1]):
                    p,aux,hi,info=infer_extra(model,tok,local,opt)
                    if not np.allclose(aux[:,20],d.frame['reference'][cal,2],rtol=0.,atol=.004):
                        raise ValueError('The integrated readout changed the original primary auxiliaries')
                    ticks[label]=info['forward_seconds']
                differences.append(ticks['candidate']-ticks['reference'])
                base_sec.append(ticks['reference'])
                extra_sec.append(ticks['candidate'])
            costs[name]=dict(seconds=max(0.,median(differences)),trials=differences,baseline=base_sec,new=extra_sec,
                             gflops=2*(2*model.width*64+64)*len(local)/1e9)
    return out_d,out_t,costs,source,None


def sparse(rotation,d,t,cal,kind):
    path=ROOT/'round24/features/matrix.npz' if kind=='sparse' else HERE/'canonical/matrix.npz'
    proof=json_at(path.with_name('manifest.json'))
    if checksum(path)!=proof['matrix_sha256']:
        raise ValueError('Training token vectors changed')
    from sparse_model import WORDS
    x=load_npz(path)[:,:WORDS.n_features].tocsr()
    pair_path=OUT/rotation/'pairs.npz'
    if checksum(pair_path)!=json_at(pair_path.with_name('manifest.json'))['predictions_sha256']:
        raise ValueError('Matched pair input changed')
    with np.load(pair_path) as z:
        rows,y,pairs=(z[k] for k in ('rows','label','pairs'))
    common=x[rows]
    diffs=common[pairs[:,0]]-common[pairs[:,1]]
    ypair=np.r_[np.ones(len(pairs),dtype=np.int8),np.zeros(len(pairs),dtype=np.int8)]
    contrasts=vstack((diffs,-diffs),format='csr')
    weights=np.ones(len(y))
    np.add.at(weights,pairs.ravel(),1.15)
    scored={}
    scores_dev={}
    scores_test={}
    costs={}
    X=d.frame['ids'][d.active]
    Y=t.frame['ids'][t.active]
    A=x[X]
    B=A if t is d else x[Y]
    fit=linear(common,y,weights, c=1.0)
    pair=linear(contrasts,ypair,None,c=1.0,intercept=False)
    for name, hs in (('point',(fit,)),('pair',(fit,pair))):
        rd=[h.decision_function(A) for h in hs]
        rt=rd if d is t else [h.decision_function(B) for h in hs]
        scores_dev[name] = np.column_stack(rd)
        scores_test[name] = np.column_stack(rt)
        scored[name] = hs
    return scores_dev,scores_test,costs,{'matrix':proof['matrix_sha256'],'pairs':checksum(pair_path)},scored


def correction(x,z,fitted):
    a=np.column_stack((x,z))
    return fitted.decision_function(a)


def compare(rotation,kind):
    d,t,fit,cal,review,old,base,p_dev,p_test=context(rotation)
    if kind == 'head':
        a,b,head_cost,sources,payload=heads(rotation,d,t,cal)
    else:
        a,b,head_cost,sources,payload=sparse(rotation,d,t,cal,kind)
    held=np.ones(len(base),dtype=bool) if review is None else review
    prefix=STUDY/kind/rotation
    prefix.mkdir(parents=True,exist_ok=True)
    tags={}
    files=dict(ids=t.frame['ids'][held],y=t.frame['y'][held],base=base[held],margin=p_test[held])
    proposals={}
    dy,ty=d.frame['y'],t.frame['y']
    actual=f.comparison(dy[cal],old[cal],old[cal])['f1_before']
    for name in a:
        x=a[name][...,None] if a[name].ndim==1 else a[name]
        y=b[name][...,None] if b[name].ndim==1 else b[name]
        xd=np.column_stack((p_dev[d.active],x))
        xt=np.column_stack((p_test[t.active],y))
        target=dy[fit & d.active]
        estimate=linear(xd[fit[d.active]],target,c=.8)
        trajectories=(estimate.decision_function(xd),estimate.decision_function(xt))
        proposals[name]=dict(readout=estimate)
        if payload is not None:
            proposals[name]['rankers']=payload[name]
        for width in WIDTHS:
            selected=d.active.copy()
            chosen=t.active.copy()
            if width is not None:
                selected &= np.abs(p_dev)<width
                chosen &= np.abs(p_test)<width
            gold=dy[selected&cal]
            anchor=old[selected&cal]
            if not len(gold):
                continue
            premium={}
            if review is not None:
                if kind=='head':
                    premium=head_cost[name]
                else:
                    names=[s for s,x in zip(d.texts,selected&cal,strict=True) if x]
                    from sparse_model import vectorize
                    from canonical import vectors
                    times=[]
                    for attempt in range(4):
                        start=time.perf_counter()
                        matrix=vectors(names) if kind=='canonical' else vectorize(names,'token')
                        v=np.column_stack([h.decision_function(matrix) for h in payload[name]])
                        correction(p_dev[selected&cal],v,estimate)
                        times.append(time.perf_counter()-start)
                    premium=dict(seconds=median(times[1:]),gflops=2*matrix.nnz*len(payload[name])/1e9,
                                 trials=times[1:],rows=len(names))
            da=trajectories[0][selected[d.active]]
            tb=trajectories[1][chosen[t.active]]
            for strength in WEIGHTS:
                key=f'{name}_{width if width is not None else "all"}_{strength:g}'
                v=(1.-strength)*p_dev[selected]+strength*da
                w=(1.-strength)*p_test[chosen]+strength*tb
                row=dict(mode=name,width=width,mix=strength,calls=int((chosen&held).sum()),
                         cal_calls=len(gold),cost=premium,cuts={})
                if review is None:
                    matched, per=match(ty,t.frame['project'],base,chosen,w)
                    row.update(matched=f.comparison(ty,base,matched),project=per)
                    files[key+'_matched']=matched
                files[key+'_selected']=chosen[held]
                files[key+'_rank']=w[held[chosen]]
                for criterion in CUTS:
                    threshold=strict_cut(gold,anchor,v[cal[selected]],criterion)
                    modified=old.copy()
                    candidate=base.copy()
                    modified[selected]=v>threshold
                    candidate[chosen]=w>threshold
                    fit_eval=f.comparison(dy[cal],old[cal],modified[cal])
                    stats=dict(cut=threshold,cal=fit_eval,frozen=f.comparison(ty[held],base[held],candidate[held]))
                    if review is not None:
                        tax=.2*f.PUBLIC_N/cal.sum()*(premium['seconds']/1000+premium['gflops']/3e6)
                        stats['proxy']=.3*(fit_eval['f1_after']-actual)-tax
                    row['cuts'][criterion]=stats
                    files[key+'_'+criterion]=candidate[held]
                tags[key]=row
    file=prefix/'readouts.joblib'
    joblib.dump(proposals,file,compress=3)
    preds=prefix/'predictions.npz'
    np.savez_compressed(preds,**files)
    write_json(prefix/'results.json',dict(options=tags,weights_sha256=checksum(file),predictions_sha256=checksum(preds),
                sources=sources,scope=dict(dev=int(d.active.sum()),held=int((t.active&held).sum()))))
    print(rotation,kind, len(tags),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    p.add_argument('kind',choices=('head','sparse','canonical'))
    a=p.parse_args()
    compare(a.part,a.kind)
