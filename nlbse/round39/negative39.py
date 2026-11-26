"""Probe matched semantic prototypes in the consensus-negative branch."""
import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from location39 import HERE, ROOT, PARTS, checksum, json_at, write_json
from data39 import load_ready, archived, f, routes, pairs, crop
from rank38 import fit, predict, cut, per_project, adjusted
from broad35 import features as score_features
from matched_veto33 import evaluation

DEST=HERE/'negative'
DENSE=ROOT/'round30/embeddings'
MODES=('base','teacher','prototype','both')
VIEWS=2
CLUSTERS=6


def take(matrix, ids, query):
    p=np.searchsorted(ids,query)
    assert np.array_equal(ids[p],query)
    return np.asarray(matrix[p],np.float32)


def transform(source, p, n):
    fit_rows=np.unique(np.r_[p,n])
    mean=source[fit_rows].mean(0)
    std=source[fit_rows].std(0).clip(.02,None)
    width=source.shape[1]//VIEWS
    pair=[]
    for s in range(VIEWS):
        columns=slice(s*width,(s+1)*width)
        local=source[:,columns]
        def learn(x):
            z=normalize((local[x]-mean[columns])/std[columns])
            kd=MiniBatchKMeans(n_clusters=CLUSTERS,n_init=2,max_iter=45,random_state=39,batch_size=256)
            kd.fit(z)
            return normalize(kd.cluster_centers_)
        pair.append((learn(np.unique(p)),learn(np.unique(n))))
    return mean,std,pair


def compare_proto(features,model):
    mean,scale,centroids=model
    parts=np.split((features-mean)/scale,VIEWS,axis=1)
    combined=[]
    for x,(p,n) in zip(parts,centroids,strict=True):
        z=normalize(x)
        a,b=z@p.T,z@n.T
        ap,bp=a.max(1),b.max(1)
        combined.extend((ap,bp,ap-bp,np.mean(a,1)-np.mean(b,1)))
    return np.column_stack(combined)


def territory(r):
    return routes(r)['negative'] & np.isfinite(r['negative_teacher1'])


def main(rotation):
    data,ref,sources,geometry=load_ready(rotation)
    choice,base=archived(rotation,data,ref,sources)
    oof=data['oof']
    subset=crop(oof,territory(oof))
    a,b,pair_proof=pairs(rotation,oof,'negative_consensus')
    masked=territory(oof)
    ix=np.flatnonzero(masked)
    good=masked[a]&masked[b]
    p,n=np.searchsorted(ix,a[good]),np.searchsorted(ix,b[good])
    assert np.array_equal(ix[p],a[good]) and np.array_equal(ix[n],b[good])
    v=json_at(DENSE/'manifest.json')
    assert checksum(DENSE/'features.npy')==v['data_sha256']
    assert checksum(DENSE/'rows.npz')==v['rows_sha256']
    with np.load(DENSE/'rows.npz') as z:
        ids=z['ids']
    dense=np.load(DENSE/'features.npy',mmap_mode='r')
    assert (ids[1:]>ids[:-1]).all()
    x=take(dense,ids,subset['ids'])
    oof_sim=np.full((len(x),VIEWS*4),np.nan)
    for fold in np.unique(subset['fold']):
        allowed=subset['duplicates']&(1<<int(fold))==0
        held=subset['fold']==fold
        assert not (allowed&held).any()
        right=allowed[p]&allowed[n]
        model=transform(x,p[right],n[right])
        oof_sim[held]=compare_proto(x[held],model)
    assert np.isfinite(oof_sim).all()
    proto=transform(x,p,n)
    _,_,family,files,_=f.inputs()
    group=[(family[i],files[i]) for i in oof['ids'][a[good]]]
    base_rank=subset['signed']
    diagnostic=dict(control=evaluation(base_rank,p,n,group))
    for element in (2,6):
        difference=oof_sim[:,element]
        diagnostic[str(element)+'_only']=evaluation(difference,p,n,group)
        diagnostic[str(element)+'_with_control']=evaluation(base_rank+2.*difference,p,n,group)
    t=ref['cutoff']
    dev=data['dev']
    test=data.get('test',dev)
    parts=(crop(dev,dev['cal']),crop(test,test['held']))
    selected=[territory(q) for q in parts]
    locals_=[crop(q,use) for q,use in zip(parts,selected,strict=True)]
    x=[compare_proto(take(dense,ids,z['ids']),proto) for z in locals_]
    trainers={}
    scored={}
    for mode in MODES:
        def design(case, values):
            raw=score_features(case,t,'regime')
            if mode in ('teacher','both'):
                raw=np.column_stack((raw,case['negative_teacher1']))
            if mode in ('prototype','both'):
                raw=np.column_stack((raw,values))
            return raw
        model=fit(design(subset,oof_sim),base_rank,subset['y'],p,n)
        trainers[mode]=model
        views=[]
        for raw,stats,chosen in zip(locals_,x,selected,strict=True):
            q=np.full(len(chosen),np.nan)
            q[chosen]=predict(model,design(raw,stats),raw['signed'])
            views.append(q)
        scored[mode]=views
    y,z=[v['y'] for v in parts]
    ab,bb=[v['signed']>0. for v in parts]
    summary={}
    save=dict(ids=parts[1]['ids'],truth=z,baseline=bb)
    for radius,mix in product((1.25,3.8,None),(.3,.7)):
        for mode in MODES:
            region=[]
            ranks=[]
            for i,(frame,consensus) in enumerate(zip(parts,selected,strict=True)):
                active=consensus if radius is None else consensus&(abs(frame['signed'])<radius)
                positive=routes(frame)['positive']&(abs(frame['signed'])<3.8)
                rank=frame['signed'].copy()
                rank[active]=(1.-mix)*rank[active]+mix*scored[mode][i][active]
                combined=active|positive
                region.append(combined)
                ranks.append(rank[combined])
            k=f'{radius}/{mix}/{mode}'
            a,b=region
            ap,bp=ranks
            row={}
            if rotation!='official':
                row['matched'],out=per_project(z,parts[1]['project'],bb,b,bp,('tp','fp'))
                save[k+'_matched']=out['fp']
            for how in ('fixed','fp','tp'):
                threshold=cut(y[a],ab[a],ap,how)
                left,_=adjusted(y,ab,a,ap,threshold)
                right,out=adjusted(z,bb,b,bp,threshold)
                row[how]=dict(cal=left,held=right,cut=threshold)
                save[k+'_'+how]=out
            summary[k]=row
    out=DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    path=out/'predictions.npz'
    weight=out/'models.joblib'
    np.savez_compressed(path,**save)
    joblib.dump(dict(prototypes=proto,readout=trainers),weight,compress=3)
    write_json(out/'results.json',dict(rotation=rotation,rows=len(subset['ids']),pairs=int(good.sum()),
                       base_sha256=base,geometry_sha256=geometry,pairs_sha256=pair_proof,code_sha256=checksum(Path(__file__)),
                       predictions_sha256=checksum(path),weights_sha256=checksum(weight),
                       features_sha256=v['data_sha256'],diagnostics=diagnostic,options=summary))
    print(rotation,{k:round(x['pair_win'],4) for k,x in diagnostic.items()},flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
