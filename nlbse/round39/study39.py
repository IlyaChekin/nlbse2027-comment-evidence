"""Compare numeric locations and nearby text with the same released control."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from location39 import HERE, ROOT, PARTS, checksum, json_at, write_json
from data39 import load_ready, archived, oof, f, routes, pairs, crop
from context39 import ordinary
from rank38 import positive_features, fit, predict, cut, per_project, adjusted
from matched_veto33 import evaluation

DEST=HERE/'comparison'
VIEW=('coordinate','layout','local','center')
ARMS=('base','base-pair',*(f'{part}-{loss}' for part,loss in product(VIEW,('point','pair'))))
STRENGTHS=(.25,.55,.85)
WIDTHS=(.9,1.8,3.8,None)
POLICIES=('pooled','dominant','strong')
DOMINANT=2


def scope(frame):
    return routes(frame)['positive']&(frame['location'][:,3]>0)


def vectorizer():
    return HashingVectorizer(analyzer='word',tokenizer=str.split,token_pattern=None,
                ngram_range=(1,2),n_features=2**17,norm='l2',alternate_sign=True,
                lowercase=False,dtype=np.float32)


def smooth_sparse(matrix,left,right):
    diff=matrix[left]-matrix[right]
    z=vstack((diff,-diff),format='csr')
    y=np.arange(z.shape[0])<len(left)
    _,key,total=np.unique(right,return_inverse=True,return_counts=True)
    mass=1./total[key]
    mass/=mass.mean()
    clf=LogisticRegression(C=.4,fit_intercept=False,solver='liblinear',random_state=39,max_iter=200)
    return clf.fit(z,y,sample_weight=np.tile(mass,2))


def sparse_scores(train,inputs,left,right,rotation):
    output={}
    heads={}
    for meaning,pair_ in enumerate(('local','center')):
        model=vectorizer()
        ob,others=inputs[0][meaning],inputs[1:]
        matrix=model.transform(ob)
        values=np.full(len(ob),np.nan)
        for fold in np.unique(train['fold']):
            allowed=train['duplicates']&(1<<int(fold))==0
            held=train['fold']==fold
            assert not (allowed&held).any()
            keep=allowed[left]&allowed[right]
            f=smooth_sparse(matrix,left[keep],right[keep])
            values[held]=f.decision_function(matrix[held])
        assert np.isfinite(values).all()
        f=smooth_sparse(matrix,left,right)
        heads[pair_]=f
        result=[values,*(f.decision_function(model.transform(x[meaning])) for x in others)]
        output[pair_]=result
        print(rotation,'sparse',pair_,len(ob),len(left),flush=True)
    return output,heads


def matrix(case, threshold, arm, shape, rank=None):
    control=positive_features(case,threshold,case['aux_positive'],case['first'],True)
    if arm.startswith('base'):
        return control
    if arm.startswith('coordinate'):
        return np.column_stack((control,shape[:,:9]))
    if arm.startswith('layout'):
        return np.column_stack((control,shape))
    return np.column_stack((control,shape,rank))


def learned(rotation,data,ref):
    train=data['oof']
    dev=data['dev']
    ext=data.get('test',dev)
    cases=(train,crop(dev,dev['cal']),crop(ext,ext['held']))
    a,b,pair_sha=pairs(rotation,train,'positive_retained')
    raw=[scope(x) for x in cases]
    pick=np.flatnonzero(raw[0])
    eligible=raw[0][a]&raw[0][b]
    left=np.searchsorted(pick,a[eligible])
    right=np.searchsorted(pick,b[eligible])
    assert np.array_equal(pick[left],a[eligible]) and np.array_equal(pick[right],b[eligible])
    reduced=[crop(x,v) for x,v in zip(cases,raw,strict=True)]
    corpus,_,family,files,_=f.inputs()
    view=[]
    for row in reduced:
        shape,actual,center=ordinary([corpus[i] for i in row['ids']],row['location'])
        view.append((shape,actual,center))
    sparse,heads=sparse_scores(reduced[0],[x[1:] for x in view],left,right,rotation)
    modelset={}
    predicted={}
    r=reduced[0]
    for mode in ARMS:
        stem=mode.split('-')[0]
        signal=sparse.get(stem,[None]*len(cases))
        x=[matrix(z,ref['cutoff'],mode,v[0],sig)
           for z,v,sig in zip(reduced,view,signal,strict=True)]
        use=(left,right) if mode.endswith('-pair') else (left[:0],right[:0])
        model=fit(x[0],r['signed'],r['y'],*use)
        modelset[mode]=model
        values=[]
        for arr,mask,rows in zip(x[1:],raw[1:],reduced[1:],strict=True):
            scores=np.zeros(len(mask))
            scores[mask]=predict(model,arr,rows['signed'])
            values.append(scores)
        predicted[mode]=values
    groups=[(family[i],files[i]) for i in train['ids'][b[eligible]]]
    baseline=r['signed']
    measures={'before':evaluation(-baseline,right,left,groups)}
    for key,val in sparse.items():
        meters=[evaluation(-z,right,left,groups) for z in (val[0],baseline+.30*val[0])]
        measures[key]={'standalone':meters[0],'plus_baseline':meters[1]}
    return cases[1:],predicted,modelset,heads,dict(pairs_sha256=pair_sha,pairs=int(eligible.sum()),
                   counts=[int(x.sum()) for x in raw],ranking=measures)


def use_for(rows,name,width):
    mask=scope(rows)
    if name in ('dominant','strong'):
        mask &= rows['location'][:,1]==DOMINANT
    if name=='strong':
        mask &= rows['signed']>=1.8
    if width is not None:
        mask &= abs(rows['signed'])<width
    return mask


def assess(rotation,data,score):
    cal,test=data
    before=[z['signed']>0. for z in data]
    y=[z['y'] for z in data]
    table={}
    saved=dict(ids=test['ids'],base=before[1],truth=y[1])
    for panel,w in product(POLICIES,WIDTHS):
        mask=[use_for(x,panel,w) for x in data]
        if not all(z.any() for z in mask):
            continue
        if panel=='strong' and w not in (3.8,None):
            continue
        band=f'{panel}/{w}'
        for arm,amount in [('threshold',0.),*product(ARMS,STRENGTHS)]:
            key=f'{band}/{arm}/{amount:g}'
            q=[]
            for i,(a,z) in enumerate(zip(data,mask,strict=True)):
                delta=a['signed'][z]
                if amount:
                    delta=(1.-amount)*delta+amount*score[arm][i][z]
                q.append(delta)
            record=dict(panel=panel,width=w,mode=arm,mix=amount,cal_rows=int(mask[0].sum()),
                        held_rows=int(mask[1].sum()))
            if rotation!='official':
                record['matched'],bits=per_project(y[1],test['project'],before[1],mask[1],q[1],('fp','tp'))
                saved[key+'_matched']=bits['fp']
            for method in ('fixed','tp','fp'):
                offset=cut(y[0][mask[0]],before[0][mask[0]],q[0],method)
                record['cal_'+method],_=adjusted(y[0],before[0],mask[0],q[0],offset)
                record['held_'+method],bits=adjusted(y[1],before[1],mask[1],q[1],offset)
                record['cut_'+method]=offset
                saved[key+'_'+method]=bits
            table[key]=record
    return table,saved


def main(rotation):
    t=perf_counter()
    data,ref,sources,geometry=load_ready(rotation)
    settings,base=archived(rotation,data,ref,sources)
    folds,digest=oof(rotation,data,ref['cutoff'],settings)
    print(rotation,'replayed',folds,flush=True)
    cases,score,models,heads,description=learned(rotation,data,ref)
    result,saved=assess(rotation,cases,score)
    out=DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    pred=out/'predictions.npz'
    coef=out/'models.joblib'
    np.savez_compressed(pred,**saved)
    joblib.dump(dict(readout=models,heads=heads),coef,compress=3)
    write_json(out/'results.json',dict(rotation=rotation,options=result,model_sha256=checksum(coef),
                       predictions_sha256=checksum(pred),base_sha256=base,geometry_sha256=geometry,
                       pairs_sha256=digest,crossfit=folds,study=description,
                       code_sha256=checksum(Path(__file__)),context_sha256=checksum(HERE/'context39.py'),
                       data_sha256=checksum(HERE/'data39.py')))
    print(rotation,'options',len(result),'seconds',perf_counter()-t,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
