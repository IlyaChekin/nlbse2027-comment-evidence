"""Predict the two components of the contest's weakness label and combine them."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from components49 import DEST as ANNOTATIONS, SOURCE
from route49 import HERE, ROOT, LAYOUT, PARTS, f, checksum, json_at, load, write_json, physical
from source41 import parts
from project_split import fingerprint
from data40 import features
from policy33 import logit
from stack40 import diagnostic_order, per_project
from eval35 import cut, adjusted

DEST = HERE/'decomposed'
FEATURES = ('numeric','layout','portfolio')
METHODS = ('direct_all','direct_matched','conditional')
PANELS = ('all','unqueried','negative','positive')
WIDTHS = (1.25,3.8,None)
MIX = (.5,1.)
PARAM = dict(max_iter=95,max_leaf_nodes=19,min_samples_leaf=55,learning_rate=.065,
             l2_regularization=8.,early_stopping=False,random_state=149)


def design(frame, threshold, mode, anatomy):
    own, old, new, _, side = features(frame,threshold)
    assert np.array_equal(side,frame['observed_regime'])
    extra = {'numeric':(), 'layout':(anatomy,), 'portfolio':(old,new)}[mode]
    return np.column_stack((own,*extra)).astype(np.float32)


def fit(x,y,components,keep,eligible,margin):
    selected = keep & eligible
    annotation_rule = (components & 1)>0
    separate = (components & 6)>0
    assert np.array_equal(y[selected],annotation_rule[selected] | separate[selected])
    weight = .45+1.6*np.exp(-.35*abs(margin))
    # Estimate the residual source only among examples where W is negative.
    uncovered = selected & ~annotation_rule
    def head(scope,target):
        return HistGradientBoostingClassifier(**PARAM).fit(x[scope],target[scope],sample_weight=weight[scope])
    controls={'direct_all': head(keep,y),'direct_matched': head(selected,y)}
    factors=(head(selected,annotation_rule),head(uncovered,separate))
    return controls,factors


def predict(bank,x):
    controls,(weak,extension)=bank
    w,e=(g.predict_proba(x)[:,1] for g in (weak,extension))
    return {name:head.decision_function(x) for name,head in controls.items()} | {
              'conditional':logit(w+(1.-w)*e)},w,e


def mask(frame,panel,width):
    region = np.ones(len(frame['ids']),bool)
    if panel!='all':
        side = {'negative':0,'positive':1,'unqueried':2}[panel]
        region &= frame['observed_regime']==side
    if width is not None:
        region &= abs(frame['signed'])<width
    return region


def report(rotation, row, estimates):
    cal,held=row
    y=[r['y'] for r in row]
    base=[r['signed']>0. for r in row]
    snapshots=dict(ids=held['ids'],base=base[1],truth=y[1],cal_ids=cal['ids'])
    records={}
    for panel,width in product(PANELS,WIDTHS):
        if panel=='all' and width==3.8:
            continue
        scope=[mask(r,panel,width) for r in row]
        if min(int(x.sum()) for x in scope)<3:
            continue
        variants=[*product(FEATURES,METHODS,MIX),('numeric','threshold',0.)]
        for kind,mode,w in variants:
            sk=f'{kind}/{mode}'
            label=f'{panel}/{width}/{sk}/{w:g}'
            q=[(1.-w)*r['signed'][m]+w*s[m]
               for r,s,m in zip(row,estimates.get(sk,[r['signed'] for r in row]),scope,strict=True)]
            item=dict(panel=panel,width=width,mode=mode,features=kind,mix=w,
                      cal_rows=int(scope[0].sum()),held_rows=int(scope[1].sum()))
            if rotation!='official':
                order,_=diagnostic_order(q[1],held['signed'][scope[1]],held['ids'][scope[1]])
                item['matched'],saved=per_project(y[1],held['project'],base[1],scope[1],order,('fp','tp'))
                for mechanism,bits in saved.items():
                    snapshots[label+'_at_'+mechanism]=bits
            for mode in ('fixed','fp','tp'):
                t=cut(y[0][scope[0]],base[0][scope[0]],q[0],mode)
                a,_=adjusted(y[0],base[0],scope[0],q[0],t)
                b,bits=adjusted(y[1],base[1],scope[1],q[1],t)
                item[mode]=dict(cut=t,calibration=a,held=b)
                snapshots[label+'_'+mode]=bits
            records[label]=item
    return records,snapshots


def main(rotation):
    start=perf_counter()
    folds,t,stamp=load(rotation)
    train = folds['oof']
    d=folds['dev']
    h=folds.get('test',d)
    roles=parts(folds)
    ids=train['ids']
    observed=np.unique(np.r_[d['ids'],h['ids']])
    assert not np.isin(ids,observed).any()
    texts,labels,family,_,_=f.inputs()
    if rotation!='official':
        assert not set(family[ids]) & set(family[h['ids']])
    dup={fingerprint(texts[i]) for i in observed}
    keep=np.fromiter((fingerprint(texts[i]) not in dup for i in ids),bool)
    proof=ANNOTATIONS/'receipt.json'
    receipt=json_at(proof)
    label_file=proof.with_name('labels.npz')
    assert receipt['predictions_sha256']==checksum(label_file)
    assert receipt['code_sha256']==checksum(HERE/'components49.py')
    assert receipt['archive_sha256']==checksum(SOURCE)
    with np.load(label_file) as source:
        y=source['truth'][ids]
        assert np.array_equal(y,train['y']) and np.array_equal(y,labels[ids,0])
        components=source['component'][ids]
        eligible=source['selected'][ids]
    card=json_at(LAYOUT/'receipt.json')
    table=LAYOUT/'features.npz'
    assert card['matrix_sources'][rotation]==stamp and card['predictions_sha256']==checksum(table)
    with np.load(table) as cached:
        where,values=cached['ids'],cached['layout']
    structure=[physical(z,where,values) for z in (train,*roles)]
    models={}
    by_view={}
    projected={}
    for scope in FEATURES:
        in_fit,*apply=[design(z,t,scope,b) for z,b in zip((train,*roles),structure,strict=True)]
        bank=fit(in_fit,y,components,keep,eligible,train['signed'])
        models[scope]=bank
        for n,q in enumerate(apply):
            forecast,pweak,rest=predict(bank,q)
            for name,res in forecast.items():
                by_view.setdefault(scope+'/'+name,[]).append(res)
            projected[str(n)+'_'+scope+'_weak']=pweak
            projected[str(n)+'_'+scope+'_residual']=rest
        print(rotation,scope,'seconds',round(perf_counter()-start,2),flush=True)
    result,saved=report(rotation,roles,by_view)
    where=DEST/rotation
    where.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(where/'predictions.npz',**saved,**projected)
    joblib.dump(models,where/'models.joblib',compress=3)
    active=keep&eligible
    known=active&train['old']
    weak=(components & 1)>0
    evidence=dict(fit=int(keep.sum()),matched=int(active.sum()),rule=int((active&weak).sum()),
                  residual=int((active&~weak&y).sum()),overlap=int((active&weak&((components&6)>0)).sum()),
                  observed_rule=int(known.sum()),weak_agreement=int((known&(weak==(train['flag']==1))).sum()))
    summary=dict(rotation=rotation,fit=evidence,results=result,source_sha256=checksum(proof),
                 matrix_sha256=stamp,layout_sha256=checksum(LAYOUT/'receipt.json'),
                 code_sha256=checksum(Path(__file__)),weights_sha256=checksum(where/'models.joblib'),
                 predictions_sha256=checksum(where/'predictions.npz'),seconds=perf_counter()-start)
    write_json(where/'results.json',summary)
    print(rotation,len(result),evidence,round(perf_counter()-start,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
