"""Compare calibration over every observed negative outcome using only existing scores."""
import argparse

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from teacher35 import HERE, PARTS, checksum, write_json, json_at
from eval35 import loaded, reference, crop, point_classifier, design, comparisons

DEST=HERE/'broad'


def regimes(source):
    prior=source['unqueried']>0.
    acquired=source['queried']>0.
    return (2*prior+acquired).astype(np.int32)


def features(data,t,mode):
    base=design(data,t)
    role=regimes(data)
    if mode=='linear':
        return base
    votes=role[:,None]==np.arange(4)[None,:]
    gauge=base[:,:3]
    return np.column_stack((base,votes,(gauge[:,:,None]*votes[:,None,:]).reshape(len(base),-1)))


def evaluate(rotation):
    data,ref,old=loaded(rotation)
    _,baseline=reference(rotation,data,ref)
    runtime=ref['cutoff']
    archive={}
    for name,item in data.items():
        original=item['support'].copy()
        item['support']=item['old']&(item['flag']==0)
        assert not (original&~item['support']).any()
        if name=='oof':
            archive.update(ids=item['ids'],fold=item['fold'],eligible=item['support'],
                           regime=regimes(item),truth=item['y'])
    d=data['dev']
    other=data.get('test',d)
    selected,held=crop(d,d['cal']),crop(other,other['held'])
    rank={}
    models={}
    reports={}
    for source,train in [('fit',crop(d,d['fit'])),('oof',data['oof'])]:
        y=train['y']
        where=train['support']
        for mode in ('linear','regime','trees'):
            x=features(train,runtime,mode)
            if mode=='trees':
                clf=HistGradientBoostingClassifier(max_iter=105,max_leaf_nodes=17,learning_rate=.06,
                    min_samples_leaf=max(30,int(where.sum())//1500),random_state=42,l2_regularization=5.,
                    early_stopping=False)
                spectrum=.3+1.35*np.exp(-.5*np.abs(train['latest'][where]))
                clf.fit(x[where],y[where],sample_weight=spectrum)
            else:
                clf=point_classifier(x,y,train['latest'],where)
            rank[source,mode]=[clf.decision_function(features(d,runtime,mode)) for d in (selected,held)]
            models[source,mode]=clf
        reports[source]=dict(rows=int(where.sum()),positive=int(y[where].sum()),
                             counts=np.bincount(regimes(train)[where],minlength=4).tolist())
    result,forecast=comparisons(rotation,selected,held,rank)
    p=DEST/rotation
    p.mkdir(parents=True,exist_ok=True)
    pred=p/'predictions.npz'
    np.savez_compressed(pred,**forecast)
    index=p/'cohorts.npz'
    np.savez_compressed(index,**archive)
    model=p/'readouts.joblib'
    joblib.dump(models,model,compress=3)
    write_json(p/'results.json',dict(rotation=rotation,options=result,cohorts_sha256=checksum(index),
                              predictions_sha256=checksum(pred),weights_sha256=checksum(model),
                              code_sha256=checksum(HERE/'broad35.py'),baseline_sha256=baseline,
                              points_sha256=old['points_sha256'],fit=reports))
    print(rotation,reports,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    evaluate(p.parse_args().rotation)
