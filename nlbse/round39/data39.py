"""Align new scanner summaries to the released Round 38 operating point."""
from pathlib import Path

import joblib
import numpy as np

from location39 import HERE, ROOT, OUTPUT, checksum, json_at, write_json
from data38 import f, phases, routes, pairs, crop
from rank38 import installed, active_set, positive_features, predict, fit

R38=ROOT/'round38'


def load_ready(rotation):
    data,ref,base=phases(rotation)
    sources=installed(rotation,data,base)
    fixture=OUTPUT/'training.npz'
    receipt=json_at(OUTPUT/'receipt.json')
    assert receipt['predictions_sha256']==checksum(fixture)
    with np.load(fixture) as z:
        ids,geometry=z['ids'],z['values']
    for scope in data.values():
        keep=scope['old']
        index=np.searchsorted(ids,scope['ids'][keep])
        assert np.array_equal(ids[index],scope['ids'][keep])
        scope['location']=np.zeros((len(keep),geometry.shape[1]),np.int32)
        scope['location'][keep]=geometry[index]
        pos=routes(scope)['positive']
        assert np.array_equal(scope['location'][pos,1:3],scope['first'][pos])
    return data,ref,sources,checksum(OUTPUT/'receipt.json')


def archived(rotation,data,ref,sources):
    folder=R38/'comparison'/rotation
    p=folder/'results.json'
    plan=json_at(R38/'selection.json')
    expected=plan['sources'][rotation]
    assert checksum(p)==expected
    proof=json_at(p)
    assert proof['sources']==sources
    path=folder/'models.joblib'
    assert checksum(path)==proof['weights_sha256']
    settings=plan['selected']
    local=proof['options'][settings['key']][settings['rule']]
    detail=settings|{'cut':local['cut']}
    assert detail['negative']=='off' and detail['positive']=='pair'
    trained=joblib.load(path)['positive','pair']
    t=ref['cutoff']
    for role,rows in data.items():
        if role=='oof':
            continue
        mask=routes(rows)['positive']&(abs(rows['signed'])<detail['width_positive'])
        frame=crop(rows,mask)
        x=positive_features(frame,t,frame['aux_positive'],frame['first'],True)
        target=frame['signed']
        y=predict(trained,x,target)
        weight=detail['mix']
        result=(1.-weight)*target+weight*y-detail['cut']
        rows['signed'][mask]=result
        rows['latest'][mask]=result
    saved=folder/'predictions.npz'
    assert checksum(saved)==proof['predictions_sha256']
    held=data.get('test',data['dev'])
    with np.load(saved) as z:
        assert np.array_equal(z['ids'],held['ids'][held['held']])
        k=settings['key']+'_'+settings['rule']
        assert np.array_equal(z[k],held['signed'][held['held']]>0.)
    return detail,expected


def oof(round_,data,threshold,choice):
    """Re-fit only the preceding compact readout, avoiding label reuse in its rank."""
    train=data['oof']
    support=routes(train)['positive']
    rows=crop(train,support)
    indexes=np.flatnonzero(support)
    p,n,digest=pairs(round_,train,'positive_retained')
    left,right=np.searchsorted(indexes,p),np.searchsorted(indexes,n)
    assert np.array_equal(indexes[left],p) and np.array_equal(indexes[right],n)
    x=positive_features(rows,threshold,rows['aux_positive'],rows['first'],True)
    prior=rows['latest']
    rank=prior.copy()
    history={}
    for fold in np.unique(rows['fold']):
        train_mask=(rows['duplicates']&(1<<int(fold)))==0
        score_mask=rows['fold']==fold
        assert not (train_mask&score_mask).any()
        both=train_mask[left]&train_mask[right]
        fit_at=np.flatnonzero(train_mask)
        l,r=np.searchsorted(fit_at,left[both]),np.searchsorted(fit_at,right[both])
        trained=fit(x[train_mask],prior[train_mask],rows['y'][train_mask],l,r)
        use=score_mask&(abs(prior)<choice['width_positive'])
        value=predict(trained,x[use],prior[use])
        w=choice['mix']
        rank[use]=(1.-w)*prior[use]+w*value-choice['cut']
        history[int(fold)]=dict(fit=int(train_mask.sum()),held=int(score_mask.sum()),pairs=int(both.sum()),
                                gradient=trained['gradient'])
    train['signed'][support]=rank
    train['latest'][support]=rank
    return history,digest
