"""Test disagreement-guided acquisition with fixed scanner calls and character budget."""
import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data40 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, attach
from data38 import crop
from policy33 import matrix, reassign
from eval35 import cut, per_project, adjusted
from stack40 import diagnostic_order, calibrate, bank, basic

DEST=HERE/'routing'
SPEC=('score','old','new','all')
FRACTIONS=(.12,.28,.55,1.)
WEIGHTS=(.35,.7,1.)


def fields(x,threshold,which):
    base=matrix(x['pre'],x['length'],threshold)
    if which=='score':
        return base
    before=x['unqueried']
    candidate=x['new'].clip(-12.,12.)
    first,second=x['sparse'].T
    right=np.column_stack((candidate,candidate-before,candidate*base[:,10],
                          abs(candidate-before),candidate*base[:,12]))
    approx=(1.-x['pre'][:,4])*first+x['pre'][:,4]*second
    left=np.column_stack((first,second,approx,approx-before,abs(approx-before),
                          approx*base[:,12]))
    if which=='old':
        return np.column_stack((base,left))
    if which=='new':
        return np.column_stack((base,right))
    return np.column_stack((base,left,right))


def decide(old,allocation):
    kept=old['old']&allocation
    new=~old['old']&allocation&(old['flag']>=0)
    result=old['unqueried'].copy()
    result[kept]=old['signed'][kept]
    result[new]=old['queried'][new]
    assert np.array_equal(result[kept],old['signed'][kept])
    return result


def train(x,y,weights,method):
    if method=='ridge':
        return make_pipeline(StandardScaler(),Ridge(alpha=48.)).fit(x,y,ridge__sample_weight=weights)
    reg=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=23,min_samples_leaf=85,
             learning_rate=.065,l2_regularization=8.,early_stopping=False,random_state=40)
    return reg.fit(x,y,sample_weight=weights)


def benchmark(rotation,frames,all_estimates):
    a,b=frames
    d={}
    predictions=dict(ids=b['ids'],truth=b['y'],reference=b['signed']>0.,old_scan=b['old'])
    truth=[p['y'] for p in frames]
    control=[p['signed']>0. for p in frames]
    full=[np.ones(len(x),bool) for x in truth]
    for (which,method),report in all_estimates.items():
        for fraction,w in product(FRACTIONS,WEIGHTS):
            key=f'{which}/{method}/{fraction:g}/{w:g}'
            rank=[]
            contracts=[]
            for case,score in zip(frames,report,strict=True):
                allocated=reassign(case['old'],case['length'],score,case['pre'][:,8],w,fraction)
                rank.append(decide(case,allocated))
                contracts.append(dict(calls=int(allocated.sum()),changed=int((allocated!=case['old']).sum()),
                            characters=int(case['length'][allocated].sum()),
                            prior_characters=int(case['length'][case['old']].sum())))
            row=dict(model=method,experts=which,fraction=fraction,strength=w,contracts=contracts)
            if rotation!='official':
                ordered,ties=diagnostic_order(rank[1],b['signed'],b['ids'])
                row['matched'],bits=per_project(truth[1],b['project'],control[1],full[1],ordered,('fp','tp'))
                row['ties']=ties
                predictions[key+'_match_fp']=bits['fp']
            for regime in ('fixed','fp','tp'):
                offset=cut(truth[0],control[0],rank[0],regime)
                x,_=adjusted(truth[0],control[0],full[0],rank[0],offset)
                v,bits=adjusted(truth[1],control[1],full[1],rank[1],offset)
                row[regime]=dict(cut=offset,cal=x,held=v)
                predictions[key+'_'+regime]=bits
            d[key]=row
    return d,predictions


def oracle(data,t):
    rows=data['oof']
    ref=rows['signed']>0.
    _,_,_,raw,side=basic(rows,t)
    head=calibrate(raw,side,rows['y'],ref)
    result={}
    dev=data['dev']
    held=data.get('test',dev)
    sections=(('oof',rows),('calibration',crop(dev,dev['cal'])),('held',crop(held,held['held'])))
    for role,frame in sections:
        _,_,_,raw,side=basic(frame,t)
        ref=frame['signed']>0.
        signals={}
        for name in SPEC:
            alternate=bank(raw,side,head,name)>0.
            scope=(alternate!=ref[:,None]).any(1)
            signals[name]=dict(disagreed=int(scope.sum()),by_regime={
                str(i):dict(fn=int((scope&(side==i)&frame['y']&~ref).sum()),
                            fp=int((scope&(side==i)&~frame['y']&ref).sum()),
                            available=int((scope&(side==i)).sum())) for i in range(3)})
        result[role]=signals
    return result


def main(rotation):
    data,ref,proof=attach(rotation)
    fitting=data['oof']
    x=fitting
    offset=np.where(x['flag']>=0,x['queried'],x['unqueried'])
    delta=(offset>0.).astype(np.int8)-(x['unqueried']>0.).astype(np.int8)
    y=delta*np.where(x['y'],1.,-.88)
    emphasis=.40+1.45*(delta!=0)
    dev=data['dev']
    val=data.get('test',dev)
    cohorts=(crop(dev,dev['cal']),crop(val,val['held']))
    predictions,models={},{}
    for name in SPEC:
        layout=fields(x,ref['cutoff'],name)
        observed=[fields(v,ref['cutoff'],name) for v in cohorts]
        for method in ('ridge','tree'):
            model=train(layout,y,emphasis,method)
            models[name,method]=model
            predictions[name,method]=[model.predict(v) for v in observed]
    stats,selected=benchmark(rotation,cohorts,predictions)
    out=DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    path=out/'predictions.npz'
    np.savez_compressed(path,**selected)
    trained=out/'models.joblib'
    joblib.dump(models,trained,compress=3)
    write_json(out/'results.json',dict(rotation=rotation,values=stats,
              oracle=oracle(data,ref['cutoff']),sources=proof,
              predictions_sha256=checksum(path),weights_sha256=checksum(trained),
              code_sha256=checksum(Path(__file__)),
              assumption='existing requests replay Round 38; a newly requested row uses the signed Round 32 endpoint'))
    print(rotation,len(stats),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
