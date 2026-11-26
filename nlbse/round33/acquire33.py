"""Test targeted additions and removals using separate flip and correctness models."""
import argparse
from itertools import product

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from points33 import HERE, PARTS, OUT, f, checksum, json_at, write_json
from experiment33 import read_phase, crop, cut, matched, budget
from policy33 import logit, matrix, reassign
from views33 import DEST as VIEW

DEST=HERE/'separate'
FRACTIONS=(.01,.025,.06,.12)
STAGES=('paired','add','remove')


def descriptors(data, threshold, extra):
    x=matrix(data['pre'],data['length'],threshold)
    if extra:
        raw,read=data['view'].T
        first=logit(raw.astype(np.float64))
        return np.column_stack((x,first,read,x[:,0]-first))
    return x


def fit(train, threshold, extra):
    x=descriptors(train,threshold,extra)
    no=train['unqueried']>0.
    flip=no!=(train['queried']>0.)
    labels=train['y']
    params=dict(max_iter=105,max_leaf_nodes=19,learning_rate=.067,l2_regularization=4.,
                min_samples_leaf=max(26,len(no)//2900),random_state=42,early_stopping=False)
    results={}
    records={}
    for name,scope in [('new',~train['old']),('already',train['old'])]:
        y=scope & flip
        if not y.any() or len(np.unique(labels[y]))!=2:
            raise ValueError('Missing outcomes for a directional router')
        detect=HistGradientBoostingClassifier(**params).fit(x[scope],flip[scope])
        judge=HistGradientBoostingClassifier(**params).fit(x[y],labels[y])
        results[name]=dict(flip=detect,correct=judge)
        records[name]=dict(rows=int(scope.sum()),switches=int(y.sum()))
    return results,records


def expected(m, data, threshold, extra, weight):
    x=descriptors(data,threshold,extra)
    estimated=np.empty(len(x))
    before=data['unqueried']>0.
    for key, scope in [('already',data['old']),('new',~data['old'])]:
        model=m[key]
        moves=model['flip'].predict_proba(x[scope])[:,1]
        truth=model['correct'].predict_proba(x[scope])[:,1]
        change=truth-weight*(1.-truth)
        estimated[scope]=moves*np.where(before[scope],-change,change)
    return estimated


def bounded(scores, eligible, sizes, count, chars):
    chosen=np.zeros(len(scores),bool)
    i=np.flatnonzero(eligible & (scores>0.))
    if not len(i) or count<1 or chars<1:
        return chosen
    priorities=i[np.argsort(-scores[i],kind='stable')]
    used=0
    accepted=0
    for k in priorities:
        size=int(sizes[k])
        if used+size>chars:
            continue
        chosen[k]=True
        used+=size
        accepted+=1
        if accepted>=count:
            break
    return chosen


def design(data, expected_gain, mode, frac):
    old=data['old']
    lengths=data['length']
    n=int(old.sum()*frac)
    chars=int(lengths[old].sum()*frac)
    if mode=='add':
        a=bounded(expected_gain,~old,lengths,n,chars)
        return old | a
    if mode=='remove':
        a=bounded(-expected_gain,old,lengths,n,chars)
        return old & ~a
    return reassign(old,lengths,expected_gain,data['pre'][:,8],1.,frac)


def evaluate(rotation):
    path=OUT/rotation
    receipt=json_at(path/'receipt.json')
    dest=VIEW/rotation
    proof=json_at(dest/'receipt.json')
    file=dest/'scores.npz'
    if proof['points_sha256']!=checksum(path/'receipt.json') or proof['predictions_sha256']!=checksum(file):
        raise ValueError('First-pass cache changed')
    phases={}
    with np.load(file) as view:
        for phase in receipt['phases']:
            data=read_phase(phase,path,receipt)
            if not np.array_equal(data['ids'],view[phase+'_ids']):
                raise ValueError('First-pass scores change row order')
            data['view']=view[phase]
            phases[phase]=data
    d=phases['dev']
    test=phases.get('test',d)
    a,b=crop(d,d['cal']),crop(test,test['held'])
    y,w=a['y'],b['y']
    cal,held=a['replay']>0.,b['replay']>0.
    t=receipt['cutoff']
    model,notes={},{}
    training={'fit':crop(d,d['fit']),'oof':phases['oof']}
    for source,extra in product(training,(False,True)):
        key=(source,extra)
        model[key],notes[f'{source}_{int(extra)}']=fit(training[source],t,extra)
        print(rotation,key,notes[f'{source}_{int(extra)}'],flush=True)
    reports={}
    preds=dict(ids=b['ids'],old=b['old'],y=w,base=held)
    ways=('fixed','tp','fp')
    results=[('released',False,.9,0.,'fit')]
    for (tag,extra),mode,weight,fraction in product(model,STAGES,(.7,1.15),FRACTIONS):
        results.append((mode,extra,weight,fraction,tag))
    cache={(tag,extra,weight):[expected(bank,s,t,extra,weight) for s in (a,b)]
           for (tag,extra),bank in model.items() for weight in (.7,1.15)}
    for mode,extra,weight,strength,tag in results:
        k=f'{mode}_{tag}_{int(extra)}_{weight:g}_{strength:g}'
        args=(tag,extra,weight)
        if mode=='released':
            q,r=a['old'],b['old']
        else:
            q,r=[design(data,u,mode,strength) for data,u in zip((a,b),cache[args],strict=True)]
        p=np.where(q,a['queried'],a['unqueried'])
        v=np.where(r,b['queried'],b['unqueried'])
        option={}
        for scheme in ways:
            point=cut(y,cal,p,scheme)
            x0,x1=p>point,v>point
            preds[k+'_'+scheme]=x1
            option[scheme]=dict(cal=f.comparison(y,cal,x0),frozen=f.comparison(w,held,x1),cut=point)
        if rotation!='official':
            for scheme in ('fp','tp'):
                option['matched_'+scheme],bits=matched(b,v,held,scheme)
                preds[k+'_matched_'+scheme]=bits
        preds[k+'_requests']=r
        usage0,usage1=budget(a,q),budget(b,r)
        for usage in (usage0,usage1):
            side=int(np.sign(usage['calls']-usage['old_calls']))
            char=np.sign(usage['characters']-usage['old_characters'])
            if mode in ('paired','released'):
                assert side==0 and char<=0
            elif mode=='add':
                assert side>=0 and char>=0
            else:
                assert side<=0 and char<=0
        reports[k]=dict(mode=mode,extra=extra,weight=weight,frac=strength,source=tag,options=option,
                        calibration=usage0,review=usage1)
    folder=DEST/rotation
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/'predictions.npz'
    np.savez_compressed(file,**preds)
    models=folder/'models.joblib'
    joblib.dump(model,models,compress=3)
    write_json(folder/'results.json',dict(rotation=rotation,settings=reports,training=notes,
               predictions_sha256=checksum(file),model_sha256=checksum(models),code_sha256=checksum(HERE/'acquire33.py'),
               points_sha256=checksum(path/'receipt.json'),view_sha256=proof['predictions_sha256']))
    print(rotation,'strategies',len(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    evaluate(p.parse_args().rotation)
