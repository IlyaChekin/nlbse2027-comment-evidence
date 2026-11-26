"""Distinguish correct and incorrect scanner vetoes by comparing both route margins."""
import argparse
from itertools import product

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from points33 import HERE, PARTS, OUT, f, checksum, json_at, write_json
from experiment33 import read_phase, crop, cut
from policy33 import matrix
from eval32 import per_project

DEST=HERE/'veto'
GUARDS=('disagreement','all_positive_prior','observed_negative')
BANDS=(.8,1.8,3.5,None)
STRENGTHS=(.35,.7)


def descriptors(data, threshold):
    first=matrix(data['pre'],data['length'],threshold)
    p,q=data['unqueried'],data['queried']
    a,b=np.clip(p,-8.,8.),np.clip(q,-8.,8.)
    return np.column_stack((b,a,b-a,(b>0.)!=(a>0.),b*first[:,8],a*first[:,8],first))


def territory(data, mode):
    exact=data['queried']
    prior=data['unqueried']
    weak=data['flag']==0
    scope=weak & data['old']
    if mode=='disagreement':
        scope &= (prior>0.) != (exact>0.)
    elif mode=='all_positive_prior':
        scope &= prior>0.
    return scope


def train(data, threshold):
    x=descriptors(data,threshold)
    y=data['y']
    reference=data['queried']
    weights=.35+2.*np.exp(-abs(reference))
    rows={}
    fits={}
    for scope in GUARDS[1:]:
        use=territory(data,scope)
        z,w,gold=x[use],weights[use],y[use]
        linear=LogisticRegression(C=.6,solver='liblinear',random_state=42,max_iter=300)
        tree=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=17,learning_rate=.06,
               min_samples_leaf=max(26,len(z)//1700),l2_regularization=5.,random_state=42,early_stopping=False)
        rows[scope]=dict(rows=int(use.sum()),positives=int(gold.sum()))
        fits[scope,'linear']=linear.fit(z,gold,sample_weight=w)
        fits[scope,'tree']=tree.fit(z,gold,sample_weight=w)
    return fits,rows


def infer(m, data, threshold):
    x=descriptors(data,threshold)
    return {key:head.decision_function(x) for key,head in m.items()}


def test(rotation):
    source=OUT/rotation
    receipt=json_at(source/'receipt.json')
    phases={k:read_phase(k,source,receipt) for k in receipt['phases']}
    d=phases['dev']
    other=phases.get('test',d)
    x,z=crop(d,d['cal']),crop(other,other['held'])
    y,t=x['y'],z['y']
    origin,baseline=x['replay']>0.,z['replay']>0.
    cutoff=receipt['cutoff']
    family={'fit':crop(d,d['fit']),'oof':phases['oof']}
    models,notes={},{}
    for role,data in family.items():
        models[role],notes[role]=train(data,cutoff)
    ranks={role:[infer(model,p,cutoff) for p in (x,z)] for role,model in models.items()}
    cfg={}
    snapshots=dict(ids=z['ids'],old=baseline,truth=t,rule=z['flag'],queried=z['old'],before=z['unqueried'],
                   after=z['queried'])
    for route,width in product(GUARDS,BANDS):
        u=territory(x,route)
        v=territory(z,route)
        if width is not None:
            u &= abs(x['queried'])<width
            v &= abs(z['queried'])<width
        if not u.any():
            continue
        code='all' if width is None else f'{width:g}'
        for style,role,strength in [('control','shared',0.),*product(('linear','tree'),models,STRENGTHS)]:
            p,q=x['queried'][u],z['queried'][v]
            key=(route if route!='disagreement' else 'all_positive_prior',style)
            if strength:
                a,b=ranks[role]
                p=(1.-strength)*p+strength*a[key][u]
                q=(1.-strength)*q+strength*b[key][v]
            name=f'{route}_{code}_{style}_{role}_{strength:g}'
            options={}
            if rotation!='official':
                matched,forecasts=per_project(t,z['project'],baseline,v,q,('tp','fp'))
                options['matched']=matched
                for mode,bits in forecasts.items():
                    snapshots[name+'_matched_'+mode]=bits
            for mode in ('tp','fp','fixed'):
                boundary=cut(y[u],origin[u],p,mode)
                a,b=origin.copy(),baseline.copy()
                a[u],b[v]=p>boundary,q>boundary
                options[mode]=dict(cut=boundary,cal=f.comparison(y,origin,a),frozen=f.comparison(t,baseline,b))
                snapshots[name+'_'+mode]=b
            cfg[name]=dict(route=route,width=width,style=style,source=role,strength=strength,
                           options=options,cal_rows=int(u.sum()),review_rows=int(v.sum()))
    dest=DEST/rotation
    dest.mkdir(parents=True,exist_ok=True)
    here=dest/'predictions.npz'
    np.savez_compressed(here,**snapshots)
    w=dest/'models.joblib'
    joblib.dump(models,w,compress=3)
    write_json(dest/'results.json',dict(rotation=rotation,training=notes,options=cfg,code_sha256=checksum(HERE/'rescue33.py'),
                  predictions_sha256=checksum(here),models_sha256=checksum(w),points_sha256=checksum(source/'receipt.json')))
    print(rotation,'veto rows',notes,'choices',len(cfg),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    test(p.parse_args().rotation)
