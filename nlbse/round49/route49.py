"""Train before-query routing policies from the two signed decision outcomes."""
import argparse
from bisect import bisect_left
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from layout49 import HERE, ROOT, DEST as LAYOUT, PARTS, f, checksum, json_at, load, write_json
from source41 import parts
from project_split import fingerprint
from route40 import fields
from policy33 import BOUNDARIES, reassign
from experiment33 import budget
from stack40 import diagnostic_order, per_project

DEST = HERE/'fitted'
VIEWS = ('numeric','portfolio','layout','portfolio_layout')
MODELS = ('direct','factor')
FRACTIONS = (.025,.08,.20)
PENALTY = .95
OPTIONS = dict(max_iter=72,max_leaf_nodes=11,min_samples_leaf=50,learning_rate=.065,
               l2_regularization=7.,early_stopping=False,random_state=49)


def physical(phase, roster, design):
    ids = phase['ids']
    at = np.searchsorted(roster,ids)
    assert np.array_equal(roster[at],ids)
    return design[at]


def inputs(phase, t, kind, structure):
    main = fields(phase,t,'all' if kind.startswith('portfolio') else 'score')
    if 'layout' in kind:
        return np.column_stack((main,structure))
    return main


def endpoint(phase):
    unqueried = phase['unqueried']
    queried = np.where(phase['old'],phase['signed'],phase['queried'])
    queried = np.where(phase['flag']>=0,queried,unqueried)
    assert np.array_equal(np.where(phase['old'],queried,unqueried)>0.,phase['signed']>0.)
    return unqueried,queried


def targets(frame):
    p,q = endpoint(frame)
    before,after = p>0.,q>0.
    switched = before != after
    beneficial = switched & (after == frame['y'])
    harmful = switched & ~beneficial
    category = np.select((beneficial,harmful),(1,2),default=0).astype(np.int8)
    delta = after.astype(np.int8)-before
    utility = delta*np.where(frame['y'],1.,-PENALTY)
    return category,utility


def hypotheses(x, flag):
    return np.column_stack((x,np.broadcast_to(flag,len(x))))


def fit(train, x, keep):
    category, utility = targets(train)
    flag = train['flag']
    observed = keep & (flag>=0)
    assert np.isin(flag[observed],(0,1)).all()
    scout = HistGradientBoostingClassifier(**OPTIONS).fit(x[observed],flag[observed])
    direct, routed = {},{}
    for old in (False,True):
        use = observed & (train['old']==old)
        direct[old] = HistGradientBoostingRegressor(**OPTIONS).fit(x[use],utility[use])
        q = hypotheses(x[use],flag[use])
        routed[old] = HistGradientBoostingClassifier(**OPTIONS).fit(q,category[use])
        assert np.array_equal(routed[old].classes_,[0,1,2])
    return dict(direct=direct,routed=routed,scout=scout)


def estimate(bank, x, frame):
    cost = np.where(frame['unqueried']>0.,PENALTY,1.)
    loss = np.where(frame['unqueried']>0.,1.,PENALTY)
    ready = {name:np.empty(len(x),np.float32) for name in MODELS}
    chance = bank['scout'].predict_proba(x)[:,1]
    for old in (False,True):
        subset = frame['old']==old
        xp = x[subset]
        assert len(xp)
        model = bank['routed'][old]
        a,b = (model.predict_proba(hypotheses(xp,v)) for v in (0,1))
        outcome = (1.-chance[subset,None])*a+chance[subset,None]*b
        ready['factor'][subset] = cost[subset]*outcome[:,1]-loss[subset]*outcome[:,2]
        ready['direct'][subset] = bank['direct'][old].predict(xp)
    return ready,chance.astype(np.float32)


def no_longer(frame, score, fraction):
    # Reject donors whose estimated loss exceeds the benefit of the proposed request.
    # Among the low-utility donors, choose the shortest adequate replacement.
    old,length = (frame[k] for k in ('old','length'))
    selected = old.copy()
    bands = np.searchsorted(BOUNDARIES,length,side='right')
    for band in np.unique(bands):
        available=(bands==band)
        a=np.flatnonzero(available & old)
        b=np.flatnonzero(available & ~old & (score>0.))
        cap=min(len(a)//3,len(b),int(len(a)*fraction))
        if not cap:
            continue
        low=a[np.argsort(score[a],kind='stable')[:len(a)//3]]
        high=b[np.argsort(-score[b],kind='stable')]
        donor=sorted((int(length[i]),float(score[i]),int(i)) for i in low)
        pay=0
        for i in high:
            start=bisect_left(donor,(int(length[i]),-np.inf,-1))
            while start<len(donor) and donor[start][1]>=score[i]:
                start+=1
            if start==len(donor):
                continue
            _,_,j=donor.pop(start)
            selected[i],selected[j]=True,False
            pay+=1
            if pay==cap:
                break
    assert length[selected].sum()<=length[old].sum()
    assert np.array_equal(np.bincount(bands[old],minlength=len(BOUNDARIES)+1),
                          np.bincount(bands[selected],minlength=len(BOUNDARIES)+1))
    return selected


def allocations(frame, scores, fraction, mode):
    if mode=='matched_length':
        return no_longer(frame,scores,fraction)
    return reassign(frame['old'],frame['length'],scores,frame['pre'][:,8],1.,fraction)


def record(frame, mask, previous, alternate):
    a,b=endpoint(frame)
    rank=np.where(mask,b,a)
    estimate=np.where(mask!=frame['old'],rank,frame['signed'])
    candidate=estimate>0.
    assert np.array_equal(candidate[mask==frame['old']],previous[mask==frame['old']])
    changes=f.comparison(frame['y'],previous,candidate)
    new=(~frame['old'])&mask
    drop=frame['old']&~mask
    contract=budget(frame,mask)
    assert contract['calls']==contract['old_calls'] and contract['characters']<=contract['old_characters']
    changed=dict(new=f.comparison(frame['y'],previous, np.where(new,candidate,previous)),
                 removed=f.comparison(frame['y'],previous,np.where(drop,candidate,previous)))
    curve={}
    saves={}
    if alternate:
        order,ties=diagnostic_order(estimate,frame['signed'],frame['ids'])
        curve,saves=per_project(frame['y'],frame['project'],previous,np.ones(len(previous),bool),order,('fp','tp'))
    return dict(budget=contract,frozen=changes,action_effect=changed,matched=curve),candidate,estimate,saves


def case_count(frame, keep=None):
    y=frame['y']
    if keep is None:
        keep=np.ones(len(y),bool)
    u,v=endpoint(frame)
    res={}
    for k,mask in (('new',~frame['old']),('already',frame['old'])):
        z=mask&keep&((u>0.)!=(v>0.))
        res[k]=dict(switches=int(z.sum()),better=int((z&((v>0.)==y)).sum()),
                    worse=int((z&((v>0.)!=y)).sum()),available=int((mask&keep).sum()))
    return res


def main(rotation):
    start=perf_counter()
    src,t,stamp=load(rotation)
    train=src['oof']
    profiles=parts(src)
    val=src['dev']
    co=src.get('test',val)
    never=np.unique(np.r_[val['ids'],co['ids']])
    assert not np.isin(train['ids'],never).any()
    documents,labels,projects,files,_=f.inputs()
    if 'test' in src:
        assert not set(projects[train['ids']])&set(projects[co['ids']])
    assert all(np.array_equal(labels[x['ids'],0],x['y']) for x in (train,*profiles))
    tokens={fingerprint(documents[i]) for i in never}
    keep=np.fromiter((fingerprint(documents[i]) not in tokens for i in train['ids']),bool)
    card=json_at(LAYOUT/'receipt.json')
    feature=LAYOUT/'features.npz'
    assert card['matrix_sources'][rotation]==stamp
    assert card['code_sha256']==checksum(HERE/'layout49.py') and card['predictions_sha256']==checksum(feature)
    with np.load(feature) as src:
        roster,structure=src['ids'],src['layout']
    all_cases=(train,*profiles)
    lookup=[physical(z,roster,structure) for z in all_cases]
    truth=[z['signed']>0. for z in profiles]
    forecasts,model,metrics={}, {}, {}
    for view in VIEWS:
        design=[inputs(z,t,view,u) for z,u in zip(all_cases,lookup,strict=True)]
        model[view]=fit(train,design[0],keep)
        forecasts[view]=[estimate(model[view],x,z) for x,z in zip(design[1:],profiles,strict=True)]
        print(rotation,view,round(perf_counter()-start,2),flush=True)
    folder=DEST/rotation
    folder.mkdir(exist_ok=True,parents=True)
    held=profiles[1]
    payload=dict(ids=held['ids'],truth=held['y'],base=truth[1],old=held['old'],size=held['length'],
                 calibration_ids=profiles[0]['ids'])
    for view,(a,b) in forecasts.items():
        for index,output in enumerate((a,b)):
            pred,flag=output
            payload.update({f'fold{index}_{view}_{mode}':scores for mode,scores in pred.items()})
            payload[f'fold{index}_{view}_flag']=flag
    print(rotation,'routing',flush=True)
    for view,kind,reason,portion in product(VIEWS,MODELS,('legacy','matched_length'),FRACTIONS):
        key=f'{view}/{kind}/{reason}/{portion:g}'
        options={}
        for j,(phase,before,scope) in enumerate(zip(profiles,truth,('calibration','held'),strict=True)):
            proposal=allocations(phase,forecasts[view][j][0][kind],portion,reason)
            report,decision,rank,matched=record(phase,proposal,before,rotation!='official' and scope=='held')
            options[scope]=report
            if scope=='held':
                payload[key+'_mask']=proposal
                payload[key+'_prediction']=decision
                for name,bits in matched.items():
                    payload[key+'_at_'+name]=bits
        metrics[key]=options
    weight=folder/'models.joblib'
    table=folder/'predictions.npz'
    joblib.dump(model,weight,compress=3)
    np.savez_compressed(table,**payload)
    write_json(folder/'results.json',dict(rotation=rotation,matrix_sha256=stamp,
               layout_sha256=checksum(LAYOUT/'receipt.json'),predictions_sha256=checksum(table),
               weights_sha256=checksum(weight),code_sha256=checksum(Path(__file__)),
               exclusions=int((~keep).sum()),cases={k:case_count(z,keep if k=='fit' else None)
               for k,z in zip(('fit','calibration','held'),all_cases,strict=True)},results=metrics,
               seconds=perf_counter()-start,
               approximation='changed newly queried rows use stored Round 32 ranks, existing requests use Round 38'))
    print(rotation,len(metrics),round(perf_counter()-start,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
