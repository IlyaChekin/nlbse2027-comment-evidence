"""Fit monotone nonlinear calibrators for the confident false positives."""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from study43 import (HERE, ROOT, PARTS, f, checksum, json_at, write_json, load_ready, archived,
                     oof, sources, pairs, crop, available, fingerprint, sparse_first, design, compare)
from select43 import budget, change

DEST = HERE/'conditional'
VIEWS = {'numeric':(),'lexical':('text','masked'),'syntax':('ast',),'combined':('text','masked','ast')}


def prepare(rotation):
    matrices,baseline,source,fixture = load_ready(rotation)
    settings,release = archived(rotation,matrices,baseline,source)
    folds,pair_proof = oof(rotation,matrices,baseline['cutoff'],settings)
    ids,geo,views,receipt = sources(fixture)
    val = matrices['dev']
    held = matrices.get('test',val)
    full = (matrices['oof'],crop(val,val['cal']),crop(held,held['held']))
    masks = [available(r) for r in full]
    frames = [crop(row,mask) for row,mask in zip(full,masks,strict=True)]
    head,train = full[0],frames[0]
    texts,_,families,_,_ = f.inputs()
    excluded = np.unique(np.r_[val['ids'],held['ids']])
    assert not np.isin(head['ids'],excluded).any()
    if 'test' in matrices:
        assert not set(families[head['ids']])&set(families[held['ids']])
    words = {fingerprint(texts[i]) for i in excluded}
    use = np.fromiter((fingerprint(texts[i]) not in words for i in train['ids']),bool)
    pos,neg,digest = pairs(rotation,head,'positive_retained')
    assert digest==pair_proof
    valid = masks[0][pos]&masks[0][neg]
    rr = np.flatnonzero(masks[0])
    a,b = [np.searchsorted(rr,x[valid]) for x in (pos,neg)]
    assert np.array_equal(rr[a],pos[valid]) and np.array_equal(rr[b],neg[valid])
    lookup = [np.searchsorted(ids,row['ids']) for row in frames]
    assert all(np.array_equal(ids[k],row['ids']) for k,row in zip(lookup,frames,strict=True))
    values,sparse = sparse_first(frames,{k:v for k,v in views.items() if k!='center'},lookup,a,b,use)
    info = dict(folds=folds,cache_sha256=receipt,pairs_sha256=digest,base_sha256=release,
                removed_copies=int((~use).sum()))
    designs = {}
    for arm,ch in VIEWS.items():
        designs[arm] = [
            np.column_stack((row['signed'],design(row,baseline['cutoff'],geo[ix],[values[k][j] for k in ch])))
            for j,(row,ix) in enumerate(zip(frames,lookup,strict=True))]
    return full[1:],masks[1:],train,use,designs,sparse,info


def main(rotation):
    start = perf_counter()
    phases,masks,train,allowed,designs,sparse,meta = prepare(rotation)
    y = train['y'][allowed]
    score = train['signed'][allowed]
    near = .7+1.7*np.exp(-.48*abs(score))
    bad = (score>=1.8)&~y
    forecasts,models = {},{}
    for tone,weight in [('near',near),('veto',near+2.*bad)]:
        for arm in VIEWS:
            name = tone+'_'+arm
            fit,cal,review = designs[arm]
            monotone = [1]+[0]*(fit.shape[1]-1)
            estimator = HistGradientBoostingClassifier(learning_rate=.08,max_iter=110,max_leaf_nodes=15,
                           min_samples_leaf=80,l2_regularization=12.,monotonic_cst=monotone,
                           early_stopping=False,random_state=43)
            estimator.fit(fit[allowed],y,sample_weight=weight)
            models[name] = estimator
            result = []
            for x,mask in zip((cal,review),masks,strict=True):
                p = np.full(len(mask),np.nan)
                p[mask] = estimator.decision_function(x)
                result.append(p)
            forecasts[name] = result
    report,pred = compare(rotation,phases,forecasts)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',**pred)
    joblib.dump(dict(readout=models,sparse=sparse),out/'weights.joblib',compress=3)
    write_json(out/'results.json',dict(options=report,training=meta|{'veto_errors':int(bad.sum())},
                   predictions_sha256=checksum(out/'predictions.npz'),weights_sha256=checksum(out/'weights.joblib'),
                   code_sha256=checksum(Path(__file__)),seconds=perf_counter()-start))
    print(rotation,len(report),'strong training FP',int(bad.sum()),round(perf_counter()-start,2),flush=True)


def select():
    frames,certificate = {},{}
    for r in PARTS:
        d = DEST/r
        path = d/'results.json'
        receipt = json_at(path)
        assert receipt['predictions_sha256']==checksum(d/'predictions.npz')
        assert receipt['weights_sha256']==checksum(d/'weights.joblib')
        assert receipt['code_sha256']==checksum(Path(__file__))
        assert receipt['training']['cache_sha256']==checksum(HERE/'features/receipt.json')
        frames[r] = receipt['options']
        certificate[r] = checksum(path)
    a,b,c = (frames[k] for k in PARTS)
    report = []
    for key in sorted(set(a)&set(b)&set(c)):
        state = c[key]
        if state['arm']=='threshold':
            continue
        prefix,name,amount = key.rsplit('/',2)
        tone,view = name.split('_')
        sources = [prefix+'/threshold/0']
        if view!='numeric':
            sources.append(f'{prefix}/{tone}_numeric/{amount}')
        if view in ('combined','syntax'):
            sources.append(f'{prefix}/{tone}_lexical/{amount}')
        if tone=='veto':
            sources.append(f'{prefix}/near_{view}/{amount}')
        x = [budget(m[key],'tp') for m in (a,b)]
        y = [budget(m[key],'fp') for m in (a,b)]
        p = [n-max(budget(m[s],'tp') for s in sources) for n,m in zip(x,(a,b),strict=True)]
        q = [n-max(budget(m[s],'fp') for s in sources) for n,m in zip(y,(a,b),strict=True)]
        holds = any(min(w)>=0 and sum(w)>0 for w in (p,q))
        for rule in ('fixed','fp','tp'):
            cal,review = (change(state[rule],stage) for stage in ('cal','held'))
            minimum = max(0.,max(change(c[s][rule],'cal') for s in sources))
            report.append(dict(key=key,arm=name,rule=rule,tone=tone,view=view,panel=state['panel'],
                               width=state['width'],mix=state['mix'],threshold=state[rule]['cut'],
                               rows_cal=state['cal_rows'],rows_review=state['review_rows'],
                               a_fp=x[0],b_fp=x[1],a_tp=y[0],b_tp=y[1],a_fp_over=p[0],b_fp_over=p[1],
                               a_tp_over=q[0],b_tp_over=q[1],cal_f1=cal,review_f1=review,
                               matched=holds,eligible=holds and cal>minimum and review>=0.))
    with (DEST/'comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=report[0])
        writer.writeheader()
        writer.writerows(report)
    rank = lambda v:(min(v['cal_f1'],v['review_f1']),v['cal_f1']+v['review_f1'],-v['rows_review'])
    eligible = [v for v in report if v['eligible']]
    structural = [v for v in report if v['matched'] and v['view'] in ('syntax','combined')]
    chosen = max(eligible,key=rank,default=None)
    best_structure = max(structural,key=rank,default=None)
    write_json(DEST/'selection.json',dict(chosen=chosen,best_structure=best_structure,points=len(report),
                   sources=certificate,csv_sha256=checksum(DEST/'comparison.csv'),code_sha256=checksum(Path(__file__))))
    print('tested',len(report),'chosen',chosen,'structural',best_structure,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=(*PARTS,'select'))
    value = cli.parse_args().stage
    select() if value=='select' else main(value)
