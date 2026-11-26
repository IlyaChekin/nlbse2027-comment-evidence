"""Train separate experts for confirmed negatives and scanner vetoes."""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from study44 import (HERE,ROOT,PARTS,f,checksum,json_at,load,write_json,parts,crop,route,
                     fingerprint,pairs,source_data,sparse_first,design,predict,assess,VIEWS)
from select44 import fp,delta,CONTROL

DEST = HERE/'special'
GROUPS = {'consensus':False,'veto':True}
KEYS = ('numeric','lexical','signature','body','middle','usage','syntax','syntax_text')


def corpus_data(rotation):
    matrices,t,stamp = load(rotation)
    roles = (matrices['oof'],*parts(matrices))
    allowed = [route(row) for row in roles]
    rows = [crop(frame,active) for frame,active in zip(roles,allowed,strict=True)]
    train = rows[0]
    val = matrices['dev']
    held = matrices.get('test',val)
    corpus = f.inputs()
    never = np.unique(np.r_[val['ids'],held['ids']])
    assert not np.isin(train['ids'],never).any()
    duplicates = {fingerprint(corpus[0][i]) for i in never}
    permitted = np.fromiter((fingerprint(corpus[0][i]) not in duplicates for i in train['ids']),bool)
    roster,pair_proof = pairs(rotation,train,permitted,corpus)
    ids,x,feature_proof = source_data(stamp,rotation)
    return roles,rows,allowed,t,permitted,roster,ids,x,dict(matrix_sha256=stamp,pairs_sha256=pair_proof,
                                                           features_sha256=feature_proof,
                                                           copies_excluded=int((~permitted).sum()))


def main(rotation):
    start = perf_counter()
    roles,rows,available,t,permitted,pair_entries,ids,x,receipt = corpus_data(rotation)
    model = {}
    found = {k:[np.full(len(z['ids']),np.nan) for z in roles[1:]] for k in KEYS}
    counts = {}
    for name,state in GROUPS.items():
        assign = [(r['unqueried']>0.)==state for r in rows]
        branched = [crop(r,m) for r,m in zip(rows,assign,strict=True)]
        positions = np.flatnonzero(assign[0])
        filtered = [np.arange(len(r['ids']))[a][z] for r,a,z in zip(roles,available,assign,strict=True)]
        select = permitted[assign[0]]
        training = branched[0]
        pool = np.unique(np.concatenate([np.column_stack(value) for value in pair_entries.values()]),axis=0)
        pool = pool[assign[0][pool].all(1)]
        p,n = np.searchsorted(positions,pool).T
        assert np.array_equal(positions[p],pool[:,0]) and np.array_equal(positions[n],pool[:,1])
        indexes = [np.searchsorted(ids,r['ids']) for r in branched]
        assert all(np.array_equal(ids[v],row['ids']) for v,row in zip(indexes,branched,strict=True))
        evidence,sparse = sparse_first(branched,x,indexes,p,n,select)
        margin = training['signed'][select]
        truth = training['y'][select]
        # Weight the eligible ranking band and the disagreement relevant to this expert.
        errors = (margin<=0.) & truth
        weights = .5+2.1*np.exp(-.68*abs(margin))+.60*errors*np.exp(-.22*abs(margin))
        heads = {}
        for key in KEYS:
            views = VIEWS[key]
            a,*rests = [design(r,t,[evidence[v][j] for v in views]) for j,r in enumerate(branched)]
            estimator = HistGradientBoostingClassifier(max_iter=130,max_leaf_nodes=13,
                        learning_rate=.06,min_samples_leaf=70,l2_regularization=11.,early_stopping=False,
                        monotonic_cst=[1]+[0]*(a.shape[1]-1),random_state=144)
            estimator.fit(a[select],truth,sample_weight=weights)
            heads[key] = estimator
            for destination,index,features in zip(found[key],filtered[1:],rests,strict=True):
                if len(index):
                    destination[index] = estimator.decision_function(features)
        model[name] = dict(readout=heads,sparse=sparse)
        counts[name] = dict(pairs=len(pool),fit=int(select.sum()),cal=len(filtered[1]),held=len(filtered[2]))
    assert all(np.isfinite(z[m]).all() for scores in found.values() for z,m in zip(scores,available[1:],strict=True))
    alternatives = {k+'_tree':v for k,v in found.items()}
    report,pred = assess(rotation,roles[1:],alternatives)
    folder = DEST/rotation
    folder.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(folder/'predictions.npz',**pred)
    joblib.dump(model,folder/'weights.joblib',compress=3)
    write_json(folder/'results.json',dict(options=report,counts=counts,training=receipt,
        predictions_sha256=checksum(folder/'predictions.npz'),weights_sha256=checksum(folder/'weights.joblib'),
        code_sha256=checksum(Path(__file__)),seconds=perf_counter()-start))
    print(rotation,counts,len(report),round(perf_counter()-start,2),flush=True)


def select():
    a,b,c = [],[],[]
    proofs = {}
    pooled_proof = {}
    out = (a,b,c)
    for phase,target in zip(PARTS,out,strict=True):
        file = DEST/phase/'results.json'
        warm = HERE/'comparison'/phase/'results.json'
        qualified,ref = json_at(file),json_at(warm)
        assert qualified['weights_sha256']==checksum(file.with_name('weights.joblib'))
        assert qualified['predictions_sha256']==checksum(file.with_name('predictions.npz'))
        assert qualified['code_sha256']==checksum(Path(__file__))
        assert ref['code_sha256']==checksum(HERE/'study44.py')
        target.extend((qualified['options'],ref['options']))
        proofs[phase] = checksum(file)
        pooled_proof[phase] = checksum(warm)
    rows = []
    for k in sorted(set(a[0])&set(b[0])&set(c[0])):
        target = c[0][k]
        if target['arm']=='threshold':
            continue
        prefix,expert,mix = k.rsplit('/',2)
        view = expert.removesuffix('_tree')
        base = prefix+'/threshold/0'
        auxiliary = [f'{prefix}/{v}_tree/{mix}' for v in CONTROL.get(view,())]
        results = []
        for group in (a,b):
            x = fp(group[0][k],'fp')
            pooled = fp(group[1][k],'fp')
            easier = max(fp(group[0][j],'fp') for j in [base,*auxiliary])
            results.append((x,x-max(pooled,easier),x-pooled))
        stable = min(o[1] for o in results)>=0 and sum(o[1] for o in results)>0
        for rule in ('fixed','fp','tp'):
            dev = delta(target[rule],'cal')
            review = delta(target[rule],'held')
            established = max(0.,delta(c[1][k][rule],'cal'),
                              *[delta(c[0][j][rule],'cal') for j in [base,*auxiliary]])
            rows.append(dict(key=k,view=view,rule=rule,threshold=target[rule]['cut'],
                         panel=target['panel'],width=target['width'],mix=target['mix'],
                         cal_rows=target['cal_rows'],review_rows=target['review_rows'],
                         a_tp=results[0][0],b_tp=results[1][0],a_extra=results[0][1],b_extra=results[1][1],
                         a_vs_pooled=results[0][2],b_vs_pooled=results[1][2],
                         cal_f1=dev,review_f1=review,over_cal=dev-established,projects=stable,
                         eligible=stable and dev>established and review>=0.))
    table = DEST/'comparison.csv'
    with table.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    rank = lambda v:(min(v['cal_f1'],v['review_f1']),v['cal_f1']+v['review_f1'],-v['review_rows'])
    candidates = [r for r in rows if r['eligible']]
    chosen = max(candidates,key=rank,default=None)
    best = {k:max([r for r in rows if r['view']==k],
                  key=lambda r:(min(r['a_extra'],r['b_extra']),r['a_extra']+r['b_extra'],rank(r)))
            for k in KEYS}
    write_json(DEST/'selection.json',dict(chosen=chosen,eligible=len(candidates),checked=len(rows),best=best,
              csv_sha256=checksum(table),sources=proofs,pooled=pooled_proof,code_sha256=checksum(Path(__file__))))
    print('specialists',len(rows),'chosen',chosen,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=(*PARTS,'select'))
    args = p.parse_args()
    select() if args.stage=='select' else main(args.stage)
