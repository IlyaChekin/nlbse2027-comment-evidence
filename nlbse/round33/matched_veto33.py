"""Build matched error pairs among negative scanner decisions on outside-file forecasts."""
from collections import defaultdict

import numpy as np

from points33 import HERE, ROOT, PARTS, OUT, f, checksum, json_at, write_json
from experiment33 import read_phase

DEST=HERE/'matched_veto'


def make_pairs(data, corpus, project, files):
    group=defaultdict(list)
    failures=[]
    possible=(data['flag']==0)&data['old']&(data['unqueried']>0.)&(data['queried']<=0.)
    def file_for(i):
        row=data['ids'][i]
        return project[row],files[row]
    for i in np.flatnonzero(possible):
        if data['y'][i]:
            failures.append(i)
        else:
            group[file_for(i)].append(i)
    pos,neg,tag=[],[],[]
    ranks=np.column_stack((np.log1p(data['length'])/.65,
                           data['unqueried']/.6,data['queried']/.6))
    for i in failures:
        file=file_for(i)
        pool=np.asarray(group[file],dtype=np.int64)
        if not len(pool):
            continue
        assert (data['fold'][pool]==data['fold'][i]).all()
        pool=np.asarray([j for j in pool if corpus[data['ids'][i]]!=corpus[data['ids'][j]]])
        if not len(pool):
            continue
        d=np.linalg.norm(ranks[pool]-ranks[i],axis=1)
        best=pool[np.lexsort((data['ids'][pool],d))[:3]]
        pos.extend([i]*len(best))
        neg.extend(best)
        tag.extend([file]*len(best))
    return np.asarray(pos,dtype=np.int64),np.asarray(neg,dtype=np.int64),len(failures),tag


def evaluation(values,a,b,groups):
    equal=(values[a]==values[b])*.5
    good=(values[a]>values[b])+equal
    ids,local=np.unique(a,return_inverse=True)
    means=np.bincount(local,weights=good)/np.bincount(local)
    scores={}
    for i,group in zip(local,groups,strict=True):
        scores.setdefault(group,set()).add(int(i))
    units=list(scores.values())
    data=np.asarray([(sum(means[list(ix)]),len(ix)) for ix in units])
    rng=np.random.default_rng(42)
    totals=data[rng.integers(0,len(units),size=(1500,len(units)))].sum(1)
    interval=np.quantile(totals[:,0]/totals[:,1],[.025,.975]).tolist()
    return dict(pair_win=float(means.mean()),interval=interval,unique_positive=int(len(ids)))


def main():
    DEST.mkdir(exist_ok=True)
    corpus,y,project,files,_=f.inputs()
    stages={}
    for rotation in PARTS:
        parents=OUT/rotation
        receipt=json_at(parents/'receipt.json')
        data=read_phase('oof',parents,receipt)
        a,b,eligible,tag=make_pairs(data,corpus,project,files)
        teacher=ROOT/'round30/tuned_teacher'/rotation
        meta=json_at(teacher/'fit.json')
        source=teacher/'oof.npz'
        assert checksum(source)==meta['predictions_sha256']
        ids=data['ids']
        assert np.array_equal(y[ids,0],data['y'])
        ix=np.unique(np.r_[a,b])
        with np.load(source) as z:
            chosen=ids[ix]
            received,inside,there=np.intersect1d(chosen,z['ids'],return_indices=True)
            if len(received)!=len(chosen):
                raise ValueError('Teacher does not cover all selected examples')
            row=ix[inside]
            assert np.array_equal(z['fold'][there],data['fold'][row])
            assert np.array_equal(z['y'][there],data['y'][row])
            score=np.full((len(ids),2),np.nan)
            score[row]=np.column_stack((z['raw'][there,2],z['calibrated'][there,4]))
        features=dict(before=data['unqueried'],after=data['queried'],
                      tuned_teacher=score[:,0],teacher_combination=score[:,1])
        path=DEST/f'{rotation}.npz'
        np.savez_compressed(path,positive=ids[a],negative=ids[b],fold=data['fold'][a],
                            **{k:np.column_stack((z[a],z[b])) for k,z in features.items()})
        record=dict(available_false_negatives=eligible,matched_false_negatives=int(len(np.unique(a))),
                    files=len(set(tag)),pairs=len(a),metrics={k:evaluation(z,a,b,tag) for k,z in features.items()},
                    predictions_sha256=checksum(path),points_sha256=checksum(parents/'receipt.json'),
                    teacher_sha256=checksum(source))
        stages[rotation]=record
        print(rotation,record,flush=True)
    write_json(DEST/'results.json',dict(analysis=stages,code_sha256=checksum(HERE/'matched_veto33.py')))


if __name__=='__main__':
    main()
