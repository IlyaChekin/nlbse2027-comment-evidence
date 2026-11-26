"""Test contrastive readouts, matched retrieval and compact distillation."""
import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch

from embed47 import HERE,ROOT,DEST as EMB,WINDOWS,f,checksum,json_at,write_json,CAPS
from fit46 import (PARTS,PHASES,DEST as CONTROL,load,parts,negative,crop,point_matrix,join,
                   fingerprint,pairs,learn,compact,evaluation)
from torch.nn.functional import normalize

DEST=HERE/'fitted'
VIEWS=('embed96','embed384','raw96','raw384')
NEIGHBORS=5


def arrays(stamp,rotation):
    previous=json_at(WINDOWS/'receipt.json')
    stamp_file=EMB/'views.json'
    manifest=json_at(stamp_file)
    assert previous['matrix'][rotation]==stamp
    assert manifest['code_sha256']==checksum(HERE/'embed47.py')
    assert manifest['window_receipt']==checksum(WINDOWS/'receipt.json')
    assert previous['roster_sha256']==manifest['roster_sha256']==checksum(WINDOWS/'rows.npz')
    ids=np.load(WINDOWS/'rows.npz')['ids']
    out,records={},{}
    for cap in CAPS:
        folder=EMB/str(cap)
        card=json_at(folder/'receipt.json')
        assert card['views_sha256']==checksum(stamp_file)
        assert card['code_sha256']==manifest['code_sha256']
        for name,file in (('embed','projection.npy'),('raw','raw.npy')):
            p=folder/file
            assert card['hashes'][file]==checksum(p)
            val=np.load(p,mmap_mode='r')
            assert len(val)==len(ids)==card['rows']
            out[name+str(cap)]=val
        records[str(cap)]=checksum(folder/'receipt.json')
    return ids,out,records


def old_scores(rotation,stamp,phases):
    d=CONTROL/rotation
    card=json_at(d/'receipt.json')
    assert card['matrix_sha256']==stamp and card['predictions_sha256']==checksum(d/'scores.npz')
    with np.load(d/'scores.npz') as table:
        for phase,r in zip(PHASES,phases,strict=True):
            assert np.array_equal(table[phase+'_ids'],r['ids'])
        names=('numeric','compact','unix96','unix384','qwen96','qwen384','transfer_unix96','transfer_unix384')
        out={k:tuple(table[phase+'_'+k] if phase+'_'+k in table else None for phase in PHASES) for k in names}
    return out,checksum(d/'receipt.json')


def bank(x,a,b,eligible,seed):
    in_pair=eligible[a]&eligible[b]
    rng=np.random.default_rng(seed)
    p,n=np.unique(a[in_pair]),np.unique(b[in_pair])
    width=min(len(p),len(n))
    assert width>=NEIGHBORS
    p=rng.choice(p,size=width,replace=False)
    n=rng.choice(n,size=width,replace=False)
    return x[p],x[n]


def intensify(x,table):
    record=[]
    for batch in x.split(512):
        hi=[(batch@reference.T).topk(NEIGHBORS).values.float() for reference in table]
        positive,negative=hi
        difference=.5*((positive[:,0]-negative[:,0])+(positive.mean(1)-negative.mean(1)))
        record.append(difference.cpu().numpy())
    return np.concatenate(record) if record else np.zeros(0,np.float32)


@torch.inference_mode()
def retrieval(dataset,frames,permitted,pair_groups):
    representations=[normalize(torch.tensor(a,device='cuda',dtype=torch.float32),dim=1).half()
                     for a in dataset]
    train=frames[0]
    choices=tuple(r['unqueried']>0. for r in frames)
    out=[np.empty(len(r['ids']),np.float32) for r in frames]
    repeats={}
    for yes,label in ((False,'negative_consensus'),(True,'negative_veto')):
        a,b=pair_groups[label]
        assert (choices[0][a]==yes).all() and (choices[0][b]==yes).all()
        in_scope=choices[0]==yes
        for fold in np.unique(train['fold']):
            allow=permitted&in_scope&((train['duplicates']&(1<<int(fold)))==0)
            scope=(train['fold']==fold)&in_scope
            x=bank(representations[0],a,b,allow,4700+int(fold))
            out[0][scope]=intensify(representations[0][scope],x)
        x=bank(representations[0],a,b,permitted&in_scope,4700)
        repeats[label]=len(x[0])
        for v,row,p in zip(representations[1:],choices[1:],out[1:],strict=True):
            scope=row==yes
            p[scope]=intensify(v[scope],x)
    assert all(np.isfinite(p).all() for p in out)
    return tuple(frame['signed']+6.*value for frame,value in zip(frames,out,strict=True)),repeats


def diagnostics(views,frame,groups,corp):
    result={}
    y=frame['signed']
    for k,(a,b) in groups.items():
        if not len(a):
            continue
        paths=[(corp[2][i],corp[3][i]) for i in frame['ids'][a]]
        signal={'reference':y}|{key:pred[0] for key,pred in views.items() if pred[0] is not None}
        stats={name:evaluation(val,a,b,paths) for name,val in signal.items()}
        first=signal['embed384']
        old=signal['unix384']
        found,lost=(first[a]>first[b]),(old[a]>old[b])
        result[k]=dict(pairs=len(a),stats=stats,unique_embed=int((found&~lost).sum()),unique_unix=int((~found&lost).sum()))
    return result


def run(rotation):
    start=perf_counter()
    source,t,stamp=load(rotation)
    rows=[crop(z,negative(z)) for z in (source['oof'],*parts(source))]
    train=rows[0]
    val=np.unique(np.r_[source['dev']['ids'],source.get('test',source['dev'])['ids']])
    corpus=f.inputs()
    assert not np.isin(train['ids'],val).any()
    if 'test' in source:
        assert not set(corpus[2][train['ids']])&set(corpus[2][source['test']['ids']])
    copies={fingerprint(corpus[0][i]) for i in val}
    permitted=np.fromiter((fingerprint(corpus[0][i]) not in copies for i in train['ids']),bool)
    groups,match=pairs(rotation,train,permitted,corpus)
    ids,representations,proof=arrays(stamp,rotation)
    take=[np.searchsorted(ids,r['ids']) for r in rows]
    assert all(np.array_equal(ids[ix],r['ids']) for ix,r in zip(take,rows,strict=True))
    output,control=old_scores(rotation,stamp,rows)
    saved_fit=np.load(CONTROL/rotation/'scores.npz')
    assert np.array_equal(saved_fit['permitted'],permitted) and np.array_equal(saved_fit['fold'],train['fold'])
    point=point_matrix(rows,t)
    models={}
    for arm,z in representations.items():
        data=join(point,[z[ix] for ix in take])
        models[arm],output[arm]=learn(data,rows,permitted)
        print(rotation,arm,round(perf_counter()-start,2),flush=True)
    bank_sizes={}
    for tag in ('embed96','embed384','raw384'):
        sequence=[representations[tag][i] for i in take]
        key='knn_'+tag
        output[key],bank_sizes[key]=retrieval(sequence,rows,permitted,groups)
    output['merge384']=tuple((x+y)*.5 for x,y in zip(output['embed384'],output['unix384'],strict=True))
    latent,stem=compact(rotation,rows,source,stamp)
    base=join(point,latent)
    for donor in ('embed96','embed384','knn_embed384'):
        key='transfer_'+donor
        models[key],output[key]=learn(base,rows,permitted,soft=output[donor][0],cross=False)
    report=diagnostics(output,train,groups,corpus)
    where=DEST/rotation
    where.mkdir(exist_ok=True,parents=True)
    payload={phase+'_ids':z['ids'] for phase,z in zip(PHASES,rows,strict=True)}
    for kind,forecast in output.items():
        payload.update({phase+'_'+kind:p for phase,p in zip(PHASES,forecast,strict=True) if p is not None})
    np.savez_compressed(where/'scores.npz',**payload,fold=train['fold'],permitted=permitted)
    joblib.dump(models,where/'weights.joblib',compress=3)
    stamp=dict(matrix_sha256=stamp,embeddings=proof,prior_sha256=control,pairs_sha256=match,
               compact_sha256=stem,excluded=int((~permitted).sum()),bank_sizes=bank_sizes,
               score_sha256=checksum(where/'scores.npz'),weights_sha256=checksum(where/'weights.joblib'),
               diagnostics=report,study_seconds=perf_counter()-start,code_sha256=checksum(Path(__file__)))
    write_json(where/'receipt.json',stamp)
    print(rotation,round(perf_counter()-start,2),bank_sizes,{k:[
        round(v['stats'][n]['pair_win'],5) for n in ('reference','unix384','embed384','knn_embed384')]
        for k,v in report.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    run(p.parse_args().rotation)
