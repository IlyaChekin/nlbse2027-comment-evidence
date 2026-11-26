"""Compare identical readouts and distill file-disjoint teacher scores."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
import torch

from views46 import HERE,ROOT,PARTS,f,checksum,json_at,load,parts,write_json,negative,DEST as WINDOWS
from extract46 import DEST as FEATURES
from source41 import crop
from heads41 import set_fit,predict
from project_split import fingerprint
from matched_veto33 import evaluation
sys.path[:0] = [str(ROOT/'round44'),str(ROOT/'round45')]
from study44 import pairs
from policy45 import common

DEST = HERE/'fitted'
BANK = {'unix96':('unix_96','features.npy'),'unix384':('unix_384','features.npy'),
        'qwen96':('qwen_96','features.npy'),'qwen384':('qwen_384','features.npy'),
        'qwen12':('qwen_96','layer12.npy')}
PHASES = ('oof','cal','held')


def prepare(rotation,stamp):
    details = json_at(WINDOWS/'receipt.json')
    assert details['matrix'][rotation]==stamp
    assert details['roster_sha256']==checksum(WINDOWS/'rows.npz')
    ids = np.load(WINDOWS/'rows.npz')['ids']
    archives,cost = {},{}
    for name,(home,key) in BANK.items():
        p = FEATURES/home
        proof = json_at(p/'receipt.json')
        assert proof['code_sha256']==checksum(HERE/'extract46.py')
        assert proof['window_receipt_sha256']==checksum(WINDOWS/'receipt.json')
        target = p/key
        assert proof['features'][key]==checksum(target)
        features = np.load(target,mmap_mode='r')
        assert features.shape==(len(ids),proof['dimensions'])
        archives[name] = features
        cost[name] = checksum(p/'receipt.json')
    return ids,archives,cost


def learn(x,frames,permitted,soft=None,cross=True):
    fit = frames[0]
    rank = torch.as_tensor(fit['signed'].clip(-12.,12.),device='cuda',dtype=torch.float32)
    y = torch.as_tensor(fit['y'],device='cuda',dtype=torch.float32)
    signal = torch.as_tensor(soft,device='cuda',dtype=torch.float32) if soft is not None else None
    xd = torch.as_tensor(x[0],device='cuda',dtype=torch.float32)
    oof = None
    if cross:
        fold = fit['fold']
        oof = np.full(len(fold),np.nan,np.float32)
        for n in np.unique(fold):
            allowed = permitted & ((fit['duplicates']&(1<<int(n)))==0)
            held = fold==n
            assert not (allowed&held).any()
            estimator = set_fit(xd,y,rank,allowed,salt=int(n))
            oof[held] = predict(x[0][held],fit['signed'][held].clip(-12.,12.),estimator)
        assert np.isfinite(oof).all()
    spec = set_fit(xd,y,rank,permitted,teacher=signal)
    score = [predict(z,frame['signed'].clip(-12.,12.),spec) for z,frame in zip(x[1:],frames[1:],strict=True)]
    return spec,(oof,*score)


def point_matrix(frames,t):
    messages = []
    for r in frames:
        design,branch = common(r,t)
        assert not branch.any()
        messages.append(design.astype(np.float32))
    return messages


def join(point,values):
    return [np.column_stack((x,z)).astype(np.float32) for x,z in zip(point,values,strict=True)]


def compact(rotation,frames,source,stamp):
    root = ROOT/'round41/compact'/rotation
    ref = root/'receipt.json'
    receipt = json_at(ref)
    states = root/'states.npz'
    assert receipt['matrix_sha256']==stamp and receipt['state_sha256']==checksum(states)
    assert receipt['code_sha256']==checksum(ROOT/'round41/compact41.py')
    steps = ('oof','dev','test' if 'test' in source else 'dev')
    found = []
    with np.load(states) as values:
        for frame,phase in zip(frames,steps,strict=True):
            ids = values[phase+'_ids']
            resolver = {key:i for i,key in enumerate(ids)}
            ix = np.asarray([resolver[k] for k in frame['ids']])
            assert np.array_equal(ids[ix],frame['ids'])
            if phase=='oof':
                assert np.array_equal(values['fold'][ix],frame['fold'])
            found.append(values[phase][ix])
    return found,checksum(ref)


def inspect(probed,frame,pair_groups,corp):
    results = {}
    for kind,(a,b) in pair_groups.items():
        if not len(a):
            continue
        group = [(corp[2][i],corp[3][i]) for i in frame['ids'][a]]
        out = {'reference':evaluation(frame['signed'],a,b,group)}
        miss = {}
        raw = {}
        for name,source in probed.items():
            if source[0] is None:
                continue
            rank = source[0]
            out[name] = evaluation(rank,a,b,group)
            raw[name] = rank[a]>rank[b]
        for q,u in (('qwen96','unix96'),('qwen384','unix384'),('qwen12','unix96')):
            x,y = raw[q],raw[u]
            miss[q] = dict(improved_over_unix=int((x&~y).sum()),
                            lost_vs_unix=int((~x&y).sum()),both_correct=int((x&y).sum()))
        results[kind] = dict(pairs=len(a),ranks=out,complement=miss)
    return results


def main(rotation):
    start = perf_counter()
    source,t,stamp = load(rotation)
    phase = (source['oof'],*parts(source))
    frames = [crop(z,negative(z)) for z in phase]
    train = frames[0]
    prop = source.get('test',source['dev'])
    texts,_,family,_,_ = corp = f.inputs()
    val = np.unique(np.r_[source['dev']['ids'],prop['ids']])
    assert not np.isin(train['ids'],val).any()
    if 'test' in source:
        assert not set(family[train['ids']])&set(family[prop['ids']])
    copies = {fingerprint(texts[i]) for i in val}
    permitted = np.fromiter((fingerprint(texts[i]) not in copies for i in train['ids']),bool)
    pair_groups,pair_hash = pairs(rotation,train,permitted,corp)
    index,probe,certs = prepare(rotation,stamp)
    roster = [np.searchsorted(index,z['ids']) for z in frames]
    assert all(np.array_equal(index[p],z['ids']) for p,z in zip(roster,frames,strict=True))
    point = point_matrix(frames,t)
    models,estimates = {},{}
    models['numeric'],estimates['numeric'] = learn(point,frames,permitted)
    for name,archive in probe.items():
        x = join(point,[archive[p] for p in roster])
        models[name],estimates[name] = learn(x,frames,permitted)
        print(rotation,name,round(perf_counter()-start,2),flush=True)
        del x
    for cap in (96,384):
        estimates['joint'+str(cap)] = tuple(.5*(u+q) for u,q in zip(estimates['unix'+str(cap)],
                                                                  estimates['qwen'+str(cap)],strict=True))
    compact_states,compact_stamp = compact(rotation,frames,source,stamp)
    controls = join(point,compact_states)
    models['compact'],estimates['compact'] = learn(controls,frames,permitted,cross=False)
    for donor in ('unix96','unix384','qwen96','qwen384','qwen12'):
        key = 'transfer_'+donor
        models[key],estimates[key] = learn(controls,frames,permitted,soft=estimates[donor][0],cross=False)
    audit = inspect(estimates,train,pair_groups,corp)
    print(rotation,'matched pairs',
          {k:{n:round(v['pair_win'],5) for n,v in d['ranks'].items()} for k,d in audit.items()},flush=True)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    result = {phase+'_ids':r['ids'] for phase,r in zip(PHASES,frames,strict=True)}
    for name,details in estimates.items():
        result.update({phase+'_'+name:z for phase,z in zip(PHASES,details,strict=True) if z is not None})
    np.savez_compressed(out/'scores.npz',**result,fold=train['fold'],permitted=permitted)
    joblib.dump(models,out/'weights.joblib',compress=3)
    write_json(out/'receipt.json',dict(matrix_sha256=stamp,source_features=certs,compact_sha256=compact_stamp,
               fit_rows=len(permitted),copies_excluded=int((~permitted).sum()),pair_sha256=pair_hash,
               diagnostics=audit,seconds=perf_counter()-start,model_sha256=checksum(out/'weights.joblib'),
               predictions_sha256=checksum(out/'scores.npz'),code_sha256=checksum(Path(__file__))))
    print(rotation,'done',round(perf_counter()-start,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
