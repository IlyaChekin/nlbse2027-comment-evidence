"""Test whether later rule outcomes add evidence beyond the first scanner hit."""
import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np

from data38 import HERE, PARTS, checksum, json_at, write_json, phases, routes, pairs, crop
from rank38 import installed, raw_values, positive_features, predict, fit, cut, per_project, adjusted

DEST = HERE/'fullscan'
VIEWS = ('point','pair','all-point','all-pair','joint-point','joint-pair')
BANDS = (.9,1.8,3.8,None)
MIX = (.25,.55,.85)


def evidence(x, mode, raw, aux):
    if not mode.startswith(('all-','joint-')):
        return x
    groups = np.log1p(raw.clip(0,64))
    present = raw>0
    unavailable = raw<0
    total = present.sum(1)
    extra = np.column_stack((groups,present,total,unavailable.sum(1)))
    if mode.startswith('joint-'):
        # Match the original 19 WeakSATD rule columns to their compact predictions.
        expected = aux[:,1:20]
        hit_groups = present[:,:19]
        agrees = expected*hit_groups
        extra = np.column_stack((extra,agrees,agrees.sum(1),agrees.max(1),
                                  (expected*(~hit_groups)).sum(1)))
    return np.column_stack((x,extra))


def score_views(rotation,data,t):
    train = data['oof']
    dev = data['dev']
    ext = data.get('test',dev)
    a,b,_ = pairs(rotation,train,'positive_retained')
    use = routes(train)['positive']
    ix = np.flatnonzero(use)
    pos,neg = np.searchsorted(ix,a),np.searchsorted(ix,b)
    assert np.array_equal(ix[pos],a) and np.array_equal(ix[neg],b)
    banks = []
    for case in (crop(train,use),crop(dev,dev['cal']),crop(ext,ext['held'])):
        active = routes(case)['positive']
        if len(banks)==0:
            assert active.all()
        local = crop(case,active)
        raw = raw_values(local['ids'])
        features = positive_features(local,t,local['aux_positive'],local['first'],True)
        banks.append((case,active,local,raw,features))
    models,scores = {},{}
    label = banks[0][2]
    for arm in VIEWS:
        x = evidence(banks[0][-1],arm,banks[0][-2],label['aux_positive'])
        pair = (pos,neg) if arm.endswith('pair') else (pos[:0],neg[:0])
        model = fit(x,label['latest'],label['y'],*pair)
        values = []
        for case,active,subset,raw,features in banks[1:]:
            x = evidence(features,arm,raw,subset['aux_positive'])
            z = np.zeros(len(active))
            z[active] = predict(model,x,case['signed'][active])
            assert np.isfinite(z).all()
            values.append(z)
        models[arm] = model
        scores[arm] = values
    return models,scores,[x[0] for x in banks[1:]]


def compare(rotation,case,pred):
    results={}
    saved=dict(ids=case[1]['ids'],truth=case[1]['y'],baseline=case[1]['signed']>0.)
    masks = [routes(x)['positive'] for x in case]
    truth = [x['y'] for x in case]
    base = [x['signed']>0. for x in case]
    for width,mix,arm in product(BANDS,MIX,VIEWS):
        use = [a if width is None else a&(abs(x['signed'])<width) for x,a in zip(case,masks,strict=True)]
        rank = [(1.-mix)*x['signed'][take]+mix*v[take] for x,take,v in zip(case,use,pred[arm],strict=True)]
        if not all(p.any() for p in use):
            continue
        key=f'{width}/{mix}/{arm}'
        row=dict(width=width,mix=mix,arm=arm,rows=[int(x.sum()) for x in use])
        if rotation!='official':
            row['matched'],bits=per_project(truth[1],case[1]['project'],base[1],use[1],rank[1],('fp','tp'))
            saved[key+'_matched']=bits['fp']
        for rule in ('fixed','fp','tp'):
            t = cut(truth[0][use[0]],base[0][use[0]],rank[0],rule)
            c,_ = adjusted(truth[0],base[0],use[0],rank[0],t)
            q,bits = adjusted(truth[1],base[1],use[1],rank[1],t)
            row[rule] = dict(cal=c,held=q,cut=t)
            saved[key+'_'+rule] = bits
        results[key]=row
    return results,saved


def main(rotation):
    data,ref,base = phases(rotation)
    sources = installed(rotation,data,base)
    models,rank,roles = score_views(rotation,data,ref['cutoff'])
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    results, saved = compare(rotation,roles,rank)
    file = out/'predictions.npz'
    np.savez_compressed(file,**saved)
    weight = out/'models.joblib'
    joblib.dump(models,weight,compress=3)
    write_json(out/'results.json',dict(rotation=rotation,sources=sources,baseline_sha256=base,
              options=results,weights_sha256=checksum(weight),predictions_sha256=checksum(file),
              code_sha256=checksum(Path(__file__))))
    print(rotation,len(results),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
