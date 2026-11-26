"""Test directional correction against abstention with calibration-only cutoffs."""
import csv
from itertools import product
from pathlib import Path

import numpy as np

from study50 import HERE, DEST as MODELS, PARTS, VIEWS, f, checksum, json_at, load, write_json
from source41 import parts

DEST=HERE/'selective'
HEADS=('conditional','linear_conditional','paired')
PRECISION=(.70,.85,.95)
MIN_CAL=5


def cutoff(utility, good, precision):
    if len(utility)<MIN_CAL:
        return 1.
    order=np.argsort(-utility,kind='stable')
    sorted_=utility[order]
    ends=np.flatnonzero(np.r_[sorted_[:-1]!=sorted_[1:],True])
    done=np.cumsum(good[order])[ends]
    valid=(ends+1>=MIN_CAL)&(done/(ends+1)>=precision)
    if not valid.any():
        return 1.
    positions=np.flatnonzero(valid)
    # Prefer the most good changes and then the smaller tie group.
    index=positions[np.argmax(done[positions])]
    return float(np.nextafter(sorted_[ends[index]],-np.inf))


def materialize(rotation):
    source,_,stamp=load(rotation)
    cal,held=parts(source)
    home=MODELS/rotation
    info=json_at(home/'results.json')
    loc=home/'annotations.npz'
    file=home/'predictions.npz'
    assert info['proof']['matrix_sha256']==stamp and checksum(loc)==info['annotation_sha256']
    assert checksum(file)==info['predictions_sha256']
    with np.load(loc) as n:
        forecast=dict(n)
    with np.load(file) as z:
        assert np.array_equal(z['ids'],held['ids'])
        assert np.array_equal(cal['ids'],z['cal_ids'])
    frames=(cal,held)
    masks=[r['old']&(r['flag']>=0) for r in frames]
    for tag,r,m in zip(('cal','held'),frames,masks,strict=True):
        assert np.array_equal(forecast[tag+'_ids'],r['ids'][m])
    return frames,masks,forecast,stamp


def main():
    DEST.mkdir(exist_ok=True)
    summary={}
    records=[]
    for rotation in PARTS:
        frames,masks,scores,stamp=materialize(rotation)
        a,b=frames
        old=[r['signed']>0. for r in frames]
        truth=[r['y'] for r in frames]
        raw_flags=[r['flag'][m] for r,m in zip(frames,masks,strict=True)]
        payload=dict(ids=b['ids'],base=old[1],truth=truth[1])
        cases={}
        for view,head,precision,rule in product(VIEWS,HEADS,PRECISION,('single','agree')):
            # Both policies compute thresholds without predictor access to archival W at inference.
            m1=[scores[k+'_'+view+'/'+head] for k in ('cal','held')]
            m2=[scores[k+'_numeric/conditional'] for k in ('cal','held')]
            priority=[]
            opportunity=[]
            for pre,w1,w2,flag,active in zip(old,m1,m2,raw_flags,masks,strict=True):
                risks=[np.where(flag==1,1.-v,v) for v in (w1,w2)]
                p=risks[0] if rule=='single' else np.minimum(risks[0],risks[1])
                applicable=pre[active]==flag
                priority.append(p)
                opportunity.append(applicable)
            edges={}
            outcome=[]
            for direction in (0,1):
                m=opportunity[0]&(raw_flags[0]==direction)
                correct=truth[0][masks[0]][m]!=(direction==1)
                edges[str(direction)]=cutoff(priority[0][m],correct,precision)
            for i,m in enumerate(masks):
                changed=opportunity[i] & (priority[i]>np.where(raw_flags[i]==0,edges['0'],edges['1']))
                pred=old[i].copy()
                pred[m]=np.where(changed,~pred[m],pred[m])
                v=f.comparison(truth[i],old[i],pred)
                outcome.append(v|dict(changes=int(changed.sum())))
            tag=f'{view}/{head}/{precision:g}/{rule}'
            for role,z in zip(('cal','held'),outcome,strict=True):
                records.append(dict(rotation=rotation,tag=tag,role=role,tp=z['delta_tp'],fp=z['delta_fp'],
                                    changes=z['changes'],f1=z['f1_after']-z['f1_before']))
            payload[tag]=pred
            cases[tag]=dict(cutoff=edges,calibration=outcome[0],held=outcome[1])
        folder=DEST/rotation
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'predictions.npz',**payload)
        summary[rotation]=dict(matrix_sha256=stamp,model_sha256=checksum(MODELS/rotation/'results.json'),
                            predictions_sha256=checksum(folder/'predictions.npz'),cases=cases)
    table=DEST/'comparison.csv'
    with table.open('w',newline='') as t:
        w=csv.DictWriter(t,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    A,B,O=(summary[r]['cases'] for r in PARTS)
    selections=[]
    def gain(h):
        return h['f1_after']-h['f1_before']
    for k,v in O.items():
        a,b=A[k]['held'],B[k]['held']
        c,d=v['calibration'],v['held']
        if (gain(a)>=0. and gain(b)>=0. and gain(c)>0. and gain(d)>0.
                and a['delta_fp']<=0 and b['delta_fp']<=0 and d['delta_fp']<=0):
            selections.append(dict(key=k,a=a,b=b,cal=c,review=d,cut=v['cutoff']))
    selections.sort(key=lambda e:min(gain(e['cal']),gain(e['review'])),reverse=True)
    best=min(records,key=lambda r:-r['f1'] if r['rotation']=='official' and r['role']=='held' else 1.)
    report=dict(selected=selections[0] if selections else None,qualified=len(selections),best_official_review=best,
                comparisons=len(O),code_sha256=checksum(Path(__file__)),table_sha256=checksum(table),
                evidence=summary)
    write_json(DEST/'selection.json',report)
    print({k:v for k,v in report.items() if k!='evidence'},flush=True)


if __name__=='__main__':
    main()
