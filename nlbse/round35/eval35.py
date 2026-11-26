"""Measure matching gains on top of the exact Round 34 operating point."""
import argparse
from itertools import product

import joblib
import numpy as np

from teacher35 import HERE, ROOT, PARTS, DEST as TRAINED, VIEWS, ARMS, f, checksum, json_at, write_json
from eval34 import loaded, signatures, crop, point_classifier, adjusted, cut, per_project
from compare34 import design
from matched_veto33 import evaluation as paired_metric

DEST=HERE/'comparison'
R34=ROOT/'round34'
WIDTHS=(.4,.75,1.25,1.8,None)
MIX=(.3,.6,.9)


def reference(rotation,data,ref):
    selection=json_at(R34/'selection_approx.json')
    cfg=selection['selected']
    phase=data['dev']
    tails=phase if rotation=='official' else data['test']
    old=R34/cfg['stage']/rotation
    table=old/'results.json'
    assert checksum(table)==selection['receipts'][cfg['stage']][rotation]
    proof=json_at(table)
    scope=R34/'approximation'/rotation
    header=json_at(scope/'receipt.json')
    ck=scope/'scores.npz'
    assert proof['heads_sha256']==checksum(scope/'receipt.json')
    assert header['predictions_sha256']==checksum(ck)
    p=old/'calibrators.joblib'
    assert checksum(p)==proof['weights_sha256']
    model=joblib.load(p)[cfg['mode'],cfg['source']]
    cut_=proof['options'][cfg['key']]['fixed']['cut']
    with np.load(ck) as z:
        items=(('dev',phase),) if tails is phase else (('dev',phase),('test',tails))
        for name,rows in items:
            inside=rows['support']
            assert np.array_equal(rows['ids'][inside],z[name+'_ids'])
            q=rows['queried'].copy()
            valid=inside & (abs(q)<cfg['width'])
            baseline=rows['replay'].copy()
            extra=z[name+'_'+cfg['mode']][valid[inside]]
            fit=design(crop(rows,valid),ref['cutoff'],extra)
            q[valid]=(1.-cfg['strength'])*q[valid]+cfg['strength']*model.decision_function(fit)-cut_
            baseline[valid]=q[valid]
            rows['latest']=q
            rows['signed']=baseline
    receipts=old/'predictions.npz'
    assert checksum(receipts)==proof['predictions_sha256']
    with np.load(receipts) as x:
        assert np.array_equal(x['ids'],tails['ids'][tails['held']])
        assert np.array_equal(x[cfg['key']+'_fixed'],tails['signed'][tails['held']]>0.)
    data['oof']['latest']=data['oof']['queried']
    return cfg,checksum(table)


def load_all(rotation):
    data,ref,old=loaded(rotation)
    config,baseline=reference(rotation,data,ref)
    folder=TRAINED/rotation
    info=json_at(folder/'receipt.json')
    assert checksum(HERE/'teacher35.py')==info['code_sha256']
    assert old['predictions_sha256']==info['prior_views_sha256']
    p=folder/'scores.npz'
    assert checksum(p)==info['predictions_sha256']
    with np.load(p) as z:
        for phase,rows in data.items():
            inside=rows['support']
            assert np.array_equal(rows['ids'][inside],z[phase+'_ids'])
            for form in VIEWS:
                for arm in ('initial',*ARMS):
                    key=f'{form}_{arm}'
                    x=np.zeros(len(inside),np.float32)
                    x[inside]=z[f'{phase}_{key}']
                    rows[key]=x
    return data,ref,baseline,info


def paired_report(rotation,bank):
    a,b=signatures(rotation,bank)
    _,_,family,files,_=f.inputs()
    group=[(family[id],files[id]) for id in bank['ids'][a]]
    variants=('queried',*[f'{k}_{v}' for k,v in product(VIEWS,('initial',*ARMS))])
    metrics={k:paired_metric(bank[k],a,b,group) for k in variants}
    for form in VIEWS:
        p,q=[bank[form+'_'+arm] for arm in ('point','pair')]
        prior=(p[a]>p[b]).astype(float)+.5*(p[a]==p[b])
        proposed=(q[a]>q[b]).astype(float)+.5*(q[a]==q[b])
        _,owner=np.unique(a,return_inverse=True)
        per=(np.bincount(owner,weights=proposed-prior)/np.bincount(owner))
        units={}
        for row,file in zip(owner,group,strict=True):
            units.setdefault(file,set()).add(int(row))
        arr=np.asarray([(per[list(values)].sum(),len(values)) for values in units.values()])
        rng=np.random.default_rng(2026)
        z=arr[rng.integers(0,len(arr),size=(1500,len(arr)))].sum(1)
        metrics[form+'_pair_minus_point']=dict(delta=float(per.mean()),
                      interval=np.quantile(z[:,0]/z[:,1],[.025,.975]).tolist())
    return metrics


def features(data,t,mode):
    low=design(data,t)
    if mode=='compact':
        return low
    form,method=mode.rsplit('_',1)
    q=data[mode].astype(np.float64)
    abstain=data[form+'_initial'].astype(np.float64)
    return np.column_stack((low,q)) if method=='initial' else np.column_stack((low,q,abstain,q-abstain))


def comparisons(rotation, cal, test, scores):
    a,b=cal['y'],test['y']
    old_a,old_b=cal['signed']>0.,test['signed']>0.
    result={}
    forecasts=dict(ids=test['ids'],base=old_b,truth=b,available=test['support'])
    for width in WIDTHS:
        u,v=cal['support'].copy(),test['support'].copy()
        if width is not None:
            u &= abs(cal['latest'])<width
            v &= abs(test['latest'])<width
        name='all' if width is None else f'{width:g}'
        for source,mode,weight in [('control','threshold',0.)]+[(k,m,w) for (k,m),w in product(scores,MIX)]:
            p,q=cal['latest'][u],test['latest'][v]
            if weight:
                t0,t1=scores[source,mode]
                p=(1.-weight)*p+weight*t0[u]
                q=(1.-weight)*q+weight*t1[v]
            key=f'{name}_{source}_{mode}_{weight:g}'
            state=dict(source=source,mode=mode,width=width,strength=weight,cal_rows=int(u.sum()),held_rows=int(v.sum()))
            if rotation!='official':
                state['matched'],bit=per_project(b,test['project'],old_b,v,q,('fp','tp'))
                for m,rows in bit.items():
                    forecasts[key+'_matched_'+m]=rows
            for scheme in ('fixed','tp','fp'):
                offset=cut(a[u],old_a[u],p,scheme)
                c,_=adjusted(a,old_a,u,p,offset)
                h,z=adjusted(b,old_b,v,q,offset)
                state[scheme]=dict(cut=offset,cal=c,held=h)
                forecasts[key+'_'+scheme]=z
            result[key]=state
    return result,forecasts


def evaluate(rotation):
    data,ref,old,pinned=load_all(rotation)
    d=data['dev']
    outward=data.get('test',d)
    cal,test=crop(d,d['cal']),crop(outward,outward['held'])
    modes=('compact',*[f'{f0}_{scheme}' for f0,scheme in product(VIEWS,('initial',*ARMS))])
    bank={}
    scores={}
    t=ref['cutoff']
    for family,fit in [('fit',crop(d,d['fit'])),('oof',data['oof'])]:
        for name in modes:
            x=features(fit,t,name)
            estimator=point_classifier(x,fit['y'],fit['latest'],fit['support'])
            bank[family,name]=estimator
            scores[family,name]=[estimator.decision_function(features(z,t,name)) for z in (cal,test)]
    result,forecasts=comparisons(rotation,cal,test,scores)
    out=DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    file=out/'predictions.npz'
    np.savez_compressed(file,**forecasts)
    book=out/'readouts.joblib'
    joblib.dump(bank,book,compress=3)
    report=dict(rotation=rotation,pairs=paired_report(rotation,data['oof']),options=result,
                baseline_sha256=old,teacher_sha256=pinned['predictions_sha256'],
                code_sha256=checksum(HERE/'eval35.py'),predictions_sha256=checksum(file),
                weights_sha256=checksum(book))
    write_json(out/'results.json',report)
    print(rotation,'same-file comparisons',report['pairs'],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    evaluate(cli.parse_args().rotation)
