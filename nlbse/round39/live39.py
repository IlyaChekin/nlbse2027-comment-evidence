"""Benchmark the selected location readout with the exact Round 38 pipeline."""
import argparse
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from location39 import HERE, ROOT, PROGRAM, f, checksum, json_at, write_json, scan
from data39 import R38
from data38 import R37
from study39 import use_for, matrix, ordinary, vectorizer, crop, predict
from live38 import correction
from live35 import forward, heads, load_choice, recipe, packages
from live37 import encode
from deferred37 import ScopedStates
from eval35 import cut
from pipeline import PUBLIC, load


def verify_scanner(texts):
    origin=ROOT/'round32/first_hit_public.npz'
    assert checksum(origin)==json_at(origin.with_suffix('.json'))['predictions_sha256']
    with np.load(origin) as saved:
        ids=saved['ids']
        got=scan([texts[i] for i in ids])
        assert np.array_equal(got[:,:3],saved['results'])
    return int(len(ids)),checksum(origin)


def location_adjust(old, threshold, choice, readout, sparse, texts):
    branch=choice['mode'].split('-')[0]
    def apply(data,rank,first,aux,context):
        baseline,_,paid=old(data,rank,first,aux,context)
        current=data|{'signed':baseline,'latest':baseline,'aux_positive':aux[:,:20],'first':first}
        picked=use_for(current,choice['panel'],choice['width'])
        if not picked.any():
            return baseline,picked,paid
        rows=crop(current,picked)
        shape,local,center=ordinary([texts[i] for i in np.flatnonzero(picked)],rows['location'])
        evidence=None
        if branch in ('local','center'):
            inputs=local if branch=='local' else center
            x=vectorizer().transform(inputs)
            evidence=sparse[branch].decision_function(x)
            paid+=2*x.nnz
        x=matrix(rows,threshold,choice['mode'],shape,evidence)
        z=predict(readout,x,baseline[picked])
        w=choice['mix']
        result=baseline.copy()
        result[picked]=(1.-w)*baseline[picked]+w*z-choice['cut']
        return result,picked,paid+5*x.size
    return apply


def main(arm):
    plan=HERE/'selection.json'
    policy=json_at(plan)
    settings=policy[arm]
    if not settings:
        raise ValueError('No frozen location candidate')
    path=HERE/'comparison/official/results.json'
    assert policy['receipts']['official']==checksum(path)
    heads39=path.with_name('models.joblib')
    assert checksum(heads39)==json_at(path)['model_sha256']
    final=joblib.load(heads39)
    anchor=R38/'public_primary/candidate.npz'
    report=json_at(anchor.with_name('results.json'))
    assert checksum(anchor)==report['comparison']['candidate']['predictions_sha256']
    released=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert checksum(released)==policy['archive_sha256']
    old_r=R38/'comparison/official/results.json'
    old_p=json_at(R38/'selection.json')
    assert checksum(old_r)==old_p['sources']['official']
    old_m=old_r.with_name('models.joblib')
    assert checksum(old_m)==json_at(old_r)['weights_sha256']
    old=joblib.load(old_m)
    frozen=json_at(R37/'selection.json')
    region=frozen['best']['semantic']
    table=R37/'comparison/official/results.json'
    assert checksum(table)==frozen['source_table']['official']
    weight=table.with_name('readouts.joblib')
    assert checksum(weight)==json_at(table)['weights_sha256']
    prev=joblib.load(weight)[region['source'],region['mode']]
    projections=ROOT/'round34/selection_approx.json'
    first=json_at(projections)['selected']
    projection,calibration,_=heads(first,projections)
    text,y,_=load(PUBLIC)
    assert checksum(PUBLIC)==report['receipt']['data_sha256']
    with np.load(anchor) as z:
        assert np.array_equal(y,z['truth'])
        old_bits=z['pred']
    count,verified=verify_scanner(text)
    model=f.pipe('official',f.small.tokenizer())
    core=recipe()
    retained=ScopedStates(core,f.pinned()[0])
    encode_optim=partial(encode,deferred=retained)
    rule,coef,_=load_choice()
    reweight=correction(retained,core[0]['threshold'],old_p['selected'],old)
    setup=(model,core,packages(),rule,coef,first,projection,calibration,region,True,prev,encode_optim,retained.features)

    def run(src,which):
        fn=location_adjust(reweight,core[0]['threshold'],settings,final['readout'][settings['mode']],final['heads'],src) if which else reweight
        result=forward(src,*setup,fn,scan if which else None)
        start=perf_counter()
        cost=result[-1]
        retained.finish(cost)
        seconds=perf_counter()-start
        cost['forward_seconds']+=seconds
        cost['wall_seconds']+=seconds
        return result

    for b in (False,True):
        run(text[:4096],b)
    kept,trials={},dict(reference=[],candidate=[])
    for turn in range(5):
        for b in ((False,True) if turn%2==0 else (True,False)):
            name='candidate' if b else 'reference'
            pred,z,mask,cost=run(text,b)
            if name in kept:
                assert np.array_equal(kept[name][0],pred)
            kept[name]=pred,z,mask
            if b:
                assert np.array_equal(pred[~mask],old_bits[~mask]) and np.array_equal(pred[:,1],old_bits[:,1])
            else:
                assert np.array_equal(pred,old_bits)
            score=f.competition_formula(f.metric(y,pred)['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            trials[name].append(cost|{'score':score})
            print(turn,name,score,cost,flush=True)
    out=HERE/f'public_{arm}'
    out.mkdir(exist_ok=True)
    runs={}
    for name,(pred,margin,mask) in kept.items():
        cost={k:median(x[k] for x in trials[name]) for k in trials[name][0] if k!='score'}
        quality=f.metric(y,pred)
        score=f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        file=out/f'{name}.npz'
        np.savez_compressed(file,pred=pred,truth=y,scope=mask,margin=margin)
        runs[name]=dict(score=score,cost=cost,quality=quality,trials=trials[name],predictions_sha256=checksum(file))
    p,m,t=kept['candidate']
    yn,baseline=y[:,0].astype(bool),old_bits[:,0]
    changes={'frozen':f.comparison(yn,baseline,p[:,0])}
    for method in ('fp','tp'):
        row=baseline.copy()
        offset=cut(yn[t],baseline[t],m[t],method)
        row[t]=m[t]>offset
        changes[method]=f.comparison(yn,baseline,row)
    paired=[b['score']-a['score'] for a,b in zip(trials['reference'],trials['candidate'],strict=True)]
    receipt=dict(selection_sha256=checksum(plan),public_sha256=checksum(PUBLIC),anchor_sha256=checksum(anchor),
                 archive_sha256=checksum(released),extended_sha256=checksum(PROGRAM),scanner_comparisons=count,
                 first_hit_sha256=verified,readout_sha256=checksum(heads39),code_sha256=checksum(Path(__file__)),
                 core_sha256=checksum(ROOT/'round35/live35.py'),context_sha256=checksum(HERE/'context39.py'))
    write_json(out/'results.json',dict(arm=arm,choice=settings,comparison=runs,receipt=receipt,changes=changes,paired=paired))
    print(changes,paired,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arm',choices=('chosen','best_local'))
    main(cli.parse_args().arm)
