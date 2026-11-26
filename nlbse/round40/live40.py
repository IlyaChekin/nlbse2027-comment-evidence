"""Profile a frozen portfolio correction against the full Round 38 release."""
import argparse
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from sparse40 import HERE, ROOT, DEST as SPARSE, f, checksum, json_at, write_json, vector, evaluate
from data40 import outside_scope
from data38 import crop
from stack40 import assemble, aperture
from live38 import correction
from live35 import forward, heads, load_choice, recipe, packages
from live37 import encode
from deferred37 import ScopedStates
from sparse_model import aligned
from eval35 import cut
from pipeline import PUBLIC, load


def experts():
    folder=ROOT/'round24/official'
    source=folder/'token.joblib'
    assert checksum(source)==json_at(folder/'fit.json')['models']['token']['weights_sha256']
    where=SPARSE/'official'
    other=where/'weights.joblib'
    assert checksum(other)==json_at(where/'receipt.json')['weights_sha256']
    return (joblib.load(source),joblib.load(other)),dict(old_weights_sha256=checksum(source),
                                                             new_weights_sha256=checksum(other))


def extend(previous,texts,threshold,spec,bank,legacy,new):
    kind=spec['experts']
    mode=spec['model']
    linear=bank['readout'][kind,mode]
    def apply(rows,rank,first,aux,context):
        baseline,_,debit=previous(rows,rank,first,aux,context)
        data=rows|{'signed':baseline}
        region=outside_scope(data)
        outer=aperture(baseline,np.ones(len(baseline),bool),region,spec['panel'],spec['width'])
        active=np.flatnonzero(outer)
        used=np.zeros(len(baseline),bool)
        if not len(active):
            return baseline,used,debit
        part=crop(data,outer)
        part['sparse']=np.zeros((len(active),2),np.float32)
        part['new']=np.zeros(len(active),np.float32)
        if kind!='score':
            subset=[texts[i] for i in active]
            if kind in ('old','all'):
                value,nnz=aligned(legacy,subset,'token')
                part['sparse']=value
                debit+=nnz*4
            if kind in ('new','all'):
                x=vector(subset)
                part['new']=evaluate(x,new)
                debit+=x.nnz*2
        design,visible,_,_=assemble(part,threshold,kind,bank['calibration'])
        may_edit=np.ones(len(active),bool) if mode=='stack' else visible
        positions=active[may_edit]
        used[positions]=True
        result=baseline.copy()
        if len(positions):
            x=design[may_edit]
            w=spec['mix']
            result[positions]=(1.-w)*baseline[positions]+w*linear.decision_function(x)-spec['cut']
            debit+=4*x.size
        return result,used,debit
    return apply


def anchor():
    previous=ROOT/'round38'
    release=json_at(previous/'release_summary.json')
    archive=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert checksum(archive)==release['archive_sha256']
    profile=previous/'public_primary'
    receipt=json_at(profile/'results.json')
    file=profile/'candidate.npz'
    assert checksum(file)==receipt['comparison']['candidate']['predictions_sha256']
    assert checksum(PUBLIC)==release['public_sha256']
    with np.load(file) as z:
        y,expected=z['truth'],z['pred']
    return archive,file,y,expected


def pipeline():
    model=f.pipe('official',f.small.tokenizer())
    core=recipe()
    runtime=ScopedStates(core,f.pinned()[0])
    encoder=partial(encode,deferred=runtime)
    region39=ROOT/'round37'
    source=region39/'comparison/official/results.json'
    plan=json_at(region39/'selection.json')
    assert checksum(source)==plan['source_table']['official']
    slot=plan['best']['semantic']
    path=source.with_name('readouts.joblib')
    assert checksum(path)==json_at(source)['weights_sha256']
    regional=joblib.load(path)[slot['source'],slot['mode']]
    path=ROOT/'round34/selection_approx.json'
    old=json_at(path)['selected']
    projection,calibrator,_=heads(old,path)
    rule,coef,_=load_choice()
    previous=ROOT/'round38'
    plan=json_at(previous/'selection.json')
    p=previous/'comparison/official/results.json'
    assert checksum(p)==plan['sources']['official']
    weights=p.with_name('models.joblib')
    assert checksum(weights)==json_at(p)['weights_sha256']
    reweight=correction(runtime,core[0]['threshold'],plan['selected'],joblib.load(weights))
    return core,runtime,reweight,(model,core,packages(),rule,coef,old,projection,calibrator,
                       slot,True,regional,encoder,runtime.features)


def profiled(arm):
    policy=HERE/'selection.json'
    plan=json_at(policy)
    selection=plan['picked']['experts']['all'] if arm=='combined' else plan['picked'][arm]
    if selection is None:
        raise ValueError('Empty experiment selection')
    archive,file,y,expected=anchor()
    assert checksum(archive)==plan['archive_sha256']
    data,gold,_=load(PUBLIC)
    assert np.array_equal(y,gold)
    candidate=HERE/'comparison/official/results.json'
    assert checksum(candidate)==plan['source']['official']
    weight=candidate.with_name('models.joblib')
    assert checksum(weight)==json_at(candidate)['weights_sha256']
    loaded=joblib.load(weight)
    (legacy,new),stamp=experts()
    core,runtime,reweight,base=pipeline()

    def run(text,which):
        selected=extend(reweight,text,core[0]['threshold'],selection,loaded,legacy,new) if which else reweight
        result=forward(text,*base,selected)
        start=perf_counter()
        cost=result[-1]
        runtime.finish(cost)
        elapsed=perf_counter()-start
        cost['forward_seconds']+=elapsed
        cost['wall_seconds']+=elapsed
        return result

    for b in (False,True):
        run(data[:4096],b)
    trials=dict(reference=[],candidate=[])
    samples={}
    for n in range(5):
        for b in ((False,True) if n%2==0 else (True,False)):
            name='candidate' if b else 'reference'
            prediction,margin,used,cost=run(data,b)
            if name in samples:
                assert np.array_equal(samples[name][0],prediction)
            samples[name]=prediction,margin,used
            if b:
                assert np.array_equal(prediction[~used],expected[~used])
                assert np.array_equal(prediction[:,1],expected[:,1])
            else:
                assert np.array_equal(prediction,expected)
            q=f.metric(y,prediction)['mean']['f1']
            score=f.competition_formula(q,cost['forward_seconds'],cost['analytic_gflops'])
            trials[name].append(cost|{'score':score})
            print(n,name,score,cost,flush=True)
    dest=HERE/f'public_{arm}'
    dest.mkdir(exist_ok=True)
    compares={}
    for name,(prediction,margin,used) in samples.items():
        runtime_cost={k:median(x[k] for x in trials[name]) for k in trials[name][0] if k!='score'}
        quality=f.metric(y,prediction)
        score=f.competition_formula(quality['mean']['f1'],runtime_cost['forward_seconds'],runtime_cost['analytic_gflops'])
        path=dest/f'{name}.npz'
        np.savez_compressed(path,pred=prediction,truth=y,scope=used,margin=margin)
        compares[name]=dict(score=score,quality=quality,cost=runtime_cost,trials=trials[name],
                            predictions_sha256=checksum(path))
    bits,rank,scope=samples['candidate']
    gold=y[:,0].astype(bool)
    initial=expected[:,0]
    differences={'frozen':f.comparison(gold,initial,bits[:,0])}
    for method in ('fp','tp'):
        offset=cut(gold[scope],initial[scope],rank[scope],method)
        proposed=initial.copy()
        proposed[scope]=rank[scope]>offset
        differences[method]=f.comparison(gold,initial,proposed)
    paired=[b['score']-a['score'] for a,b in zip(trials['reference'],trials['candidate'],strict=True)]
    fingerprints=dict(selection_sha256=checksum(policy),archive_sha256=checksum(archive),
                      public_sha256=checksum(PUBLIC),anchor_sha256=checksum(file),weights_sha256=checksum(weight),
                      code_sha256=checksum(Path(__file__)),runtime_sha256=checksum(ROOT/'round35/live35.py'))
    write_json(dest/'results.json',dict(arm=arm,selection=selection,comparison=compares,changes=differences,
                                       paired=paired,receipt=stamp|fingerprints))
    print(differences,paired,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=('chosen','combined'))
    profiled(parser.parse_args().arm)
