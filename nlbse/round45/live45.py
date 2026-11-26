"""Measure the cheap directional and lexical policies against the released tool."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from policy45 import (HERE,ROOT,DEST as FIT,f,checksum,json_at,load as matrices,parts,write_json,
                      experts,score,priority)
from compare45 import DEST,scope,route_query,cut
from source41 import crop
from live40 import anchor,pipeline,experts as sparse_models,vector,evaluate,aligned
from live35 import forward
from pipeline import PUBLIC,load

PAIRS = 6


def fill(frame,texts,legacy,new):
    old,work = aligned(legacy,texts,'token')
    matrix = vector(texts)
    result = frame | {'sparse':old,'new':evaluate(matrix,new)}
    return result,4*work+2*matrix.nnz


def operate(frame,text,t,choice,model,price,sparse,switch):
    m = scope(frame,choice['panel'],choice['width'])
    active = np.flatnonzero(m)
    signed = frame['signed']
    if not len(active):
        return signed,m,0,0
    sub = crop(frame,m)
    x = experts(sub,t)
    kind = choice['kind']
    point = kind=='point'
    q = score(model['point' if point else 'numeric'],x,signed[m],not point)
    paid = 5*x.size
    count = 0
    if kind=='lex':
        rule = choice['screen']
        u = signed[m]
        better = q
        feature = x
        if switch is not None:
            u = .25*signed[m]+.75*q
            feature = np.column_stack((x,u,q,u-signed[m]))
        gate = model['gate_lex'] if switch is None else switch['lex']
        benefit = priority(gate,feature,u,price['trade'],price['queries']['lex']['cost'])
        order = -abs(signed[m]) if rule in ('margin','margin_matched') else benefit
        limit = benefit if rule in ('priced','margin_matched') else None
        use = route_query(sub,np.ones(len(x),bool),order,choice['rate'],limit)
        count = int(use.sum())
        if count:
            more,ops = fill(crop(sub,use),[text[i] for i in active[use]],*sparse)
            z = experts(more,t,'lex')
            better[use] = score(model['lex'],z,signed[m][use])
            paid += ops+5*z.size
    result = signed.copy()
    amount = choice['mix']
    result[m] = (1.-amount)*signed[m]+amount*q-choice['cut']
    return result,m,paid,count


def micro():
    rotation = 'official'
    frames,t,check = matrices(rotation)
    hold = parts(frames)[1]
    fitted = FIT/rotation
    proof = json_at(fitted/'receipt.json')
    assert proof['matrix_sha256']==check and proof['files']['weights.joblib']==checksum(fitted/'weights.joblib')
    model = joblib.load(fitted/'weights.joblib')
    data = f.inputs()[0]
    ids = np.random.default_rng(4500).choice(len(hold['ids']),1500,replace=False)
    sources = [data[i] for i in hold['ids'][ids]]
    local = {k:v[ids] for k,v in hold.items()}
    sparse,stamps = sparse_models()
    reference,_ = fill(local,sources,*sparse)
    assert np.allclose(reference['sparse'],local['sparse'],atol=3e-5,rtol=0.)
    assert np.allclose(reference['new'],local['new'],atol=3e-5,rtol=0.)
    settings = dict(panel='all',width=3.8,mix=.75,cut=0.,screen='utility',rate=.25)
    times = {kind:[] for kind in ('numeric','lex')}
    count = {}
    for repeat in range(6):
        for kind in (tuple(times) if repeat%2==0 else tuple(reversed(times))):
            start = perf_counter()
            answer,_,_,queries = operate(local,sources,t,settings|{'kind':kind},model,proof,sparse,None)
            times[kind].append(perf_counter()-start)
            assert np.isfinite(answer).all()
            assert count.setdefault(kind,queries)==queries
    path = HERE/'micro_rows.npz'
    np.savez_compressed(path,ids=hold['ids'][ids])
    write_json(HERE/'micro.json',dict(rows=len(ids),queries=count,trials=times,medians={k:median(v) for k,v in times.items()},
                receipts=stamps,roster_sha256=checksum(path),code_sha256=checksum(Path(__file__))))
    print(count,{k:median(v) for k,v in times.items()},flush=True)


def public(which):
    board = HERE/'switch_comparison' if which=='switch' else DEST
    file = board/'selection.json'
    plan = json_at(file)
    choice = plan['cheap']
    if choice is None:
        raise ValueError('No validated cheap policy')
    archive,proof,y,base = anchor()
    assert checksum(archive)==plan['archive_sha256']
    assert checksum(board/'official/results.json')==plan['sources']['official']
    source = FIT/'official/receipt.json'
    price = json_at(source)
    files = FIT/'official/weights.joblib'
    assert checksum(files)==price['files']['weights.joblib']
    model = joblib.load(files)
    switch = None
    if which=='switch':
        home = HERE/'switch/official'
        info = json_at(home/'receipt.json')
        assert checksum(source)==info['fit_sha256']
        assert checksum(home/'weights.joblib')==info['weights_sha256']
        switch = joblib.load(home/'weights.joblib')
    sparse,stamps = sparse_models() if choice['kind']=='lex' else (None,{})
    texts,truth,_ = load(PUBLIC)
    assert np.array_equal(truth,y) and len(texts)==30655
    core,state,previous,setup = pipeline()
    def infer(values,active):
        asks = 0
        def hook(frame,rank,first,aux,context):
            nonlocal asks
            old,_,cost = previous(frame,rank,first,aux,context)
            value,use,ops,asks = operate(frame|{'signed':old},values,core[0]['threshold'],choice,model,price,sparse,switch)
            return value,use,cost+ops
        bits,score,select,cost = forward(values,*setup,hook if active else previous)
        start = perf_counter()
        state.finish(cost)
        dt = perf_counter()-start
        cost['forward_seconds'] += dt
        cost['wall_seconds'] += dt
        return bits,score,select,cost|{'extra_queries':asks}
    for mode in (False,True):
        infer(texts[:4096],mode)
    trials = {name:[] for name in ('reference','candidate')}
    drawn = {}
    for repeat in range(PAIRS):
        for mode in ((False,True) if repeat%2==0 else (True,False)):
            name = 'candidate' if mode else 'reference'
            p,z,s,cost = infer(texts,mode)
            if name in drawn:
                assert np.array_equal(p,drawn[name][0])
            drawn[name] = p,z,s
            if mode:
                assert np.array_equal(p[:,1],base[:,1]) and np.array_equal(p[~s],base[~s])
            else:
                assert np.array_equal(p,base)
            quality = f.metric(y,p)['mean']['f1']
            score = f.competition_formula(quality,cost['forward_seconds'],cost['analytic_gflops'])
            trials[name].append(cost|{'score':score})
            print(repeat,name,score,cost,flush=True)
    where = HERE/('public_'+which)
    where.mkdir(exist_ok=True)
    output = {}
    for name,(p,z,s) in drawn.items():
        store = where/(name+'.npz')
        np.savez_compressed(store,pred=p,truth=y,margin=z,scope=s)
        quality = f.metric(y,p)
        cost = {k:median(x[k] for x in trials[name]) for k in trials[name][0] if k!='score'}
        score = f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        output[name] = dict(score=score,quality=quality,cost=cost,trials=trials[name],predictions_sha256=checksum(store))
    predicted,margin,selected = drawn['candidate']
    gold = y[:,0].astype(bool)
    baseline = base[:,0]
    stats = {'frozen':f.comparison(gold,baseline,predicted[:,0])}
    for target in ('fp','tp'):
        proposal = baseline.copy()
        proposal[selected] = margin[selected]>cut(gold[selected],baseline[selected],margin[selected],target)
        stats[target] = f.comparison(gold,baseline,proposal)
    paired = [b['score']-a['score'] for a,b in zip(trials['reference'],trials['candidate'],strict=True)]
    a,b = (output[name] for name in ('reference','candidate'))
    confirmed = json_at(ROOT/'round38/release_summary.json')['score']
    promoted = (sum(v>0 for v in paired)>=5 and b['score']>max(a['score'],confirmed)
                and b['quality']['mean']['f1']>a['quality']['mean']['f1'])
    report = dict(mode=which,chosen=choice,comparison=output,changes=stats,paired=paired,promoted=promoted,
                  receipt=stamps|dict(selection_sha256=checksum(file),archive_sha256=checksum(archive),
                    anchor_sha256=checksum(proof),data_sha256=checksum(PUBLIC),weights_sha256=checksum(files),
                    code_sha256=checksum(Path(__file__)),policy_sha256=checksum(HERE/'policy45.py')))
    write_json(where/'results.json',report)
    print(stats,paired,promoted,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('micro','initial','switch'))
    mode = cli.parse_args().mode
    micro() if mode=='micro' else public(mode)
