"""Measure selected first-hit features in isolation and with the released pipeline."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from features43 import (HERE, ROOT, R39, OUTPUT, PROGRAM, f, checksum, json_at, write_json,
                        profile, contexts, labels_for, syntax, letters, without_literals,
                        vectorizer, WINDOW, CAP)
from study43 import DEST, READOUTS, design, eligible_scope, crop, cut, predict
from live39 import verify_scanner
from live40 import anchor, pipeline
from live35 import forward
from location39 import scan
from pipeline import PUBLIC, load


def live_context(text,values,wanted):
    g,raw,_ = profile(text,values)
    output = {}
    if 'text' in wanted:
        output['text'] = raw
    if any(view in wanted for view in ('masked','ast','center')):
        rows = text.split('\n')
        line = int(values[3])-1
        length = len(rows)
        begin,end = max(0,line-WINDOW),min(length,line+WINDOW+1)
        if 'masked' in wanted:
            masked = without_literals('\n'.join(rows[begin:end]))[:CAP]
            output['masked'] = ' '.join(letters(masked))
        if 'ast' in wanted or 'center' in wanted:
            tree,_ = syntax(text)
            if 'ast' in wanted:
                output['ast'] = labels_for(tree,line,begin,end)
            if 'center' in wanted:
                middle = (length-1)//2
                last = min(length,max(end-begin,middle+WINDOW+1))
                output['center'] = labels_for(tree,middle,last-(end-begin),last)
    return g,output


def features(rows, texts, modes, models, cutoff):
    geo = np.empty((len(texts),17),np.float32)
    terms = {k:[] for k in modes}
    for i,(p,v) in enumerate(zip(texts,rows['location'],strict=True)):
        geo[i],data = live_context(p,v,modes)
        for key in data:
            terms[key].append(data[key])
    extras,ops = [],0
    for key in modes:
        matrix = vectorizer(key).transform(terms[key])
        extras.append(models[key].decision_function(matrix))
        ops += 2*matrix.nnz
    return design(rows,cutoff,geo,extras),ops


def hook(previous, texts, threshold, selected, weights):
    nonlinear = 'view' in selected
    if nonlinear:
        from conditional43 import VIEWS
        parts = VIEWS[selected['view']]
        main = weights['readout'][selected['arm']]
    else:
        parts = READOUTS[selected['kind']]
        main = weights['readout'][selected['mode']]
    def adjust(frame,rank,first,aux,context):
        current,_,ops = previous(frame,rank,first,aux,context)
        ready = frame | {'signed':current,'latest':current,'first':first,'aux_positive':aux[:,:20]}
        scope = eligible_scope(ready,selected['panel'],selected['width'])
        if not scope.any():
            return current,scope,ops
        x,n = features(crop(ready,scope),[texts[i] for i in np.flatnonzero(scope)],parts,weights['sparse'],threshold)
        if nonlinear:
            x = np.column_stack((current[scope],x))
            z = main.decision_function(x)
        else:
            z = predict(main,x,current[scope])
        result = current.copy()
        s = selected['mix']
        result[scope] = (1.-s)*current[scope]+s*z-selected['threshold']
        return result,scope,ops+n+5*x.size
    return adjust


def median_time(src,data,kind,readout):
    start = perf_counter()
    terms = []
    for text,location in zip(src,data,strict=True):
        value = live_context(text,location,(kind,))[1][kind]
        terms.append(value)
    design = vectorizer(kind).transform(terms)
    score = readout.decision_function(design)
    assert np.isfinite(score).all()
    return perf_counter()-start


def micro():
    # Use only the labelled training parquet and a roster fixed without its labels.
    geo_file = OUTPUT/'training.npz'
    assert json_at(OUTPUT/'receipt.json')['predictions_sha256']==checksum(geo_file)
    cache = np.load(geo_file)
    pool = (cache['values'][:,0]==1)&(cache['values'][:,3]>0)
    rng = np.random.default_rng(4301)
    pick = rng.choice(np.flatnonzero(pool),1500,replace=False)
    locations = cache['values'][pick]
    train = f.inputs()[0]
    src = [train[i] for i in cache['ids'][pick]]
    signed = DEST/'official/weights.joblib'
    assert checksum(signed)==json_at(signed.with_name('results.json'))['weights_sha256']
    model = joblib.load(signed)['sparse']
    modes = ('text','masked','ast','center')
    for txt,loc in zip(src[:64],locations[:64],strict=True):
        g,*terms,_,_ = contexts(txt,loc)
        v,o = live_context(txt,loc,modes)
        assert np.array_equal(g,v) and terms==[o[k] for k in modes]
    for view in modes:
        median_time(src[:64],locations[:64],view,model[view])
    stats = {mode:[] for mode in modes}
    for trial in range(5):
        for view in (modes if trial%2==0 else modes[::-1]):
            stats[view].append(median_time(src,locations,view,model[view]))
    output = dict(rows=len(src),times=stats,medians={k:median(v) for k,v in stats.items()},
                  model_sha256=checksum(signed),geometry_sha256=checksum(geo_file),
                  code_sha256=checksum(Path(__file__)))
    np.savez_compressed(HERE/'micro_rows.npz',ids=cache['ids'][pick],scanner=locations)
    output['rows_sha256'] = checksum(HERE/'micro_rows.npz')
    write_json(HERE/'micro.json',output)
    print(output,flush=True)


def main(arm):
    directory = HERE/'conditional' if arm=='conditional' else DEST
    history = directory/'selection.json'
    plan = json_at(history)
    chosen = plan['chosen' if arm=='conditional' else arm]
    if chosen is None:
        raise ValueError('An eligible policy is required for public evaluation')
    ready = directory/'official/results.json'
    assert checksum(ready)==plan['sources']['official']
    weights_file = ready.with_name('weights.joblib')
    assert checksum(weights_file)==json_at(ready)['weights_sha256']
    weights = joblib.load(weights_file)
    archive,plate,y,baseline = anchor()
    assert checksum(archive)==json_at(ROOT/'round38/release_summary.json')['archive_sha256']
    texts,truth,_ = load(PUBLIC)
    assert np.array_equal(y,truth)
    checked,digest = verify_scanner(texts)
    core,state,previous,setup = pipeline()
    def run(src,active):
        fn = hook(previous,src,core[0]['threshold'],chosen,weights) if active else previous
        result = forward(src,*setup,fn,scan if active else None)
        cost = result[-1]
        start = perf_counter()
        state.finish(cost)
        elapsed = perf_counter()-start
        cost['forward_seconds'] += elapsed
        cost['wall_seconds'] += elapsed
        return result
    for key in (False,True):
        run(texts[:4096],key)
    samples = {'candidate':[],'reference':[]}
    forecasts = {}
    for i in range(5):
        for choice in ((False,True) if i%2==0 else (True,False)):
            kind = 'candidate' if choice else 'reference'
            bits,rank,scope,cost = run(texts,choice)
            if kind in forecasts:
                assert np.array_equal(bits,forecasts[kind][0])
            forecasts[kind] = bits,rank,scope
            if choice:
                assert np.array_equal(bits[:,1],baseline[:,1])
                assert np.array_equal(bits[~scope],baseline[~scope])
            else:
                assert np.array_equal(bits,baseline)
            q = f.metric(y,bits)['mean']['f1']
            s = f.competition_formula(q,cost['forward_seconds'],cost['analytic_gflops'])
            samples[kind].append(cost | {'score':s})
            print(i,kind,s,cost,flush=True)
    out = HERE/('public_'+arm)
    out.mkdir(exist_ok=True)
    scores = {}
    for name,observations in samples.items():
        pred,rank,active = forecasts[name]
        cost = {k:median(x[k] for x in observations) for k in observations[0] if k!='score'}
        q = f.metric(y,pred)
        s = f.competition_formula(q['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        path = out/(name+'.npz')
        np.savez_compressed(path,pred=pred,truth=y,margin=rank,scope=active)
        scores[name] = dict(score=s,cost=cost,quality=q,trials=observations,predictions_sha256=checksum(path))
    pred,r,s = forecasts['candidate']
    ytrue = y[:,0].astype(bool)
    original = baseline[:,0]
    change = {'frozen':f.comparison(ytrue,original,pred[:,0])}
    if s.any():
        for method in ('fp','tp'):
            proposal = original.copy()
            proposal[s] = r[s]>cut(ytrue[s],original[s],r[s],method)
            change[method] = f.comparison(ytrue,original,proposal)
    paired = [b['score']-a['score'] for a,b in zip(samples['reference'],samples['candidate'],strict=True)]
    release = json_at(ROOT/'round38/release_summary.json')
    qualifies = (sum(d>0. for d in paired)>=4 and scores['candidate']['score']>max(scores['reference']['score'],release['score'])
                  and scores['candidate']['quality']['mean']['f1']>scores['reference']['quality']['mean']['f1'])
    receipt = dict(selection_sha256=checksum(history),public_sha256=checksum(PUBLIC),anchor_sha256=checksum(plate),
                   archive_sha256=checksum(archive),scanner_sha256=checksum(PROGRAM),scanner_comparisons=checked,
                   first_hit_sha256=digest,readout_sha256=checksum(weights_file),code_sha256=checksum(Path(__file__)),
                   features_sha256=checksum(HERE/'features43.py'))
    write_json(out/'results.json',dict(arm=arm,choice=chosen,scores=scores,change=change,
                       paired=paired,qualified=qualifies,receipt=receipt))
    print(change,paired,qualifies,flush=True)


if __name__=='__main__':
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('arm',choices=('micro','chosen','chosen_ast','conditional'))
    arm = arg.parse_args().arm
    if arm=='micro':
        micro()
    else:
        main(arm)
