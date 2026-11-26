"""Time the negative-branch correction while preserving signed archived decisions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from study44 import HERE,ROOT,DEST,ARMS,VIEWS,f,checksum,json_at,write_json,scope,design,crop,predict,cut
from structure44 import DEST as FEATURES,extract,mapping
from live40 import anchor,pipeline
from live35 import forward
from pipeline import PUBLIC,load

PAIRS = 6
WORKERS = 6


def score_channels(texts,forms,models,pool):
    if not forms:
        return [],0
    output = (pool.map(extract,texts,[forms]*len(texts),chunksize=32)
              if pool is not None else (extract(text,forms) for text in texts))
    data = [row for row,_,_ in output]
    scores = []
    operations = 0
    for kind in forms:
        x = mapping(kind).transform(r[kind] for r in data)
        scores.append(models[kind].decision_function(x))
        operations += 2*x.nnz
    return scores,operations


def chosen_head(previous,texts,t,selected,model,pool):
    extras = VIEWS[selected['view']]
    readout = model['readout'][selected['arm']]
    def apply(frame,rank,first,aux,context):
        base,_,ops = previous(frame,rank,first,aux,context)
        active = frame | {'signed':base}
        member = scope(active,selected['panel'],selected['width'])
        if not member.any():
            return base,member,ops
        q = [texts[i] for i in np.flatnonzero(member)]
        scores,work = score_channels(q,extras,model['sparse'],pool)
        x = design(crop(active,member),t,scores)
        result = (predict(readout,x,base[member]) if selected['model']=='linear'
                  else readout.decision_function(x))
        z = base.copy()
        mix = selected['mix']
        z[member] = (1.-mix)*base[member]+mix*result-selected['cut']
        return z,member,ops+work+5*x.size
    return apply


def micro():
    receipt = json_at(FEATURES/'receipt.json')
    assert checksum(FEATURES/'rows.npz')==receipt['roster_sha256']
    rows = np.load(FEATURES/'rows.npz')['ids']
    chosen = np.random.default_rng(4400).choice(rows,size=1500,replace=False)
    corpus = f.inputs()[0]
    texts = [corpus[k] for k in chosen]
    weights = DEST/'official/weights.joblib'
    assert checksum(weights)==json_at(weights.with_name('results.json'))['weights_sha256']
    models = joblib.load(weights)['sparse']
    views = {'lexical':VIEWS['lexical'],'syntax':VIEWS['syntax'],'body_only':('body',)}
    samples = {k:[] for k in views}
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
        t = perf_counter()
        got,_ = score_channels(texts[:96],ARMS,models,pool)
        init = perf_counter()-t
        expected,_ = score_channels(texts[:96],ARMS,models,None)
        assert all(np.array_equal(a,b) for a,b in zip(got,expected,strict=True))
        for i in range(5):
            order = list(views) if i%2==0 else list(reversed(views))
            for key in order:
                start = perf_counter()
                scores,_ = score_channels(texts,views[key],models,pool)
                samples[key].append(perf_counter()-start)
                assert all(np.isfinite(s).all() for s in scores)
    roster = HERE/'micro_rows.npz'
    np.savez_compressed(roster,ids=chosen)
    report = dict(rows=len(chosen),process_workers=WORKERS,samples=samples,
                  medians={k:median(v) for k,v in samples.items()},first_seconds=init,
                  readout_sha256=checksum(weights),roster_sha256=checksum(roster),
                  code_sha256=checksum(Path(__file__)))
    write_json(HERE/'micro.json',report)
    print(report,flush=True)


def public():
    source = DEST/'selection.json'
    plan = json_at(source)
    chosen = plan['chosen']
    if chosen is None:
        raise ValueError('No candidate passed the held-out and internal gates')
    proof = DEST/'official/results.json'
    weights = proof.with_name('weights.joblib')
    assert checksum(proof)==plan['sources']['official']
    assert checksum(weights)==json_at(proof)['weights_sha256']
    model = joblib.load(weights)
    archive,plate,truth,baseline = anchor()
    assert checksum(archive)==plan['control_sha256']
    texts,y,_ = load(PUBLIC)
    assert np.array_equal(truth,y) and len(texts)==30655
    modes = VIEWS[chosen['view']]
    if modes:
        with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
            start = perf_counter()
            score_channels(texts[:192],modes,model['sparse'],pool)
            ready = perf_counter()-start
            measures(texts,y,baseline,archive,plate,source,weights,chosen,model,pool,ready)
    else:
        measures(texts,y,baseline,archive,plate,source,weights,chosen,model,None,0.)


def measures(texts,y,baseline,archive,plate,source,weights,chosen,model,pool,ready):
    core,runtime,previous,setup = pipeline()
    def execute(batch,use):
        adjust = chosen_head(previous,batch,core[0]['threshold'],chosen,model,pool) if use else previous
        result = forward(batch,*setup,adjust)
        cost = result[-1]
        start = perf_counter()
        runtime.finish(cost)
        elapsed = perf_counter()-start
        cost['forward_seconds'] += elapsed
        cost['wall_seconds'] += elapsed
        return result
    for arm in (False,True):
        execute(texts[:4096],arm)
    med = {'reference':[],'candidate':[]}
    outputs = {}
    for i in range(PAIRS):
        for active in ((False,True) if i%2==0 else (True,False)):
            name = 'candidate' if active else 'reference'
            bits,rank,member,cost = execute(texts,active)
            if name in outputs:
                assert np.array_equal(bits,outputs[name][0])
            outputs[name] = bits,rank,member
            if active:
                assert np.array_equal(bits[:,1],baseline[:,1])
                assert np.array_equal(bits[~member],baseline[~member])
            else:
                assert np.array_equal(bits,baseline)
            q = f.metric(y,bits)['mean']['f1']
            score = f.competition_formula(q,cost['forward_seconds'],cost['analytic_gflops'])
            med[name].append(cost|{'score':score})
            print(i,name,score,cost,flush=True)
    out = HERE/'public'
    out.mkdir(exist_ok=True)
    data = {}
    for name,(bits,rank,member) in outputs.items():
        profile = {k:median(s[k] for s in med[name]) for k in med[name][0] if k!='score'}
        q = f.metric(y,bits)
        value = f.competition_formula(q['mean']['f1'],profile['forward_seconds'],profile['analytic_gflops'])
        file = out/(name+'.npz')
        np.savez_compressed(file,pred=bits,truth=y,margin=rank,scope=member)
        data[name] = dict(score=value,cost=profile,quality=q,trials=med[name],predictions_sha256=checksum(file))
    pred,r,scope = outputs['candidate']
    truth = y[:,0].astype(bool)
    control = baseline[:,0]
    stats = {'frozen':f.comparison(truth,control,pred[:,0])}
    for rule in ('fp','tp'):
        p = control.copy()
        p[scope] = r[scope]>cut(truth[scope],control[scope],r[scope],rule)
        stats[rule] = f.comparison(truth,control,p)
    pairs = [cand['score']-base['score'] for base,cand in zip(med['reference'],med['candidate'],strict=True)]
    release = json_at(ROOT/'round38/release_summary.json')
    c,r = data['candidate'],data['reference']
    promoted = (sum(p>0 for p in pairs)>=5 and c['score']>max(r['score'],release['score'])
                and c['quality']['mean']['f1']>r['quality']['mean']['f1'])
    receipt = dict(selection_sha256=checksum(source),public_sha256=checksum(PUBLIC),archive_sha256=checksum(archive),
                 anchor_sha256=checksum(plate),model_sha256=checksum(weights),code_sha256=checksum(Path(__file__)),
                 extractor_sha256=checksum(HERE/'structure44.py'),warmup_pool_seconds=ready,
                 workers=0 if pool is None else WORKERS)
    write_json(out/'results.json',dict(comparison=data,chosen=chosen,paired=pairs,changes=stats,
                       promoted=promoted,receipt=receipt))
    print(stats,pairs,promoted,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('micro','public'))
    args = parser.parse_args()
    micro() if args.stage=='micro' else public()
