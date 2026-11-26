"""Evaluate selected before-scanner whitespace repairs in the full released pipeline."""
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np
import torch

from causal54 import HERE, ROOT, SCRIPT, checksum, f, json_at, load, parts, scan, write_json
from effects54 import alterations, estimates
from indent54 import propose, rewrite
from select54 import DEST as BOARD
from live35 import forward
from live40 import anchor, pipeline
from pipeline import PUBLIC, load as corpus_load

DEST = HERE / 'live'


def transform(choice):
    kind, variant = choice['kind'], choice['variant']
    mode, count, _ = variant.split('/')
    count = int(count)
    folder = HERE / kind / 'official'
    plan = folder / 'results.json'
    result = json_at(plan)
    assert checksum(plan) == choice['source']['official']
    edge = choice['official_threshold']
    stats = dict(edited=0, scanner_rows=0, text_seconds=0., ml_gflops=0.)
    if kind == 'edit':
        path = folder / 'models.joblib'
        assert checksum(path) == result['proof']['models_sha256']
        bank = joblib.load(path)
    else:
        path = folder / 'model.joblib'
        assert checksum(path) == result['model_sha256']
        model = joblib.load(path)

    def change(text):
        ready = list(text)
        started = perf_counter()
        if kind == 'edit':
            rows, raw = estimates(text, bank)
            ranked = raw[mode]
            for i, score in ranked.items():
                if edge is not None and score.max() >= edge and (mode == 'source' or score.max()>0.):
                    ready[i] = alterations(text[i],rows[i],score,count)
            rows_done = sum(len(x) for x in ranked.values())
            trees = sum(sum(len(stage) for stage in m._predictors) for m in bank.values())
        else:
            packed = propose(text,model)
            for i, p in packed.items():
                if edge is not None and p[3].max() >= edge:
                    ready[i] = rewrite(text[i],p,count)
            rows_done = sum(len(p[1]) for p in packed.values())
            trees = sum(len(stage) for stage in model._predictors)
        stats.update(edited=sum(a!=b for a,b in zip(text,ready,strict=True)),scanner_rows=len(text),
                     text_seconds=perf_counter()-started,ml_gflops=rows_done*trees*8/1e9)
        return scan(ready,True)[0]

    return change,stats,checksum(path)


def execute(text, setup, change=None, stats=None):
    _, run, stage, args = setup
    if stats is not None:
        stats.update(edited=0, scanner_rows=0, text_seconds=0., ml_gflops=0.)
    answer, _, _, cost = forward(text,*args,stage,scanner_fn=change)
    start = perf_counter()
    run.finish(cost)
    extra = perf_counter()-start
    cost['forward_seconds'] += extra
    cost['wall_seconds'] += extra
    if stats is not None:
        assert cost['scanner_calls']==stats['scanner_rows']
        cost['analytic_gflops']+=stats['ml_gflops']
        cost.update(stats)
    return answer,cost


def profile(choice,setup):
    modify,stats,weights = transform(choice)
    archive,previous,gold,reference = anchor()
    texts,y,_ = corpus_load(PUBLIC)
    assert np.array_equal(gold,y)
    samples,trials={}, {'reference':[], 'candidate':[]}
    execute(texts[:4096],setup)
    execute(texts[:4096],setup,modify,stats)
    for i in range(5):
        for tag in (('reference','candidate') if i%2==0 else ('candidate','reference')):
            cand = tag=='candidate'
            q,cost = execute(texts,setup,modify,stats) if cand else execute(texts,setup)
            if tag in samples:
                assert np.array_equal(samples[tag],q)
            assert np.array_equal(q[:,1],reference[:,1])
            if not cand:
                assert np.array_equal(q,reference)
            samples[tag] = q
            score = f.competition_formula(f.metric(y,q)['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            trials[tag].append(cost | {'score': score})
            print(i, tag, score, cost['forward_seconds'], cost.get('edited',0),flush=True)
    result = {}
    for tag, q in samples.items():
        quality=f.metric(y,q)
        pay={k:median(v[k] for v in trials[tag]) for k in trials[tag][0] if k!='score'}
        score=f.competition_formula(quality['mean']['f1'],pay['forward_seconds'],pay['analytic_gflops'])
        result[tag]=dict(quality=quality,cost=pay,score=score,trials=trials[tag])
    diff=[b['score']-a['score'] for a,b in zip(trials['reference'],trials['candidate'],strict=True)]
    out=DEST/'predictions.npz'
    np.savez_compressed(out,truth=y,**samples)
    chosen=bool(sum(x>0 for x in diff)>=4 and result['candidate']['score']>max(result['reference']['score'],
                json_at(ROOT/'round38/release_summary.json')['score']))
    write_json(DEST/'public.json',dict(choice=choice,qualified=chosen,measurements=result,paired_differences=diff,
               predictions_sha256=checksum(out),archive_sha256=checksum(archive),weights_sha256=weights,
               baseline_sha256=checksum(previous),public_sha256=checksum(PUBLIC),scanner_sha256=checksum(SCRIPT),
               code_sha256=checksum(Path(__file__))))


def main():
    plan=BOARD/'selection.json'
    selected=json_at(plan)
    assert selected['comparison_sha256']==checksum(plan.with_name('comparison.csv'))
    assert selected['code_sha256']==checksum(HERE/'select54.py')
    DEST.mkdir(exist_ok=True)
    if not selected['selected']:
        write_json(DEST/'selection.json',dict(selected=None, reason='no candidate passed held-out W',
                   offline_sha256=checksum(plan),code_sha256=checksum(Path(__file__))))
        return
    setup=pipeline()
    source,_,_=load('official')
    rows=parts(source)
    text,labels,project,path,_=f.inputs()
    fit_files={(project[i],path[i]) for i in source['oof']['ids'] if path[i]}
    bases=[execute([text[i] for i in frame['ids']],setup) for frame in rows]
    evidence=[]
    for choice in selected['selected']:
        trans,measures,weights=transform(choice)
        proof={}
        for (before,cost),frame,role in zip(bases, rows, ('cal','held'),strict=True):
            ids=frame['ids']
            after,pay=execute([text[i] for i in ids],setup,trans,measures)
            assert np.array_equal(after[:,1],before[:,1]) and np.array_equal(labels[ids,0],frame['y'])
            assert pay['scanner_calls']==cost['scanner_calls']
            m=np.asarray([bool(path[i]) and (project[i],path[i]) not in fit_files for i in ids])
            stats=f.comparison(frame['y'],before[:,0],after[:,0])
            unseen=f.comparison(frame['y'][m],before[m,0],after[m,0])
            proof[role]=dict(change=stats,new_files=unseen,review_new_file_rows=int(m.sum()),base_cost=cost,candidate_cost=pay)
        allowed=all(o['change']['f1_after']>o['change']['f1_before'] and o['new_files']['f1_after']>=o['new_files']['f1_before']
                    for o in proof.values())
        evidence.append(dict(policy=choice,review=proof,weights_sha256=weights,supported=allowed))
        print(choice['variant'],{k:v['change'] for k,v in proof.items()},allowed,flush=True)
    finalists=[v for v in evidence if v['supported']]
    winner=max(finalists,key=lambda v:v['review']['cal']['change']['f1_after']) if finalists else None
    write_json(DEST/'selection.json',dict(selected=winner,candidates=evidence,offline_sha256=checksum(plan),
               code_sha256=checksum(Path(__file__))))
    if winner is not None:
        profile(winner['policy'],setup)


if __name__=='__main__':
    main()
