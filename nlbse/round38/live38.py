"""Benchmark a frozen cheap correction against the exact Round 37 release."""
import argparse
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np

from data38 import HERE, ROOT, R37, f, checksum, json_at, write_json, crop
from rank38 import active_set, positive_features, predict
from broad35 import features as score_features
from live35 import forward, heads, load_choice, recipe, packages
from live37 import encode
from deferred37 import ScopedStates
from eval35 import cut
from pipeline import PUBLIC, load


def correction(cap, threshold, selected, models):
    modes = tuple((role,selected[role]) for role in ('negative','positive'))
    widths = tuple(selected['width_'+role] for role,_ in modes)

    def adjust(live, current, first, aux, context):
        data = live | {'latest':current,'signed':current}
        selected_rows = active_set(data,modes,widths)
        chosen = np.logical_or(*selected_rows.values())
        result = current.copy()
        pay = 0
        for role,m in modes:
            scope = selected_rows[role]
            if not scope.any():
                continue
            z = crop(data,scope)
            if role=='positive':
                x = positive_features(z,threshold,aux[scope,:20],first[scope],m!='control')
            elif m=='control':
                x = score_features(z,threshold,'regime')
            else:
                x = cap.features(z,threshold,'regions_specific',context[scope,2:])
            estimate = predict(models[role,m],x,current[scope])
            w = selected['mix']
            result[scope] = (1.-w)*current[scope]+w*estimate-selected['cut']
            pay += 5*x.size
        return result,chosen,pay
    return adjust


def main(arm):
    plan = HERE/'selection.json'
    policy = json_at(plan)
    selected = policy['selected'] if arm=='primary' else policy['arms'][arm+'_cheap']
    assert selected and selected['deployable']
    path = HERE/'comparison/official/results.json'
    assert policy['sources']['official']==checksum(path)
    weight = path.with_name('models.joblib')
    assert checksum(weight)==json_at(path)['weights_sha256']
    models = joblib.load(weight)
    release = json_at(R37/'release_summary.json')
    archive = ROOT/'submission/NLBSE2027_round37_deferred_sections_tool.zip'
    assert checksum(archive)==policy['releases']==release['archive_sha256']
    public = R37/'public_semantic_deferred'
    plate = json_at(public/'results.json')
    ref_npz = public/'candidate.npz'
    assert checksum(ref_npz)==plate['comparison']['candidate']['predictions_sha256']
    assert checksum(PUBLIC)==release['public_sha256']
    source = R37/'comparison/official/results.json'
    parameters = json_at(R37/'selection.json')
    assert checksum(source)==parameters['source_table']['official']
    regional = parameters['best']['semantic']
    ref_weight = source.with_name('readouts.joblib')
    assert checksum(ref_weight)==json_at(source)['weights_sha256']
    previous = joblib.load(ref_weight)[regional['source'],regional['mode']]
    old_plan = ROOT/'round34/selection_approx.json'
    old = json_at(old_plan)['selected']
    projection,calibrator,_ = heads(old,old_plan)

    texts,y,_ = load(PUBLIC)
    with np.load(ref_npz) as z:
        reference = z['pred']
        assert np.array_equal(y,z['truth'])
    encoder = f.pipe('official',f.small.tokenizer())
    score = recipe()
    run = ScopedStates(score,f.pinned()[0])
    enc = partial(encode,deferred=run)
    adjust = correction(run,score[0]['threshold'],selected,models)
    rule,coef,_ = load_choice()
    common = (encoder,score,packages(),rule,coef,old,projection,calibrator)
    def execute(src,which):
        result = forward(src,*common,regional,True,previous,enc,run.features,adjust if which else None)
        t = perf_counter()
        cost = result[-1]
        run.finish(cost)
        elapsed = perf_counter()-t
        cost['forward_seconds'] += elapsed
        cost['wall_seconds'] += elapsed
        return result
    for flag in (False,True):
        execute(texts[:4096],flag)
    kept = {}
    trials = {'reference':[],'candidate':[]}
    for i in range(5):
        for flag in ((False,True) if i%2==0 else (True,False)):
            bits,rank,scope,cost = execute(texts,flag)
            role = 'candidate' if flag else 'reference'
            if role in kept:
                assert np.array_equal(bits,kept[role][0])
            kept[role] = bits,rank,scope
            if flag:
                assert np.array_equal(bits[:,1],reference[:,1])
                assert np.array_equal(bits[~scope],reference[~scope])
            else:
                assert np.array_equal(bits,reference)
            F = f.metric(y,bits)['mean']['f1']
            measured = f.competition_formula(F,cost['forward_seconds'],cost['analytic_gflops'])
            trials[role].append(cost|{'score':measured})
            print(i,role,measured,cost,flush=True)

    answer = HERE/f'public_{arm}'
    answer.mkdir(exist_ok=True)
    lines = {}
    for key,(pred,margin,scope) in kept.items():
        cost = {n:median(t[n] for t in trials[key]) for n in trials[key][0] if n!='score'}
        info = f.metric(y,pred)
        adjusted = f.competition_formula(info['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        save = answer/f'{key}.npz'
        np.savez_compressed(save,pred=pred,truth=y,scope=scope,margin=margin)
        lines[key] = dict(score=adjusted,quality=info,cost=cost,trials=trials[key],
                          predictions_sha256=checksum(save))
    pred,rank,use = kept['candidate']
    original = reference[:,0]
    gold = y[:,0].astype(bool)
    changes = {'frozen':f.comparison(gold,original,pred[:,0])}
    for rule in ('fp','tp'):
        new = original.copy()
        t = cut(gold[use],original[use],rank[use],rule)
        new[use] = rank[use]>t
        changes[rule] = f.comparison(gold,original,new)
    paired = [b['score']-a['score'] for a,b in zip(trials['reference'],trials['candidate'],strict=True)]
    report = dict(arm=arm,selected=selected,comparison=lines,changes=changes,paired=paired,
                  receipt=dict(plan_sha256=checksum(plan),data_sha256=checksum(PUBLIC),anchor_sha256=checksum(ref_npz),
                               archive_sha256=checksum(archive),code_sha256=checksum(Path(__file__)),
                               runtime_sha256=checksum(ROOT/'round35/live35.py'),model_sha256=checksum(weight)))
    write_json(answer/'results.json',report)
    print(changes,paired,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arm',choices=('primary','negative','positive','joint'))
    main(cli.parse_args().arm)
