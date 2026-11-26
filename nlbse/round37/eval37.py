"""Test regional evidence against score-only and position-matched controls."""
import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from section37 import HERE, ROOT, PARTS, STATE, NATIVE, SECTIONS, OUTPUTS, checksum, json_at, write_json, inputs
from eval35 import crop, comparisons, point_classifier
from broad35 import regimes, features as score_features

DEST = HERE/'comparison'
VIEWS = {'native':(), 'half':(2,3), 'declaration':(0,), 'body':(1,), 'regions':(0,1),
         'seam':(0,4,5), 'joint':(0,1,2,3,4,5)}
TREE_MODES = ('control', 'native', 'regions', 'seam', 'joint')


def features(rows, threshold, mode, scoped):
    names = mode.split('_')
    view = names[0]
    core = score_features(rows,threshold,'regime')
    if view=='control':
        return core
    data = np.asarray(scoped)
    base = np.clip(data[:,:NATIVE],-12.,12.)
    choice = VIEWS[view]
    if not choice:
        return np.column_stack((core,base))
    signals = np.clip(data[:,NATIVE:NATIVE+SECTIONS*OUTPUTS],-8.,8.)
    pieces = signals.reshape(-1,SECTIONS,OUTPUTS)[:,choice,:]
    count = data[:,NATIVE+SECTIONS*OUTPUTS:][:,choice]
    flat = pieces.reshape(len(core),-1)
    if len(names)>1 and names[1]=='specific':
        role = regimes(rows)
        mask = role[:,None]==np.arange(4)[None,:]
        # Keep the semantic outputs global and let regional deltas modify
        # calibration separately according to the signs before/after scanning.
        conditional = (flat[:,:,None]*mask[:,None,:]).reshape(len(core),-1)
        return np.column_stack((core,base,flat,count,conditional))
    return np.column_stack((core,base,flat,count))


def estimate(x, data, mode):
    y = data['y']
    score = data['latest']
    all_rows = np.ones(len(y),bool)
    if mode.endswith('_trees'):
        h = HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=12,learning_rate=.055,
                        min_samples_leaf=max(26,len(y)//1350),random_state=42,l2_regularization=8.,
                        early_stopping=False)
        return h.fit(x,y,sample_weight=.3+1.35*np.exp(-.5*np.abs(score)))
    if not mode.endswith('_specific'):
        return point_classifier(x,y,score,all_rows)
    role = regimes(data)
    scale = np.bincount(role,minlength=4)
    target = np.clip(np.sqrt(len(y)/np.maximum(4*scale,1)),.64,2.25)
    sample_weight = (.3+2.*np.exp(-.8*np.abs(score)))*target[role]
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.06,solver='liblinear',
                                 max_iter=250,random_state=42))
    return estimator.fit(x,y,logisticregression__sample_weight=sample_weight)


def unpack(rotation, phases, base, roster, proof):
    result = {}
    origin = STATE/rotation
    receipt = json_at(origin/'receipt.json')
    cached = origin/'scores.npz'
    assert checksum(cached)==receipt['state_sha256']
    assert checksum(HERE/'section37.py')==receipt['code_sha256']
    assert receipt['cohorts_sha256']==roster and receipt['baseline_sha256']==base
    assert receipt['points']==proof['phases']
    with np.load(cached) as z:
        for phase,data in phases.items():
            use = data['support']
            assert np.array_equal(z[phase+'_ids'],data['ids'][use])
            matrix = np.zeros((len(use),z[phase].shape[1]),np.float32)
            matrix[use] = z[phase]
            result[phase] = data | {'section':matrix}
    return result,checksum(origin/'receipt.json')


def evaluated(rotation):
    raw,ref,base,roster = inputs(rotation)
    points,source = unpack(rotation,raw,base,roster,ref)
    d = points['dev']
    other = points.get('test',d)
    fit, cal, held = crop(d,d['fit']),crop(d,d['cal']),crop(other,other['held'])
    t = ref['cutoff']
    oof = points['oof']
    modes = ('control',*VIEWS,*(k+'_specific' for k in ('regions','seam','joint')),
             *(k+'_trees' for k in TREE_MODES))
    models, rank = {}, {}
    evals = [crop(p,p['support']) for p in (cal,held)]
    for key,trial in [('fit',fit),('oof',oof)]:
        tagged = crop(trial,trial['support'])
        for mode in modes:
            model = estimate(features(tagged,t,mode,tagged['section']),tagged,mode)
            key_id = (key,mode)
            models[key_id] = model
            values = []
            for original,subset in zip((cal,held),evals,strict=True):
                output = np.zeros(len(original['ids']),np.float64)
                output[original['support']] = model.decision_function(features(subset,t,mode,subset['section']))
                values.append(output)
            rank[key_id] = values
        print(rotation,key,len(tagged['ids']),len(modes),flush=True)
    result, forecast = comparisons(rotation,cal,held,rank)
    out = DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    cached = out/'predictions.npz'
    weights = out/'readouts.joblib'
    np.savez_compressed(cached,**forecast)
    joblib.dump(models,weights,compress=3)
    write_json(out/'results.json',dict(rotation=rotation,options=result,predictions_sha256=checksum(cached),
               weights_sha256=checksum(weights),cohorts_sha256=roster,baseline_sha256=base,
               section_receipt_sha256=source,code_sha256=checksum(Path(__file__))))
    print(rotation,len(result),'settings',flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    evaluated(cli.parse_args().rotation)
