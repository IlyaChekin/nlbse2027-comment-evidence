"""Replay the Round 32 decision both with and without each scanner response."""
from pathlib import Path
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R32 = ROOT/'round32'
sys.path.insert(0,str(R32))
from context32 import (PARTS, R24, f, checksum, json_at, write_json, reference, signed_source, early_out,
                       prepare, section_margin, neg_cut, replay, raw_values, REFERENCE, strict_cut)
from disagreement import clip_logit
from length_eval import base_context
from fuse_sparse import calibrate_sparse
from veto import sparse_calibration
from subgroups32 import features as group_features

OUT = HERE/'points'
CHOICE = R32/'selection.json'


def rule_calibration(rotation):
    study = json_at(CHOICE)
    key = study['benchmark_candidate']['option']
    home = R32/'calibrated_rules'/rotation
    result = json_at(home/'results.json')
    if checksum(home/'results.json') != study['option_tables']['calibrated_rules'][rotation]:
        raise ValueError('The signed rule calibration changed')
    path = home/'models.joblib'
    if checksum(path) != result['weights_sha256']:
        raise ValueError('The signed rule coefficients changed')
    current = result['options'][key]
    cfg = current['setting'] | {'cut':current['tp']['cut']}
    return cfg, joblib.load(path)[cfg['style']]


def apply_calibration(rank, state, counts, cfg, coef):
    take = state['known'] & state['flag'] & (abs(rank)<cfg['width'])
    if take.any():
        x = group_features(rank[take],counts[take],cfg['style'])
        rank[take] += cfg['strength']*(x@coef)-cfg['cut']
    return rank


def score_points(frame,texts,spec,shift,cfg,coef):
    values = frame[REFERENCE]
    pred = f.estimate(spec['heads'],values)
    before, no, yes, asked = f.choose(values,pred,spec['routing'],spec['gate'])
    valid = f.valid(frame['raw'])
    flag = f.weak(frame['raw'])
    full = np.where(valid,np.where(flag,yes,no),before).astype(np.float32)
    state = dict(p=full,no=no,yes=yes,w=pred['w'],known=valid,flag=flag,asked=asked)
    ctx = base_context(frame,state,spec['sparse'],spec['combiner'],spec['cutoff'],spec['variant']['threshold'],texts)
    # Preserve the original full-column evaluation order. The section model can reuse
    # cached regional scores when a later pass uses a smaller query set.
    cache = dict(available=np.zeros(len(full),bool), value=np.empty((len(full),4)))
    mid = section_margin(ctx,spec,cache)
    raw = early_out(frame['raw'])
    exact = apply_calibration(replay(ctx,mid,spec,shift),state,raw,cfg,coef)
    cutoff = clip_logit(np.asarray(np.float32(spec['cutoff']),dtype=np.float64))
    no_scan = clip_logit(before.astype(np.float32).astype(np.float64))-cutoff
    marginal = np.where(asked,exact,no_scan)
    lengths = np.fromiter((len(text) for text in texts),dtype=np.int32)
    utility = f.query_value(spec['routing'],before,no,yes,pred['w'],spec['gate']['base'])
    usable = np.column_stack((values,pred['w'],no,yes,before,utility))
    return dict(ids=frame['ids'],y=frame['y'].astype(bool),length=lengths,pre=usable,old=asked,
                queried=exact,unqueried=no_scan,replay=marginal,flag=np.where(valid,flag.astype(np.int8),-1))


def sparse_fit_source(rotation, raw):
    if rotation=='official':
        pack = R24/'official/tuned'
        info = json_at(pack/'policy.json')
        if checksum(pack/'models.joblib')!=info['models_sha256']:
            raise ValueError('Sparse calibration changed')
        model = joblib.load(pack/'models.joblib')['token']['oof']
    else:
        pack = R24/rotation
        info = json_at(pack/'fit.json')['models']['token']
        model = sparse_calibration(pack,'token',info)
    return calibrate_sparse(model,raw)


def outside_files(rotation,spec,shift,cfg,coef):
    source, signed = signed_source(rotation)
    base = ROOT/'round26/compact'/rotation
    info = json_at(base/'manifest.json')
    fo = base/'scores.npz'
    if info['sparse_oof_sha256']!=signed or checksum(fo)!=info['sha256']:
        raise ValueError('The outside-file compact predictions changed')
    with np.load(fo) as scores:
        if not all(np.array_equal(scores[k],source[k]) for k in ('ids','y','fold')):
            raise ValueError('Incompatible outside-file folds')
        values = scores['values']
    corpus,labels,project,_,_ = f.inputs()
    ids = source['ids']
    if not np.array_equal(labels[ids,0],source['y']):
        raise ValueError('Out-of-file labels no longer align')
    raw = raw_values(ids)
    flag = np.where(f.valid(raw),f.weak(raw).astype(np.int8),-1)
    if not np.array_equal(flag,source['weak']):
        raise ValueError('Exact rule cache changed')
    text = [corpus[i] for i in ids]
    frame = {REFERENCE:values,'raw':raw,'ids':ids,'y':source['y']}
    package = spec | {'raw':source['logits'],'sparse':sparse_fit_source(rotation,source['logits'])}
    result = score_points(frame,text,package,shift,cfg,coef)
    result.update(fold=source['fold'],duplicates=source['dup'],project=project[ids])
    print(rotation,'outside files',len(ids),'switches',int(((result['queried']>0)!=(result['unqueried']>0)).sum()),flush=True)
    return result,dict(compact_oof_sha256=info['sha256'],sparse_oof_sha256=signed)


def collect(rotation,include_oof):
    dev,test,fit,cal,review,db,tb,m0,m1 = reference(rotation)
    specs = prepare(rotation,dev,test,fit,cal)
    cfg,coef = rule_calibration(rotation)
    cached = dict(available=np.zeros(len(dev.texts),bool),value=np.empty((len(dev.texts),4)))
    d29 = section_margin(dev,specs[0],cached)
    shift = neg_cut(dev,d29,specs[0],cal)
    scope = np.ones(len(tb),bool) if review is None else review
    path = OUT/rotation
    path.mkdir(parents=True,exist_ok=True)
    state = []
    for phase,ctx,p in zip(('dev','test'),(dev,) if dev is test else (dev,test),specs,strict=False):
        observed = score_points(ctx.frame,ctx.texts,p,shift,cfg,coef)
        baseline = db if phase=='dev' else tb
        cache = dict(available=np.zeros(len(ctx.texts),bool),value=np.empty((len(ctx.texts),4)))
        rank29 = replay(ctx,section_margin(ctx,p,cache),p,shift)
        if not np.array_equal(rank29>0.,baseline):
            raise ValueError('The original observed decision changed')
        rank32 = apply_calibration(rank29,ctx.state,early_out(ctx.frame['raw']),cfg,coef)
        if not np.array_equal(observed['replay']>0.,rank32>0.):
            indices = np.flatnonzero((observed['replay']>0.)!=(rank32>0.))
            raise ValueError(f'The counterfactual disagrees with observed flow on {len(indices)} rows')
        observed['project'] = ctx.frame.get('project',np.full(len(ctx.texts),'internal'))
        observed['fit'] = fit if phase=='dev' else np.zeros(len(tb),bool)
        observed['cal'] = cal if phase=='dev' else np.zeros(len(tb),bool)
        observed['held'] = scope if phase=='test' or dev is test else np.ones(len(db),bool)
        state.append((phase,observed))
    signed_path = R32/'calibrated_rules'/rotation/'predictions.npz'
    report = json_at(signed_path.with_name('results.json'))
    assert checksum(signed_path)==report['predictions_sha256']
    with np.load(signed_path) as z:
        archived = z[json_at(CHOICE)['benchmark_candidate']['option']+'_tp']
        observed = state[-1][1]
        assert np.array_equal(observed['ids'][scope],z['ids'])
        assert np.array_equal(observed['replay'][scope]>0.,archived)
    if include_oof:
        bank,meta=outside_files(rotation,specs[0],shift,cfg,coef)
        assert not np.intersect1d(bank['ids'],state[0][1]['ids']).size
        assert not np.intersect1d(bank['ids'],state[-1][1]['ids']).size
        state.append(('oof',bank))
    else:
        meta={}
    signatures={}
    for phase,data in state:
        out=path/f'{phase}.npz'
        np.savez_compressed(out,**data)
        signatures[phase]=checksum(out)
        if phase!='oof':
            flips=(data['queried']>0.)!=(data['unqueried']>0.)
            beneficial=flips & ((data['queried']>0.)==data['y'])
            print(rotation,phase,len(flips),int(data['old'].sum()),int(flips.sum()),int(beneficial.sum()),flush=True)
    write_json(path/'receipt.json',dict(rotation=rotation,phases=signatures,source=report['predictions_sha256'],
               exact_replay=True,cutoff=specs[0]['cutoff'],code_sha256=checksum(HERE/'points33.py'),**meta))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    p.add_argument('--with-oof',action='store_true')
    args=p.parse_args()
    collect(args.rotation,args.with_oof)
