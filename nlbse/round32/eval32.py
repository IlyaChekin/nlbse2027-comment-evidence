"""Compare first-hit and full-rule heads at identical held-out TP and FP budgets."""
import argparse
from itertools import product

import joblib
import numpy as np

from context32 import (HERE, ROOT, PARTS, f, checksum, json_at, write_json, signed_source, frames, raw_values,
                       early_out, rule_order, strict_cut)
from risk32 import STYLES, design, fit
from agreement32 import OPTIONS, features as agreement, inputs as aux_cache

WIDTHS = (.8,1.8,3.8,None)
MIXES = (.4,.8)
DEST = HERE/'evaluation'


def training(rotation, threshold, extra):
    source, fingerprint = signed_source(rotation)
    saved = ROOT/'round26/compact'/rotation
    meta = json_at(saved/'manifest.json')
    path = saved/'scores.npz'
    if meta['sparse_oof_sha256']!=fingerprint or checksum(path)!=meta['sha256']:
        raise ValueError('Independent compact scores changed')
    corpus, _, project, _, _ = f.inputs()
    mask = source['weak']==1
    ids = source['ids'][mask]
    raw = raw_values(ids)
    interim = early_out(raw)
    if not (interim[:,0]>0).all():
        raise ValueError('Signed positive examples disagree with rule cache')
    limits = np.array([len(corpus[i]) for i in ids])
    with np.load(path) as z:
        if not all(np.array_equal(z[name],source[name]) for name in ('ids','fold','y')):
            raise ValueError('Outside-file rows or folds do not align')
        data = design(z['values'][mask],limits,interim,raw)
    styles = STYLES
    aux = None
    if extra:
        aux, signature = aux_cache(rotation)
        if not np.array_equal(aux['ids'],ids) or not np.array_equal(aux['fold'],source['fold'][mask]):
            raise ValueError('Rule predictions cover different outside-file rows')
        data.update(agreement(data['shape'],interim,aux['rules']))
        styles = OPTIONS
    y = source['y'][mask]
    folds = source['fold'][mask]
    anchor = np.empty(len(y))
    for fold in np.unique(folds):
        visible = data['core'][folds!=fold]
        weight = np.ones(len(visible))
        model = fit(visible,y[folds!=fold],weight,'core')
        anchor[folds==fold] = model.decision_function(data['core'][folds==fold])
    cutoff = np.log(threshold/(1.-threshold))
    weight = .25+2.*np.exp(-.7*abs(anchor-cutoff))
    head = {name:fit(data[name],y,weight,name) for name in styles}
    fn = dict(rows=len(y),positive=int(y.sum()),token_oof_sha256=fingerprint,compact_oof_sha256=meta['sha256'],
              fit_families=np.unique(project[ids]).tolist(),rule_order=rule_order()[0],
              widths=WIDTHS,mixes=MIXES)
    if aux is not None:
        fn.update(aux_sha256=signature)
    return head,fn,aux


def inputs(ctx,phase,aux):
    state = ctx.state
    mask = state['known'] & state['flag']
    size = np.fromiter((len(text) for text in ctx.texts),dtype=np.int32)
    raw = ctx.frame['raw'][mask]
    j = early_out(raw)
    if not (j[:,0]>0).all():
        raise ValueError('Cached gate and rule positives disagree')
    data = design(ctx.frame['reference'][mask],size[mask],j,raw)
    if aux is not None:
        if not np.array_equal(aux[phase+'_ids'],ctx.frame['ids'][mask]):
            raise ValueError('Saved rule predictions do not align with the reference')
        data = agreement(data['shape'],j,aux[phase])
    return mask,data


def cut_for(y, signed, values, habit):
    if not len(y):
        return 0.
    return strict_cut(y,signed,values,habit)


def compare(y, baseline, scope, rank, cutoff):
    pred = baseline.copy()
    pred[scope] = rank>cutoff
    return f.comparison(y,baseline,pred),pred


def per_project(y, groups, baseline, scope, rank, habits):
    lookup = np.full(len(y),np.nan)
    lookup[scope] = rank
    rows, results = {}, {}
    for habit in habits:
        pred = baseline.copy()
        detail = {}
        for name in np.unique(groups):
            sub = groups==name
            active = sub & scope
            t = cut_for(y[active],baseline[active],lookup[active],habit)
            pred[active] = lookup[active]>t
            diff = f.comparison(y[sub],baseline[sub],pred[sub])
            if diff['delta_'+habit]!=0:
                raise ValueError(f'Tied ranking prevents identical {habit} on {name}')
            detail[name] = diff
        assert np.array_equal(pred[~scope],baseline[~scope])
        results[habit] = dict(change=f.comparison(y,baseline,pred),projects=detail)
        rows[habit] = pred
    return results,rows


def evaluate(rotation,extra=False):
    dev,test,cal,review,(p,q),threshold = frames(rotation)
    old,base = p>0.,q>0.
    y0,y1 = dev.frame['y'].astype(bool),test.frame['y'].astype(bool)
    models, notes, aux = training(rotation,threshold,extra)
    if test is not dev and set(notes['fit_families']) & set(test.frame['project']):
        raise ValueError('Held-out project present in outside-file supervision')
    scope,x = inputs(dev,'dev',aux)
    active,z = (scope,x) if test is dev else inputs(test,'test',aux)
    save = dict(ids=test.frame['ids'][review],truth=y1[review],baseline=base[review],
                available=active[review],original_rank=q[review])
    ideal = ('tp','fp')
    stats, cal_by_width = {}, {}
    dest = (HERE/'agreement' if extra else DEST)/rotation
    dest.mkdir(parents=True,exist_ok=True)
    for width in WIDTHS:
        width_key = 'all' if width is None else f'{width:g}'
        dx,tx = scope.copy(),active.copy()
        if width is not None:
            dx &= abs(p)<width
            tx &= abs(q)<width
        if not (dx & cal).any():
            continue
        cal_by_width[width_key] = dict(cal_rows=int((dx & cal).sum()),held_rows=int((tx & review).sum()))
        for name, strength in [('threshold',0.),*product(models,MIXES)]:
            key = f'{name}_{width_key}_{strength:g}'
            first, second = p[dx],q[tx]
            if strength:
                a = models[name].decision_function(x[name][dx[scope]])
                b = a if test is dev else models[name].decision_function(z[name][tx[active]])
                first = (1.-strength)*first+strength*a
                second = (1.-strength)*second+strength*b
            options = {}
            if test is not dev:
                conditioned, rows = per_project(y1,test.frame['project'],base,tx,second,ideal)
                options['matched'] = conditioned
                for mode, result in rows.items():
                    save[key+'_matched_'+mode] = result
            rank0 = p.copy()
            rank1 = q.copy()
            rank0[dx] = first
            rank1[tx] = second
            for criterion in (*ideal,'fixed'):
                cut = cut_for(y0[dx & cal],old[dx & cal],rank0[dx & cal],criterion)
                a,b = compare(y0[cal],old[cal],dx[cal],rank0[dx & cal],cut)
                c,d = compare(y1[review],base[review],tx[review],rank1[tx & review],cut)
                options[criterion] = dict(cut=cut,cal=a,frozen=c)
                save[key+'_'+criterion] = d
            save[key+'_rank'] = rank1[review]
            options['setting'] = dict(style=name,strength=strength,width=width,publishable=name!='tree_full')
            stats[key] = options
    predictions = dest/'predictions.npz'
    weights = dest/'models.joblib'
    np.savez_compressed(predictions,**save)
    joblib.dump(models,weights,compress=3)
    result = dict(rotation=rotation,training=notes,options=stats,scope=cal_by_width,
                  predictions_sha256=checksum(predictions),weights_sha256=checksum(weights),
                  code_sha256=checksum(HERE/'eval32.py'),runtime_sha256=checksum(HERE/'risk32.py'))
    write_json(dest/'results.json',result)
    for name, row in stats.items():
        if row['setting']['width'] is None:
            print(rotation,name,row['tp']['cal']['delta_fp'],row['tp']['frozen']['delta_fp'],
                 row.get('matched',{}).get('tp',{}).get('change',{}).get('delta_fp'),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    parser.add_argument("--agreement",action="store_true")
    args = parser.parse_args()
    evaluate(args.rotation,args.agreement)
