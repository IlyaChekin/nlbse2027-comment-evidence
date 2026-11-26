"""Fit corrections for two distinct scanner errors with a shared false-positive budget."""
import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from data38 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, phases, routes, pairs, crop
from eval35 import cut, per_project, adjusted
from broad35 import features as score_features
from eval37 import features as regional_features
from context32 import early_out, raw_values
from agreement32 import inputs as aux_inputs, features as agreement, rule_order

DEST = HERE/'comparison'
ARMS = ('control','point','pair','distill','teacher','teacherpair')
CHEAP = ('control','point','pair','distill')
PENALTY = 18.
OFFSET = .75
PAIR_PENALTY = .55
DISTILL = .30
WIDTHS = ((.6,.9),(1.2,1.8),(2.5,3.8),(None,None))
STRENGTHS = (.25,.55,.85)


def fit(x, baseline, y, pos, neg, teacher=None):
    scale = StandardScaler().fit(x)
    x = np.column_stack((scale.transform(x).clip(-12.,12.),np.ones(len(x))))
    weight = .25+2.25*np.exp(-.55*abs(baseline))
    total_weight = weight.sum()
    anchor = OFFSET*np.clip(baseline,-8.,8.)
    valid = np.zeros(len(y),bool) if teacher is None else np.isfinite(teacher)
    if valid.any():
        signal = teacher[valid]
        width = max(.25,np.std(signal))
        target = expit(np.clip((signal-np.mean(signal))/width,-5.,5.)*.9)
        m = DISTILL*weight[valid]

    if len(pos):
        delta = x[pos]-x[neg]
        base_delta = anchor[pos]-anchor[neg]
        _,owner,total = np.unique(neg,return_inverse=True,return_counts=True)
        coeff = (1./total[owner])
        coeff *= PAIR_PENALTY*total_weight/coeff.sum()

    def objective(beta):
        signed = anchor+x@beta
        loss = weight@(np.logaddexp(0.,signed)-y*signed)+.5*PENALTY*(beta@beta)
        grad = x.T@(weight*(expit(signed)-y))+PENALTY*beta
        if len(pos):
            order = base_delta+delta@beta
            loss += coeff@np.logaddexp(0.,-order)
            grad -= delta.T@(coeff*expit(-order))
        if valid.any():
            value = signed[valid]
            loss += m@(np.logaddexp(0.,value)-target*value)
            grad += x[valid].T@(m*(expit(value)-target))
        return loss/total_weight,grad/total_weight

    params = minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                      options=dict(maxiter=550,ftol=2e-9,gtol=2e-5))
    gradient = float(np.max(np.abs(params.jac)))
    if not params.success and gradient>1e-4:
        raise RuntimeError(params.message)
    return dict(scale=scale,coef=params.x[:-1],intercept=params.x[-1],iters=params.nit,gradient=gradient)


def predict(model, x, margin):
    return OFFSET*np.clip(margin,-8.,8.)+model['scale'].transform(x).clip(-12.,12.)@model['coef']+model['intercept']


def positive_features(data,t,aux,first,expanded):
    control = score_features(data,t,'regime')
    count = np.log1p(first[:,1].clip(0,64))
    identity = first[:,0,None]==np.arange(1,21)[None,:]
    score = data['latest'].clip(-4.,4.)
    common = np.column_stack((control,identity,identity*score[:,None],count))
    if not expanded:
        return common
    source = agreement(control,first,aux)['observed_interactions']
    return np.column_stack((common,source[:,control.shape[1]:]))


def design_at(data,t,branch,mode):
    extended = mode!='control'
    if branch=='positive':
        x = positive_features(data,t,data['aux_positive'],data['first'],extended)
    elif extended:
        x = regional_features(data,t,'regions_specific',data['section'])
    else:
        x = score_features(data,t,'regime')
    if mode.startswith('teacher'):
        t = data['teacher1']
        x = np.column_stack((x,t,np.where(np.isfinite(t),t,0.)-data['latest'],data['teacher2']))
    return x


def installed(rotation,data,base):
    digests = {}
    for branch in ('negative','positive'):
        folder = HERE/f'{branch}_teacher'/rotation
        p = folder/'receipt.json'
        receipt = json_at(p)
        z = folder/'scores.npz'
        assert receipt['scores_sha256']==checksum(z)
        assert receipt['base_sha256']==base
        assert receipt['code_sha256']==checksum(HERE/('negative38.py' if branch=='negative' else 'teacher38.py'))
        digests[branch] = checksum(p)
        with np.load(z) as store:
            names = ('','') if branch=='negative' else ('_point','_pair')
            for role,info in data.items():
                selected = routes(info)[branch]
                assert np.array_equal(store[role+'_ids'],info['ids'][selected])
                if role=='oof':
                    assert np.array_equal(store['fold'],info['fold'][selected])
                for n,text in enumerate(names,1):
                    key = f'{branch}_teacher{n}'
                    info[key] = np.full(len(selected),np.nan,np.float32)
                    info[key][selected] = store[role+text]
    aux,sha = aux_inputs(rotation)
    for role,item in data.items():
        mask = routes(item)['positive']
        ids = aux['ids'] if role=='oof' else aux[role+'_ids']
        src = aux['rules'] if role=='oof' else aux[role]
        loc = {key:i for i,key in enumerate(ids)}
        aligned = np.asarray([loc[k] for k in item['ids'][mask]])
        assert np.array_equal(ids[aligned],item['ids'][mask])
        item['aux_positive'] = np.zeros((len(mask),src.shape[1]),np.float32)
        item['aux_positive'][mask] = src[aligned]
        item['first'] = np.zeros((len(mask),2),np.int32)
        item['first'][mask] = early_out(raw_values(item['ids'][mask]))
        assert (item['first'][mask,0]>0).all()
    return digests | {'aux':sha}


def matrix(frame,t,branch,arm):
    data = frame | {'teacher1':frame[branch+'_teacher1'],'teacher2':frame[branch+'_teacher2']}
    return design_at(data,t,branch,arm)


def scored(rotation,data,ref):
    t = ref['cutoff']
    train = data['oof']
    dev = data['dev']
    out = data.get('test',dev)
    assess = crop(dev,dev['cal']),crop(out,out['held'])
    models, scores, roster = {}, {}, {}
    for branch,name in (('negative','negative_consensus'),('positive','positive_retained')):
        support = routes(train)[branch]
        left,right,pair_sha = pairs(rotation,train,name)
        assert support[left].all() and support[right].all()
        for arm in ARMS:
            active = support.copy()
            if arm.startswith('teacher'):
                active &= np.isfinite(train[branch+'_teacher1']) & np.isfinite(train[branch+'_teacher2'])
            good = active[left]&active[right]
            pick = np.flatnonzero(active)
            pos,neg = np.searchsorted(pick,left[good]),np.searchsorted(pick,right[good])
            if arm not in ('pair','teacherpair','distill'):
                pos,neg = np.empty(0,np.int64),np.empty(0,np.int64)
            local = crop(train,active)
            x = matrix(local,t,branch,arm)
            extra = local[branch+'_teacher1'] if arm=='distill' else None
            model = fit(x,local['latest'],local['y'],pos,neg,extra)
            models[branch,arm] = model
            values = []
            for frame in assess:
                take = routes(frame)[branch]
                subset = crop(frame,take)
                q = np.full(len(take),np.nan)
                q[take] = predict(model,matrix(subset,t,branch,arm),subset['signed'])
                assert np.isfinite(q[take]).all()
                values.append(q)
            scores[branch,arm] = values
            roster[branch,arm] = dict(rows=len(local['ids']),pairs=len(pos),pairs_sha256=pair_sha,
                                     iterations=model['iters'],gradient=model['gradient'])
        print(rotation,branch,{k[1]:v['rows'] for k,v in roster.items() if k[0]==branch},flush=True)
    return models,scores,assess,{f'{k[0]}/{k[1]}':v for k,v in roster.items()}


def active_set(frame,options,widths):
    masks = routes(frame)
    out = {}
    for (branch,mode),w in zip(options,widths,strict=True):
        chosen = masks[branch]
        if mode=='off':
            chosen = np.zeros(len(chosen),bool)
        elif w is not None:
            chosen = chosen&(abs(frame['signed'])<w)
        out[branch] = chosen
    return out


def options():
    for left,right in (('negative','positive'),):
        for a,b in [*(('off',x) for x in ARMS),*((x,'off') for x in ARMS),
                    *((x,x) for x in ARMS),('pair','point'),('point','pair'),('distill','pair'),('pair','distill')]:
            yield ((left,a),(right,b))


def evaluate(rotation, cal, test, scores):
    search = {}
    forecasts = dict(ids=test['ids'],truth=test['y'],base=test['signed']>0.)
    y = [x['y'] for x in (cal,test)]
    bits = [x['signed']>0. for x in (cal,test)]
    for opt,w in product(options(),WIDTHS):
        tag = '~'.join(arm for _,arm in opt)
        key = '/'.join(str(v) for v in w)+'/'+tag
        areas = [active_set(frame,opt,w) for frame in (cal,test)]
        selected = [np.logical_or(*area.values()) for area in areas]
        # Each single- or two-branch option has a threshold-only comparator on identical rows.
        for mixing in (0.,*STRENGTHS):
            code = key+f'/{mixing:g}'
            rank = []
            for i,(frame,regions,mask) in enumerate(zip((cal,test),areas,selected,strict=True)):
                z = frame['signed'].copy()
                if mixing:
                    for branch,method in opt:
                        active = regions[branch]
                        if active.any():
                            other = scores[branch,method][i]
                            z[active] = (1.-mixing)*z[active] + mixing*other[active]
                rank.append(z[mask])
            if not selected[0].any() or not selected[1].any():
                continue
            report = dict(arms=dict(opt),widths=w,mix=mixing,cal_rows=int(selected[0].sum()),
                          held_rows=int(selected[1].sum()),branches={k:int(v.sum()) for k,v in areas[1].items()})
            if rotation!='official':
                budget,z = per_project(y[1],test['project'],bits[1],selected[1],rank[1],('fp','tp'))
                report['matched'] = budget
                forecasts[code+'_budget_fp'] = z['fp']
            for rule in ('fixed','fp','tp'):
                t = cut(y[0][selected[0]],bits[0][selected[0]],rank[0],rule)
                one,_ = adjusted(y[0],bits[0],selected[0],rank[0],t)
                two,z = adjusted(y[1],bits[1],selected[1],rank[1],t)
                report[rule] = dict(cut=t,cal=one,held=two)
                forecasts[code+'_'+rule] = z
            search[code] = report
    return search,forecasts


def main(rotation):
    data,ref,base = phases(rotation)
    sources = installed(rotation,data,base)
    models,values,(cal,test),counts = scored(rotation,data,ref)
    experiment,pred = evaluate(rotation,cal,test,values)
    d = DEST/rotation
    d.mkdir(exist_ok=True,parents=True)
    position = d/'predictions.npz'
    np.savez_compressed(position,**pred)
    readout = d/'models.joblib'
    joblib.dump(models,readout,compress=3)
    write_json(d/'results.json',dict(rotation=rotation,base_sha256=base,sources=sources,
                      predictions_sha256=checksum(position),weights_sha256=checksum(readout),
                      fits=counts,options=experiment,code_sha256=checksum(Path(__file__))))
    print(rotation,len(experiment),'options',flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
