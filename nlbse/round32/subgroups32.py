"""Calibrate the released margin separately for common first-hit outcomes."""
import argparse
from itertools import product

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from context32 import HERE, PARTS, f, checksum, json_at, write_json, frames, early_out, rule_order
from eval32 import cut_for, compare, per_project

WIDTHS=(.8,1.8,3.8,None)
STRENGTHS=(.35,.7,1.)
PENALTY=20.
OUT=HERE/'calibrated_rules'


def transform(ctx):
    state = ctx.state
    scope = state['known'] & state['flag']
    first = early_out(ctx.frame['raw'][scope])
    if (first[:,0]<1).any():
        raise ValueError('Candidate has a missing first rule')
    return scope,first


def features(margin, first, mode):
    tag = first[:,0]
    code = tag[:,None]==np.arange(1,21)[None,:]
    score = margin.clip(-3.,3.)
    shared = np.column_stack((np.ones(len(score)),score))
    if mode=='global':
        return shared
    if mode=='rule':
        return np.column_stack((shared,code))
    if mode=='slope':
        return np.column_stack((shared,code,code*score[:,None]))
    if mode=='counts':
        count = np.log1p(first[:,1].clip(0,64))
        return np.column_stack((shared,code,code*score[:,None],code*count[:,None]))
    raise ValueError(f'Unknown calibration form {mode}')


def calibrate(rank, first, gold, mode):
    x = features(rank,first,mode)
    w = .25+2.*np.exp(-.5*abs(rank))
    eta = np.full(x.shape[1],PENALTY)
    eta[:2] = [4.,8.]
    def loss(coef):
        u = rank+x@coef
        z = np.logaddexp(0,u)-gold*u
        g = x.T @ (w*(expit(u)-gold))
        return w@z+.5*(eta*coef)@coef, g+eta*coef
    fitted = minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                      options=dict(maxiter=160,ftol=1e-10))
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x


def evaluate(rotation):
    dev,test,cal,review,(p,q),_ = frames(rotation)
    old,base = p>0.,q>0.
    truth,test_truth = dev.frame['y'].astype(bool),test.frame['y'].astype(bool)
    if test is dev:
        fit = ~(cal | review)
    else:
        folds = json_at(f.WEAK/rotation/'policy.json')
        fit = np.isin(dev.frame['ids'],folds['fit_rows'])
        if (fit & cal).any():
            raise ValueError('Calibration and fit share examples')
    use,first = transform(dev)
    active,second = (use,first) if test is dev else transform(test)
    dy=fit[use]
    modes=('global','rule','slope','counts')
    models={mode:calibrate(p[use][dy],first[dy],truth[use][dy],mode) for mode in modes}
    z0={key:features(p[use],first,key)@model for key,model in models.items()}
    z1=z0 if test is dev else {key:features(q[active],second,key)@model for key,model in models.items()}
    outcomes={}
    outputs=dict(ids=test.frame['ids'][review],truth=test_truth[review],baseline=base[review],
                 available=active[review],original_rank=q[review])
    for width in WIDTHS:
        u,v=use.copy(),active.copy()
        if width is not None:
            u &= abs(p)<width
            v &= abs(q)<width
        if not (u & cal).any():
            continue
        for mode,strength in [('threshold',0.),*product(modes,STRENGTHS)]:
            key=f"{mode}_{'all' if width is None else f'{width:g}'}_{strength:g}"
            left,right=p.copy(),q.copy()
            if strength:
                left[u]+=strength*z0[mode][u[use]]
                right[v]+=strength*z1[mode][v[active]]
            result=dict(setting=dict(style=mode,width=width,strength=strength,publishable=True),
                        calls=dict(cal=int((u & cal).sum()),held=int((v & review).sum())))
            if test is not dev:
                result['matched'],forecast=per_project(test_truth,test.frame['project'],base,v,right[v],('tp','fp'))
                for protocol, decision in forecast.items():
                    outputs[key+'_matched_'+protocol]=decision
            for protocol in ('tp','fp','fixed'):
                t=cut_for(truth[u & cal],old[u & cal],left[u & cal],protocol)
                record,_=compare(truth[cal],old[cal],u[cal],left[u & cal],t)
                held,forecast=compare(test_truth[review],base[review],v[review],right[v & review],t)
                result[protocol]=dict(cut=t,cal=record,frozen=held)
                outputs[key+'_'+protocol]=forecast
            outcomes[key]=result
            outputs[key+'_rank']=right[review]
    folder=OUT/rotation
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictions.npz'
    np.savez_compressed(path,**outputs)
    weight=folder/'models.joblib'
    joblib.dump(models,weight,compress=3)
    write_json(folder/'results.json',dict(rotation=rotation,options=outcomes,fit_rows=int(dy.sum()),
               penalty=PENALTY,rule_order=rule_order()[0],weights_sha256=checksum(weight),
               predictions_sha256=checksum(path),code_sha256=checksum(HERE/'subgroups32.py')))
    print(rotation,'matched strata',int(dy.sum()),'options',len(outcomes),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    evaluate(parser.parse_args().rotation)
