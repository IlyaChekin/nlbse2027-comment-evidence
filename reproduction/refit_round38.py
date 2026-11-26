"""Refit the selected Round 38 correction from signed numerical inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

OFFSET = 0.75
PENALTY = 18.0
PAIR_PENALTY = 0.55
DISTILL = 0.3
REFIT_ATOL = 2e-5

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


def strict_cut(yp, old, q, how):
    if how == 'fixed':
        return 0.
    if how == 'fp':
        n = int((old & (yp == 0)).sum())
        if not len(q):
            return 0.
        values = np.sort(q[yp == 0])[::-1]
        return float(values[n]) if n < len(values) else float(np.nextafter(q.min(), -np.inf))
    n = int((old & (yp == 1)).sum())
    if not n:
        return float(max(0., np.max(q, initial=0.)))
    selected = np.sort(q[yp == 1])[::-1]
    return float(np.nextafter(selected[n-1], -np.inf))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(__file__).with_name('round38_refit_inputs.npz'))
    args = p.parse_args()
    receipt = json.loads(args.input.with_name('refit_manifest.json').read_text())
    with args.input.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['input_sha256']
    policy = receipt['policy']
    with np.load(args.input) as z:
        model = fit(z['x'], z['margin'], z['y'], z['positive'], z['negative'])
        np.testing.assert_allclose(model['coef'], z['reference_coef'], atol=REFIT_ATOL, rtol=0.)
        np.testing.assert_allclose(model['intercept'], z['reference_intercept'], atol=REFIT_ATOL, rtol=0.)
        for name in ('cal', 'held'):
            before = z[name+'_margin']
            w = policy['mix']
            rank = (1. - w)*before + w*predict(model, z[name+'_x'], before)
            if name == 'cal':
                threshold = strict_cut(z[name+'_y'], before>0., rank, policy['rule'])
                np.testing.assert_allclose(threshold, policy['cut'], atol=REFIT_ATOL, rtol=0.)
            # Recalibration keeps a TP exactly at the strict cutoff stable across BLAS builds.
            assert np.array_equal(rank > threshold, z[name+'_expect'])
        print(f"Refit {len(z['y']):,} rows and {len(z['positive']):,} pairs. Calibration and review decisions match exactly, with fitted coefficients within {REFIT_ATOL:g}.")


if __name__ == '__main__':
    main()
