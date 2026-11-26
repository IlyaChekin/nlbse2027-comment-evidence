"""Allocate the available scanner calls with a fixed count and nonincreasing input size."""
import numpy as np


BOUNDARIES = np.asarray([170,512,1024,2048,4096,8192,16384])


def logit(p):
    p = np.clip(p,1e-5,1.-1e-5)
    return np.log(p)-np.log1p(-p)


def matrix(pre, size, threshold):
    z = logit(pre[:,:8].astype(np.float64))
    main,weak,guess_no,guess_yes,pw,pn,py,prior = z.T
    cutoff = logit(float(threshold))
    pn,py,prior = pn-cutoff,py-cutoff,prior-cutoff
    length = np.log1p(size)
    u = np.log1p(15.*pre[:,8])
    balance = np.minimum(np.abs(pn),np.abs(py))
    balance = np.where((pn>0.)==(py>0.),-balance,balance)
    return np.column_stack((main,weak,guess_no,guess_yes,pw,pn,py,prior,length,u,balance,
                            np.abs(prior), np.abs(py-pn), (py-pn)*pre[:,4]*(1.-pre[:,4]),
                            (main-guess_no), (main-guess_yes),length*prior))


def order_scores(x):
    # Average tied ranks so equal predictions never encode input position into utility.
    if len(x)<2:
        return np.zeros(len(x))
    order = np.argsort(x,kind='stable')
    y = x[order]
    ends = np.r_[0,np.flatnonzero(y[:-1]!=y[1:])+1,len(y)]
    rank = np.empty(len(x),dtype=np.float64)
    rank[order] = np.repeat((ends[:-1]+ends[1:]-1)/(2.*(len(y)-1)),np.diff(ends))
    return rank


def reassign(old, size, raw, anchor, strength, fraction):
    if not (len(old)==len(size)==len(raw)==len(anchor)):
        raise ValueError('The router inputs differ in length')
    selected = old.copy()
    band = np.searchsorted(BOUNDARIES,size,side='right')
    sources, targets, score_delta = [], [], []
    for bucket in np.unique(band):
        ids = np.flatnonzero(band==bucket)
        keep = old[ids]
        if not keep.any() or keep.all():
            continue
        proposal = (1.-strength)*order_scores(anchor[ids])+strength*order_scores(raw[ids])
        from_old = np.flatnonzero(keep)
        from_new = np.flatnonzero(~keep)
        low = from_old[np.argsort(proposal[from_old],kind='stable')]
        high = from_new[np.argsort(-proposal[from_new],kind='stable')]
        cap = min(len(low),len(high),max(1,int(keep.sum()*fraction)))
        low,high = low[:cap],high[:cap]
        gain = proposal[high]-proposal[low]
        better = gain>1e-12
        sources.extend(ids[low[better]])
        targets.extend(ids[high[better]])
        score_delta.extend(gain[better])
    if not sources:
        return selected
    a,b = np.asarray(sources),np.asarray(targets)
    cost_delta = size[b].astype(np.int64)-size[a]
    retained = np.ones(len(a),bool)
    shortage = cost_delta.sum()
    if shortage>0:
        candidates = np.flatnonzero(cost_delta>0)
        ratios = np.asarray(score_delta)[candidates]/cost_delta[candidates]
        ordered = candidates[np.argsort(ratios,kind='stable')]
        prefix = np.cumsum(cost_delta[ordered]).searchsorted(shortage,side='left')
        retained[ordered[:prefix+1]] = False
    selected[a[retained]] = False
    selected[b[retained]] = True
    assert selected.sum()==old.sum()
    assert np.array_equal(np.bincount(band[selected],minlength=len(BOUNDARIES)+1),
                          np.bincount(band[old],minlength=len(BOUNDARIES)+1))
    assert size[selected].sum()<=size[old].sum()
    return selected


def prediction(models, mode, features):
    if mode=='classification':
        prob = models['classifier'].predict_proba(features)
        return prob @ models['class_utility']
    if mode=='separate':
        negative = features[:,7]<=0.
        result = np.zeros(len(features))
        for name, scope in [('negative',negative),('positive',~negative)]:
            if scope.any():
                result[scope] = models[name].predict(features[scope])
        return result
    if mode=='greedy':
        # Approximate whether the no/yes scenarios can cross the final threshold.
        balance = features[:,10]
        return np.maximum(0.,balance)+.05*features[:,9]
    return models[mode].predict(features)
