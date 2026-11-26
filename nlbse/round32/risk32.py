"""Estimate whether a positive auxiliary result is trustworthy."""
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

STYLES = ('core','shape','first','interactions','tree_shape','tree_first','tree_full')


def design(values, size, first, raw=None):
    z = logit(values.astype(np.float64).clip(1e-5,1.-1e-5))
    main, expect, no, yes = np.clip(z,-10.,10.).T
    core = np.column_stack((main,yes,expect,no))
    length = np.log1p(size)-6.
    geom = np.column_stack((core, length, main-yes, yes-no, np.minimum(main,yes),
                            np.abs(main-yes), length*yes))
    index, number = first.T
    tag = (index[:,None]==np.arange(1,21)[None,:]).astype(np.float64)
    freq = np.log1p(np.clip(number,0,64))
    compact = np.column_stack((geom,tag,freq))
    depends = np.column_stack((compact,tag*yes[:,None],tag*expect[:,None],tag*length[:,None]))
    branch = np.column_stack((geom,index,freq))
    features = dict(core=core,shape=geom,first=compact,interactions=depends,
                    tree_shape=geom,tree_first=branch)
    if raw is not None:
        full = np.log1p(raw.clip(0,64).astype(np.float64))
        features['tree_full'] = np.column_stack((branch,full,(full>0).sum(1)))
    return features


def fit(x,y,weight,mode):
    if mode.startswith('tree'):
        categorical = [x.shape[1]-2] if mode=='tree_first' else None
        if mode=='tree_observed':
            categorical = [x.shape[1]-1]
        if mode=='tree_full':
            categorical = [x.shape[1]-23]
        model = HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,max_features=.85,
                  l2_regularization=3.,min_samples_leaf=100,learning_rate=.06,early_stopping=False,
                  categorical_features=categorical,random_state=42)
    else:
        model = LogisticRegression(C=.4,solver='liblinear',max_iter=260,random_state=42)
    return model.fit(x,y,sample_weight=weight)
