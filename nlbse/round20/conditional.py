"""Calibrated conditional probabilities and the exact-rule acquisition gate."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logit


@dataclass(frozen=True)
class Policy:
    mix:float
    fraction:float
    method:str

    @property
    def name(self):
        return f"mix{self.mix:g}_{self.method}_{self.fraction:g}"


def valid(raw):
    exists=(raw>0).any(1)
    return exists | (raw>=0).all(1)


def weak(raw):
    return (raw>0).any(1)


def features(values, kind):
    key={"w":[0,1],"negative":[0,2],"positive":[0,3]}[kind]
    return logit(values[:,key].clip(1e-5,1-1e-5))


def estimate(heads,values):
    return {kind:model.predict_proba(features(values,kind))[:,1]
            for kind,model in heads.items()}


def probabilities(values,est,policy):
    no=(1-policy.mix)*values[:,0]+policy.mix*est["negative"]
    yes=(1-policy.mix)*values[:,0]+policy.mix*est["positive"]
    w=est["w"]
    prior=(1-w)*no+w*yes
    return prior,no,yes,w


def query_value(policy,prior,no,yes,w,threshold):
    if policy.method=="margin":
        return np.abs(yes-no)*w*(1-w)/(.08+np.abs(prior-threshold))
    # The threshold is the false-positive tradeoff. Query value is the expected
    # increase in positive-decision utility when the exact rule outcome is known.
    after=(1-w)*np.maximum(no-threshold,0.)+w*np.maximum(yes-threshold,0.)
    return np.maximum(after-np.maximum(prior-threshold,0.),0.)


def choose(values,est,policy,gate):
    prior,no,yes,w=probabilities(values,est,policy)
    value=query_value(policy,prior,no,yes,w,gate["base"])
    return prior,no,yes,value>gate["utility"]


