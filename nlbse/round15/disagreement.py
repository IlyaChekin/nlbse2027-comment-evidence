"""Check whether teacher complementarity can be routed from cheap text and student signals."""

import argparse
import csv
import json

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor

from crossfit import HERE, ROOT, checksum, write_json
from evaluate import inputs, raw, counts, exact_budget
from precision import summary


OUT=HERE/"disagreement"
FRACTIONS=(.02,.05,.10,.25,.50,1.)


def mix(a,b,coefficient):
    to_logit=lambda p:logit(p.clip(1e-6,1-1e-6))
    return expit((1-coefficient)*to_logit(a)+coefficient*to_logit(b))


def describe(text):
    return (len(text),text.count("\n"),text.count("//"),text.count("/*"),text.count("#"),
            text.count("("),text.count("{"),text.count(";"),text.count('"'))


def load(part):
    texts,labels,_,_,roles=inputs()
    ids=roles[part]
    shape=np.asarray([describe(texts[i]) for i in ids],dtype=np.float32)
    proxies=np.maximum(1,np.minimum(shape[:,0],2200))
    p=raw(part)
    x=np.column_stack((logit(p["compact"].clip(1e-6,1-1e-6)),np.log1p(shape)))
    return ids,labels[ids,0],p["compact"],p["unix"],x,proxies


def rank(mode,scorer,features,base,threshold,proxy):
    if mode=="loss_gain":
        return scorer.predict(features) / np.sqrt(proxy)
    return -np.abs(logit(base.clip(1e-6,1-1e-6))-logit(threshold))


def cut_fraction(scores,cost,fraction):
    if fraction==1.:
        return -1e20
    order=np.argsort(-scores,kind="stable")
    cumulative=np.cumsum(cost[order])/cost.sum()
    return float(scores[order[min(np.searchsorted(cumulative,fraction,side="right"),len(order)-1)]])


def apply(a,b,score,rule):
    active=score>rule["eligibility_cutoff"]
    return np.where(active,mix(a,b,rule["teacher_weight"]),a),active


def select():
    OUT.mkdir(exist_ok=True)
    _,y,c,u,x,_=load("probe")
    average=mix(c,u,.55)
    # Learn whether a teacher query reduces per-function log loss, including cases it harms.
    correct_likelihood=lambda p: np.where(y,p,1-p).clip(1e-5,1.)
    gain=np.log(correct_likelihood(average))-np.log(correct_likelihood(c))
    gain=np.clip(gain,-2.5,2.5)
    model=HistGradientBoostingRegressor(max_leaf_nodes=9,learning_rate=.07, max_iter=130,
                min_samples_leaf=120,l2_regularization=2.,random_state=42).fit(x,gain)
    _,y,c,u,x,cost=load("dev")
    threshold=json.loads((ROOT/"round13/study/policy.json").read_text())["compact_threshold"]
    budget=counts(y,c>threshold)["fp"]
    b,_=exact_budget(c,y,budget)
    target=(y==0)&(c>.9)&(u>.9)
    candidates={}
    for mode in ("uncertainty","loss_gain"):
        ordered=rank(mode,model,x,c,threshold,cost)
        for fraction in FRACTIONS:
            cut=cut_fraction(ordered,cost,fraction)
            records=[]
            for weight in (.3,.55,.8,1.):
                rule=dict(mode=mode,eligibility_cutoff=cut,teacher_weight=weight,fraction=fraction)
                vec,routed=apply(c,u,ordered,rule)
                updated,cutoff=exact_budget(vec,y,budget)
                # A separate frozen threshold is calibrated on each development policy.
                records.append(dict(**rule,threshold=cutoff,development=summary(y,b,updated,target),
                                    actual_fraction=float(cost[routed].sum()/cost.sum()),dev_routed=int(routed.sum())))
            key=f"{mode}/{fraction:g}"
            candidates[key]=max(records,key=lambda r:(r["development"]["delta_tp"],-r["teacher_weight"]))
            print(key,candidates[key]["development"]["delta_tp"],flush=True)
    path=OUT/"regressor.joblib"
    joblib.dump(model,path,compress=3)
    write_json(OUT/"policy.json",dict(reference_threshold=threshold,development_budget=budget,models=candidates,
               proxy="length in characters capped at 2200",regressor_sha256=checksum(path)))


def test():
    cfg=json.loads((OUT/"policy.json").read_text())
    artifact=OUT/"regressor.joblib"
    if checksum(artifact)!=cfg["regressor_sha256"]:
        raise ValueError("Teacher routing model changed")
    scorer=joblib.load(artifact)
    ids,y,c,u,x,cost=load("test")
    threshold=cfg["reference_threshold"]
    reference=c>threshold
    budget=counts(y,reference)["fp"]
    base,_=exact_budget(c,y,budget)
    target=(y==0)&(c>.9)&(u>.9)
    output={}
    orders={mode:rank(mode,scorer,x,c,threshold,cost) for mode in ("uncertainty","loss_gain")}
    for key,rule in cfg["models"].items():
        q,routed=apply(c,u,orders[rule["mode"]],rule)
        updated,_=exact_budget(q,y,budget)
        frozen=q>rule["threshold"]
        output[key]=dict(exact=summary(y,base,updated,target),frozen=summary(y,reference,frozen,target),
                         routed=int(routed.sum()),proxy_share=float(cost[routed].sum()/cost.sum()))
        with (OUT/f"{key.replace('/','_')}.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","label","p_compact","p_unix","p_fused","routed","ref_equal","exact_fp","frozen"))
            w.writerows(zip(ids,y,c,u,q,routed.astype(int),base.astype(int),updated.astype(int),frozen.astype(int),strict=True))
        print(key,output[key]["exact"]["delta_tp"],output[key]["frozen"]["delta_fp"],output[key]["routed"],flush=True)
    write_json(OUT/"results.json",output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("tune","test"))
    select() if p.parse_args().stage=="tune" else test()
