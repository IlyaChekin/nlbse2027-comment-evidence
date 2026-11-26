"""Learn a precision-aware readout on the two views that the released gate already computes."""
import argparse
import json
from statistics import median
import time

import joblib
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rotate import HERE,ROOT,checksum
from evaluate import counts,exact_budget,summarize_row
import efficient_aug as previous
from fast_mat import infer
from normalize_code import batch as canonicalize
from pipeline import PUBLIC,load,metric,write_json
from students import dataset,group_codes,predict_bits,save
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT=HERE/"fusion"


def settings():
    policy=ROOT/"round12/efficient_aug/policy.json"
    return json.loads(policy.read_text())["selected"]["selective"]


def two_views(pipe,texts):
    start=time.perf_counter()
    first,cost=infer(pipe,texts,probabilities=True)
    r=settings()
    chosen=(first[:,0]>r["lower"])&(first[:,0]<r["upper"])
    local=[text for text,flag in zip(texts,chosen,strict=True) if flag]
    q=np.empty(len(local),dtype=np.float32)
    if local:
        t=time.perf_counter()
        normalized=canonicalize(local)
        spent=time.perf_counter()-t
        second,over=infer(pipe,normalized,probabilities=True)
        q=second[:,0]
        cost["forward_seconds"]+=over["forward_seconds"]+spent
        cost["analytic_gflops"]+=over["analytic_gflops"]
    base=first.copy()
    base[chosen,0]=.5*(first[chosen,0]+q)
    cost.update(wall_seconds=time.perf_counter()-start,routed=int(chosen.sum()))
    return base,first[chosen,0],q,chosen,local,cost


def features(a,b,texts,mode):
    x,y=(logit(z.clip(1e-6,1-1e-6)) for z in (a,b))
    if mode=="linear":
        return np.column_stack((x,y))
    chars=np.log1p([len(t) for t in texts])
    return np.column_stack((x,y,np.abs(x-y),chars,x*chars,y*chars))


def fit():
    _,_,texts,labels,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if f["rows"].tolist()!=ids:
            raise ValueError("Official holdout order differs")
        folds=f["folds"]
    base,a,b,routed,local,_=two_views(previous.startup(),texts)
    base=base[:,0]>settings()["threshold"]
    split=folds[routed]
    gold=labels[routed,0]
    fit,cal,review=split<=2,split==3,split==4
    budget=counts(gold[cal],base[routed][cal])["fp"]
    orig=(a+b)/2
    base_rank,_=exact_budget(orig[cal],gold[cal],budget)
    records=[]
    weights={}
    for name in ("linear","interaction"):
        x=features(a,b,local,name)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.08,max_iter=350)).fit(x[fit],gold[fit])
        weights[name]=model
        p=model.predict_proba(x)[:,1]
        for mix in (.25,.5,.75,1.):
            q=(1-mix)*orig+mix*p
            decisions,threshold=exact_budget(q[cal],gold[cal],budget)
            delta=summarize_row(gold[cal],base_rank,decisions,np.zeros(int(cal.sum()),dtype=bool))
            verify=summarize_row(gold[review],base[routed][review],q[review]>threshold,np.zeros(int(review.sum()),dtype=bool))
            records.append(dict(mode=name,mix=mix,threshold=threshold,development=delta,review=verify))
    selected=max(records,key=lambda r:(r["development"]["delta_tp"],-r["mix"]))
    OUT.mkdir(parents=True,exist_ok=True)
    w=OUT/"head.joblib"
    joblib.dump(weights[selected["mode"]],w)
    write_json(OUT/"policy.json",dict(selected=selected,choices=records,sha256=checksum(w),
                   routed_fit=int(fit.sum()),routed_cal=int(cal.sum()),routed_review=int(review.sum())))
    print(selected,flush=True)


def run(pipe,texts,head=None,recipe=None):
    started=time.perf_counter()
    out,a,b,mask,local,cost=two_views(pipe,texts)
    base=out>(settings()["threshold"],.5)
    t=time.perf_counter()
    if head is not None and mask.any():
        q=head.predict_proba(features(a,b,local,recipe["mode"]))[:,1]
        score=(1-recipe["mix"])*.5*(a+b)+recipe["mix"]*q
        base[mask,0]=score>recipe["threshold"]
    cost["forward_seconds"]+=time.perf_counter()-t
    cost["wall_seconds"]=time.perf_counter()-started
    return base,cost


def public():
    path=OUT/"head.joblib"
    config=json.loads((OUT/"policy.json").read_text())
    if checksum(path)!=config["sha256"]:
        raise ValueError("The fusion checkpoint changed")
    model=joblib.load(path)
    rule=config["selected"]
    pipe=previous.startup()
    texts,y,m=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    group=group_codes(zip(m["Projectname"],m["Filepath"],strict=True))
    policy=settings()
    base=predict_bits(ROOT/"round12/efficient_aug/selective/predictions.csv",ids,{"thresholds":(policy["threshold"],.5)})
    all_costs={name:[] for name in ("baseline","learned")}; p={}
    run(pipe,texts[:2048])
    run(pipe,texts[:2048],model,rule)
    for i in range(3):
        for name in all_costs if i%2==0 else reversed(all_costs):
            bits,profile=run(pipe,texts,model if name=="learned" else None,rule)
            if name=="baseline" and not np.array_equal(bits,base):
                raise ValueError("The new runner differs from the frozen predictions")
            if name in p and not np.array_equal(bits,p[name]):
                raise ValueError("A fused readout changed across runs")
            p[name]=bits
            all_costs[name].append(profile)
            print(name,profile,flush=True)
    result={}
    for name,obs in all_costs.items():
        timing={k:median(r[k] for r in obs) for k in obs[0]}
        bits=p[name]
        summary=metric(y,bits)
        result[name]=dict(metrics=summary,cost=timing,trials=obs,counts=counts(y[:,0],bits[:,0]),
            score=competition_formula(summary["mean"]["f1"],timing["forward_seconds"],timing["analytic_gflops"]),
            interval=bootstrap_delta(y,base,bits,group))
        save(OUT/name,y,ids,np.zeros(len(y),dtype=bool),bits,timing,(.5,.5))
    write_json(OUT/"analysis.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","public"))
    stage=parser.parse_args().stage
    fit() if stage=="fit" else public()
