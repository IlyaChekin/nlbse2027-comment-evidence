"""Measure the inference cost of a conditional adviser on held-back internal files."""
import csv
from itertools import product
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from adviser import joint,changed,loss
from scope_eval import HERE,ROOT,REFERENCE,f,checksum,json_at,write_json,baseline
from project_eval import raw_values
from students import load_selected,scores

OUT=HERE/"adviser"/"official"
FAMILY="unixcoder9"


def variables(text,frame,b):
    shape=np.asarray([[len(s),s.count("\n"),s.count("("),s.count(";"),s.count("{"),s.count("[")]
                      for s in text],dtype=np.float32)
    return np.column_stack((f.features(frame[REFERENCE],"negative"),np.log1p(shape))),shape[:,0].clip(20,2048)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    _,_,text,y,ids,_=f.dataset(full=True)
    ids=np.asarray(ids)
    origin=ROOT/"round10/runs"/FAMILY
    setup=json_at(origin/"selected.json")
    archive=origin/f"{Path(setup['checkpoint']).stem}_holdout.npz"
    with np.load(archive) as z,np.load(ROOT/"round8/frozen/holdout.npz") as split:
        if not np.array_equal(z["ids"],ids) or not np.array_equal(y,z["y"]) or not np.array_equal(z["fold"],split["folds"]):
            raise ValueError("Published internal rows and archived adviser rows differ")
        advice=z["p"][:,0]
        groups=z["fold"]
    policy=ROOT/"round20/factor_numeric/official"
    data=json_at(policy/"policy.json")
    config=data["modes"][data["selected"]]
    op=f.Policy(**config["policy"])
    values,mat,_=f.factor_data("official",text,f.small.tokenizer())
    raw=raw_values(ids)
    frame={REFERENCE:values,"raw":raw}
    base=baseline(frame,joblib.load(policy/"heads.joblib"),config["gate"],policy=op)
    train,cal,review=groups<3,groups==3,groups==4
    eligible=f.valid(raw)&~f.weak(raw)
    pool=train&eligible
    x=joint(frame,advice)
    truth=y[:,0]
    adviser=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(x[pool],truth[pool])
    q=adviser.predict_proba(x)[:,1]
    oof=np.full(len(ids),np.nan)
    for i in range(3):
        unseen=pool&(groups==i)
        known=pool&~unseen
        fit=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(x[known],truth[known])
        oof[unseen]=fit.predict_proba(x[unseen])[:,1]
    v,length=variables(text,frame,base)
    target=(loss(base["no"][pool],truth[pool])-loss(oof[pool],truth[pool])).clip(-2.5,2.5)
    router=HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=13,min_samples_leaf=80,
               l2_regularization=2.,random_state=42).fit(v[pool],target)
    joblib.dump(dict(adviser=adviser,router=router),OUT/"models.joblib",compress=3)
    threshold=config["threshold"]
    old=base["p"]>threshold
    budget=int((old&cal&(truth==0)).sum())
    teacher=load_selected(FAMILY)[:2]
    profiles={}
    live={}
    records={}
    for mode,fraction in product(("closest","expected"),(.01,.025,.05,.10)):
        at=time.perf_counter()
        priority=(1/(.035+np.abs(base["p"]-threshold)) if mode=="closest" else router.predict(v))/np.sqrt(length)
        eligible_cal=np.flatnonzero(cal&base["negative"])
        order=eligible_cal[np.argsort(-priority[eligible_cal],kind="stable")]
        n=int(np.searchsorted(np.cumsum(length[order]),fraction*length[cal].sum(),side="right"))
        limit=float(priority[order[n]]) if n<len(order) else -1e6
        route=(priority>limit)&base["negative"]
        prep=(time.perf_counter()-at)*float(cal.mean())
        columns=np.flatnonzero(route&cal)
        sample=[text[i] for i in columns]
        if not sample:
            continue
        if len(columns) and sample!=live.get(mode):
            scores(*teacher,sample,FAMILY)
            times=[]
            for _ in range(3):
                exact,current=scores(*teacher,sample,FAMILY)
                times.append(current)
            if not np.allclose(exact[:,0],advice[columns],atol=.025,rtol=0):
                raise ValueError("Measured adviser scores differ from the archived probabilities")
            profiles[mode]={k:median(v[k] for v in times) for k in times[0]}
            live[mode]=sample
        costs=profiles[mode]
        for weight in (.5,1.):
            p=changed(base,q,route,weight)
            estimated,cut=f.strict_budget(p[cal],truth[cal],budget)
            dev=f.comparison(truth[cal],old[cal],estimated)
            rev=f.comparison(truth[review],old[review],p[review]>cut)
            quality=.3*(dev["f1_after"]-dev["f1_before"])
            scale=f.PUBLIC_N/cal.sum()
            expense=.2*scale*((costs["forward_seconds"]+prep)/1000 + costs["analytic_gflops"]/3e6)
            name=f"{mode}_{fraction:g}_{weight:g}"
            records[name]=dict(quality=quality,measured_costs=costs,proxy_cost=expense,
                projected_delta_score=quality-expense,cal=dev,review=rev,route_cut=limit,
                decision_cutoff=cut,cal_calls=len(columns),review_calls=int((route&review).sum()))
            with (OUT/f"{name}_internal.csv").open("w",newline="") as stream:
                w=csv.writer(stream)
                w.writerow(("row","fold","gold","original","proposal","request"))
                w.writerows(zip(ids[cal|review],groups[cal|review],truth[cal|review],base["p"][cal|review],
                                p[cal|review],route[cal|review].astype(int),strict=True))
            print(name,"TP",dev["delta_tp"],"score",quality-expense,"calls",len(columns),flush=True)
    best=max(records,key=lambda k:records[k]["projected_delta_score"])
    choice=best if records[best]["projected_delta_score"]>0 else REFERENCE
    write_json(OUT/"policy.json",dict(selected=choice,best_variant=best,options=records,calibration_rows=int(cal.sum()),
               weights_sha256=setup["settings"]["sha256"],holdout_sha256=checksum(archive),
               model_sha256=checksum(OUT/"models.joblib"),source_policy_sha256=checksum(policy/"policy.json"),
               elapsed=time.perf_counter()-started))


if __name__=="__main__":
    main()
