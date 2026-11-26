"""Transfer the short specialist to official development and profile the frozen decision rule."""
import argparse
import json
from statistics import median
import time

import joblib
import numpy as np
from scipy.special import expit, logit
from scipy.sparse import load_npz, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rotate import HERE, ROOT, PLAN, checksum, read_partition
import bindings
from evaluate import adjusted, counts, exact_budget, matrix, summarize_row
import efficient_aug as previous
from pipeline import TRAIN, PUBLIC, load, select_indices, write_json, metric
from students import dataset, group_codes, predict_bits, save
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT=HERE/"official"
BASE=ROOT/"round12"
NEW=("bindings","grammar_bindings","token_bindings","interface","interface_bindings")


def read(path):
    return json.loads(path.read_text())


def recipe():
    return read(BASE/"efficient_aug/policy.json")["selected"]["selective"]


def extract_official():
    texts,y,_=load(TRAIN)
    fit,internal=select_indices(False)
    cutoff=read(PLAN)["short_chars_max"]
    need=np.asarray([i for i,text in enumerate(texts) if len(text)<=cutoff],dtype=np.int32)
    ids=[]
    sparse={n:[] for n in bindings.SOURCES}
    for role in read_partition():
        with np.load(bindings.OUT/f"{role}.npz") as f:
            ids.extend(f["ids"].tolist())
        for n in sparse:
            sparse[n].append(load_npz(bindings.OUT/f"{role}_{n}.npz"))
    missing=sorted(set(need.tolist())-set(ids))
    if missing:
        x=bindings.transform([texts[i] for i in missing],workers=3)
        for n in sparse:
            sparse[n].append(x[n])
        ids.extend(missing)
    order=np.argsort(ids)
    if not np.array_equal(np.asarray(ids)[order],need):
        raise ValueError("Cached short inputs do not cover the official train")
    return need,{k:vstack(m,format="csr")[order] for k,m in sparse.items()},y,fit,internal


def train_features():
    ids,x,y,fit,internal=extract_official()
    first=np.isin(ids,fit)
    second=np.isin(ids,internal)
    fit_bundle=bindings.fit_bundle({k:v[first] for k,v in x.items()},y[ids[first],0])
    OUT.mkdir(parents=True,exist_ok=True)
    artifact=OUT/"features.joblib"
    joblib.dump(fit_bundle,artifact,compress=3)
    rows=ids[second]
    p={name:model.predict_proba(bindings.chosen_inputs({k:v[second] for k,v in x.items()},name))[:,1]
       for name,model in fit_bundle.items()}
    np.savez_compressed(OUT/"holdout_features.npz",ids=rows,**p)
    write_json(OUT/"features.json",dict(sha256=checksum(artifact),official_fit_short=int(first.sum()),
                                     official_internal_short=int(second.sum())))


def tune():
    _,_,texts,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if f["rows"].tolist()!=ids:
            raise ValueError("Original calibration files no longer align")
        folds=f["folds"]
    head_fit,cal,review=folds<=2,folds==3,folds==4
    pipe=previous.startup()
    base,_=previous.run(pipe,texts,recipe())
    with np.load(OUT/"holdout_features.npz") as f:
        if not np.all(np.isin(f["ids"],ids)):
            raise ValueError("Official feature subset no longer aligns")
        loc=np.searchsorted(ids,f["ids"])
        shape={n:np.full(len(y),.5) for n in bindings.MODELS}
        for n,vec in shape.items():
            vec[loc]=f[n]
    p=base[:,0]
    old=p>recipe()["threshold"]
    short=np.zeros(len(y),dtype=bool)
    short[loc]=True
    confident=short&(p<.05)
    budget=counts(y[cal,0],old[cal])["fp"]
    rank_base,_=exact_budget(p[cal],y[cal,0],budget)
    candidates=[]
    suite={}
    for name in (*NEW,"token","logits"):
        train=head_fit&confident
        x=matrix(name,{"compact":p},shape,"compact_low")
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.25,max_iter=250)).fit(x[train],y[train,0])
        suite[name]=clf
        q=clf.predict_proba(x[confident])[:,1]
        for boost in (0.,.7,1.4,2.1,2.8,3.5,4.2):
            for strength in (.5,.75,1.):
                score=adjusted(p,confident,q,strength,boost)
                test,cutoff=exact_budget(score[cal],y[cal,0],budget)
                rr=summarize_row(y[cal,0],rank_base,test,confident[cal])
                rec=summarize_row(y[review,0],old[review],score[review]>cutoff,confident[review])
                candidates.append(dict(method=name,boost=boost,strength=strength,cutoff=cutoff,development=rr,review=rec))
    selected=max((c for c in candidates if c["method"] in NEW),key=lambda c:(c["development"]["delta_tp"],
              c["development"]["repaired_in_target"],-c["boost"],-c["strength"]))
    length_rows=[]
    for shift in (0.,-.9,-.6,-.35,-.15,.15,.35,.6,.9):
        q=length_shift(p,short,shift)
        match,t=exact_budget(q[cal],y[cal,0],budget)
        length_rows.append(dict(method="length",shift=shift,cutoff=t,
                development=summarize_row(y[cal,0],rank_base,match,confident[cal]),
                review=summarize_row(y[review,0],old[review],q[review]>t,confident[review])))
    calibrated_length=max(length_rows,key=lambda z:(z["development"]["delta_tp"],-abs(z["shift"])))
    artifact=OUT/"heads.joblib"
    joblib.dump(suite,artifact)
    write_json(OUT/"policy.json",dict(selected=selected,length_control=calibrated_length,length_candidates=length_rows,candidates=candidates,sha256=checksum(artifact),
              official_short_high_confidence=int(confident.sum()),development_fp_limit=budget,calibration_labels=int(cal.sum())))
    print(selected,flush=True)


def length_shift(p,active,shift):
    q=p.astype(np.float64,copy=True)
    q[active]=expit(logit(p[active].clip(1e-7,1-1e-7))+shift)
    return q


def adjusted_pipeline(pipe,texts,policy=None,models=None,head=None):
    started=time.perf_counter()
    p,c=previous.run(pipe,texts,recipe())
    if policy is None:
        c.update(feature_seconds=0.,specialist_rows=0)
        return p,c
    cutoff=read(PLAN)["short_chars_max"]
    begin=time.perf_counter()
    short=np.fromiter(map(len,texts),dtype=np.int32,count=len(texts))<=cutoff
    p=p.astype(np.float64)
    if policy["method"]=="length":
        p[:,0]=length_shift(p[:,0],short,policy["shift"])
        elapsed=time.perf_counter()-begin
        c["forward_seconds"]+=elapsed
        c.update(wall_seconds=time.perf_counter()-started,feature_seconds=elapsed,specialist_rows=int(short.sum()))
        return p,c
    mask=(p[:,0]<.05)&short
    if mask.any():
        blocks=bindings.transform([t for t,b in zip(texts,mask,strict=True) if b])
        need=("token","bindings") if policy["method"]=="token_bindings" else (policy["method"],)
        q={name:models[name].predict_proba(bindings.chosen_inputs(blocks,name))[:,1] for name in need}
        x=matrix(policy["method"],{"compact":p[mask,0]},q,"compact_low")
        extra=head.predict_proba(x)[:,1]
        p[:,0]=adjusted(p[:,0],mask,extra,policy["strength"],policy["boost"])
    elapsed=time.perf_counter()-begin
    c["forward_seconds"]+=elapsed
    c.update(wall_seconds=time.perf_counter()-started,feature_seconds=elapsed,specialist_rows=int(mask.sum()))
    return p,c


def public():
    policy=read(OUT/"policy.json")
    path=OUT/"heads.joblib"
    features=OUT/"features.joblib"
    if checksum(path)!=policy["sha256"] or checksum(features)!=read(OUT/"features.json")["sha256"]:
        raise ValueError("Binding weights changed since calibration")
    selected=policy["selected"]
    pipe=previous.startup()
    models=joblib.load(features)
    head=joblib.load(path)[selected["method"]]
    texts,y,m=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    groups=group_codes(zip(m["Projectname"],m["Filepath"],strict=True))
    archived=predict_bits(BASE/"efficient_aug/selective/predictions.csv",ids,{"thresholds":(recipe()["threshold"],.5)})
    controls={"baseline":None,"specialist":selected,"length":policy["length_control"]}
    trials={k:[] for k in controls}
    results={}
    for k in trials:
        adjusted_pipeline(pipe,texts[:2048],controls[k],models,head)
    for repeat in range(3):
        for key in trials if repeat%2==0 else reversed(trials):
            use=controls[key]
            p,cost=adjusted_pipeline(pipe,texts,use,models,head)
            threshold=use["cutoff"] if use is not None else recipe()["threshold"]
            bits=p>(threshold,.5)
            if key=="baseline" and not np.array_equal(bits,archived):
                raise ValueError("The round 12 reference no longer replays")
            if key in results and not np.array_equal(results[key]>(threshold,.5),bits):
                raise ValueError("A decision changed on public repetition")
            trials[key].append(cost)
            results[key]=p
            print(key,cost,flush=True)
    output={}
    for key,costs in trials.items():
        times={k:median(r[k] for r in costs) for k in costs[0]}
        p=results[key]
        threshold=controls[key]["cutoff"] if controls[key] is not None else recipe()["threshold"]
        value=metric(y,p>(threshold,.5))
        output[key]=dict(metrics=value,cost=times,trials=costs,
            score=competition_formula(value["mean"]["f1"],times["forward_seconds"],times["analytic_gflops"]),
            comparison=summarize_row(y[:,0],archived[:,0],p[:,0]>threshold,
                        (np.fromiter(map(len,texts),dtype=int)<=read(PLAN)["short_chars_max"])&(results["baseline"][:,0]<.05)),
            interval=bootstrap_delta(y,archived,p>(threshold,.5),groups))
        save(OUT/key,y,ids,np.zeros(len(y),dtype=bool),p,times,(threshold,.5))
    write_json(OUT/"analysis.json",output)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("feature_fit","tune","public"))
    args=ap.parse_args()
    {"feature_fit":train_features,"tune":tune,"public":public}[args.stage]()
