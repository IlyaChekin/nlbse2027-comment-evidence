"""Learn when a second identifier view improves the same encoder's decision."""

import argparse
import csv
from hashlib import sha256
import time

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor

from view_train import HERE,ROOT,inputs,json_at,open_view,small,tokenize
from experiment import checksum,comparison,infer,strict_budget
from evaluate import counts,exact_budget,raw
from local_public import heldout,run
from normalize_code import batch as normalize
from pipeline import PUBLIC,TRAIN,load,metric,write_json
from score_runtime import competition_formula
from view_eval import benchmark,pinned,selected_views


PROJECT=HERE/"router"/"projects"
OFFICIAL=HERE/"router"/"official"
KINDS=("edge","loss","utility")


def features(texts,p):
    shape=np.asarray([(len(s),s.count("\n"),s.count("_"),s.count("("),s.count("{"),s.count(";"),s.count("->"),s.count("#"))
                      for s in texts],dtype=np.float64)
    return np.column_stack((logit(p.clip(1e-6,1-1e-6)),np.log1p(shape)))


def views(pipe,texts):
    p,cost=infer(pipe,texts,encoding=tokenize)
    q,other=infer(pipe,normalize(texts),encoding=tokenize)
    if not np.array_equal(p[:,1],q[:,1]):
        raise ValueError("Normalization changed the comment rule")
    return p,q[:,0],cost,other


def training_features(texts,y,p,q,anchor,selected):
    x=features(texts,p)
    pair=(p+q)/2
    likelihood=lambda v:np.where(y==1,v,1-v).clip(1e-5,1)
    delta=np.clip(np.log(likelihood(pair))-np.log(likelihood(p)),-2.,2.)
    utility=(np.where(y==1,1.,-.87))*(expit(14*(pair-anchor))-expit(14*(p-anchor)))
    options=dict(max_leaf_nodes=7,max_iter=100,learning_rate=.065,min_samples_leaf=55,l2_regularization=2.,random_state=42)
    return {key:HistGradientBoostingRegressor(**options).fit(x[selected],ages[selected])
            for key,ages in (("loss",delta),("utility",utility))}


def order(kind,x,models,anchor):
    if kind=="edge":
        return -np.abs(x[:,0]-logit(anchor))
    # Estimate benefit per query. sqrt of the visible length is a conservative cost proxy.
    return models[kind].predict(x)/np.maximum(np.exp(.5*x[:,1])/10,1.)


def budget_mask(score,complexity,share):
    positions=np.argsort(-score,kind="stable")
    cum=np.cumsum(complexity[positions])/complexity.sum()
    return float(score[positions[min(np.searchsorted(cum,share,side="right"),len(positions)-1)]])


def apply(p,q,texts,models,rule,route,anchor,gate_scores=None):
    kind=rule["kind"]
    if kind in {"reference","threshold"}:
        return selected_views(p,q,route)
    if kind=="raw":
        return p,np.zeros(len(p),dtype=bool)
    if gate_scores is None:
        gate_scores=order(kind,features(texts,p),models,anchor)
    active=gate_scores>rule["eligibility"]
    return np.where(active,(1-rule["weight"])*p+rule["weight"]*q,p),active


def calibrate(texts,y,p,q,cal,rev,models,route,anchor,budget,cost=None):
    x=features(texts,p[:,0])
    complexity=np.maximum(np.exp(x[:,1])-1,1)
    initial,_=selected_views(p[:,0],q,route)
    reference,_=exact_budget(initial[cal],y[cal,0],budget)
    fixed=initial>anchor
    primary=[]
    for kind in ("reference","threshold","raw",*KINDS):
        scores=order(kind,x,models,anchor) if kind in KINDS else None
        for share in ((.03,.06,.09,.13,.19) if scores is not None else (0.,)):
            gate=budget_mask(scores[cal],complexity[cal],share) if scores is not None else -1e20
            for weight in ((.35,.5,.7,1.) if scores is not None else (.5,)):
                rule=dict(kind=kind,share=share,eligibility=gate,weight=weight)
                s,active=apply(p[:,0],q,texts,models,rule,route,anchor,scores)
                snap,cut=strict_budget(s[cal],y[cal,0],budget)
                cut=anchor if kind=="reference" else cut
                estimate=float(complexity[active&cal].sum()/complexity[cal].sum())
                before=comparison(y[cal,0],reference,snap)
                rule.update(threshold=cut,dev=before,review=comparison(y[rev,0],fixed[rev],s[rev]>cut),
                            selected_rows=int((active&cal).sum()),estimated_share=estimate)
                if cost is not None:
                    g=(cost[0]+estimate*cost[1])*30655/len(y)
                    labels=metric(y[cal],np.column_stack((s[cal]>cut,p[cal,1]>.5)))
                    rule["projected_score"]=competition_formula(labels["mean"]["f1"],0.,g)
                primary.append(rule)
    priority=(lambda r:(r["dev"]["delta_tp"],-r["estimated_share"])) if cost is None else (lambda r:r["projected_score"])
    return {k:max((v for v in primary if v["kind"]==k),key=priority) for k in ("reference","threshold","raw",*KINDS)}


def project():
    PROJECT.mkdir(parents=True,exist_ok=True)
    texts,label,projects,_,partition=inputs()
    tok=small.tokenizer()
    pipe=open_view("projects",42,"reference",tok)
    route,_=pinned()
    anchor=json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    i=partition["probe"]
    t=[texts[j] for j in i]
    p,q,_,_=views(pipe,t)
    models=training_features(t,label[i,0],p[:,0],q,anchor,np.ones(len(i),dtype=bool))
    i=partition["dev"]
    t=[texts[j] for j in i]
    p,q,_,_=views(pipe,t)
    budget=counts(label[i,0],raw("dev")["compact"]>anchor)["fp"]
    rules=calibrate(t,label[i],p,q,np.ones(len(i),dtype=bool),np.zeros(len(i),dtype=bool),models,route,anchor,budget)
    best=max(rules,key=lambda k:(rules[k]["dev"]["delta_tp"],-rules[k]["estimated_share"]))
    write_json(PROJECT/"policy.json",dict(modes=rules,selected=best,fp=budget))
    i=partition["test"]
    t=[texts[j] for j in i]
    p,q,_,_=views(pipe,t)
    initial,_=selected_views(p[:,0],q,route)
    budget=counts(label[i,0],raw("test")["compact"]>anchor)["fp"]
    base,_=exact_budget(initial,label[i,0],budget)
    fixed=initial>rules["threshold"]["threshold"]
    results={}
    for name,rule in rules.items():
        v,active=apply(p[:,0],q,t,models,rule,route,anchor)
        cand,_=exact_budget(v,label[i,0],budget)
        frozen=v>rule["threshold"]
        results[name]=dict(matched=comparison(label[i,0],base,cand),frozen=comparison(label[i,0],fixed,frozen),routed=int(active.sum()))
        with (PROJECT/f"{name}_test.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","label","raw","canonical","score","routed","before_equal","after_equal","before_frozen","after_frozen"))
            w.writerows(zip(i,projects[i],label[i,0],p[:,0],q,v,active.astype(int),base.astype(int),cand.astype(int),
                            fixed.astype(int),frozen.astype(int),strict=True))
        print(name,results[name],flush=True)
    write_json(PROJECT/"results.json",dict(modes=results,rows=len(i),budget=budget,selected=best))


def tune():
    OFFICIAL.mkdir(parents=True,exist_ok=True)
    t,y,ids,_,_=heldout(include_test=True)
    _,_,meta=load(TRAIN,audit=True)
    code=lambda s:int.from_bytes(sha256(s.encode()).digest()[:4],"little")%10
    fold=np.asarray([code(f"NLBSE-R17-ROUTE\0{meta['Projectname'][i]}\0{meta['Filepath'][i]}") for i in ids])
    train,cal,review=fold<5,(fold>=5)&(fold<8),fold>=8
    if not train.any() or not cal.any() or not review.any():
        raise ValueError("Incomplete internal router split")
    route,anchor=pinned()
    pipe=open_view("official",42,"reference",small.tokenizer())
    p,q,own,extra=views(pipe,t)
    original,_=selected_views(p[:,0],q,route)
    budget=counts(y[cal,0],original[cal]>anchor)["fp"]
    models=training_features(t,y[:,0],p[:,0],q,anchor,train)
    plan=calibrate(t,y,p,q,cal,review,models,route,anchor,budget,(own["analytic_gflops"],extra["analytic_gflops"]))
    selected=max(plan,key=lambda k:(plan[k]["projected_score"], k=="reference"))
    artifact=OFFICIAL/"router.joblib"
    joblib.dump(models,artifact,compress=3)
    write_json(OFFICIAL/"policy.json",dict(modes=plan,selected=selected,fp=budget,fit_rows=ids[train].tolist(),
        cal_rows=ids[cal].tolist(),review_rows=ids[review].tolist(),route=route,anchor=anchor,sha256=checksum(artifact)))
    print(selected,plan,flush=True)


def measured(pipe,texts,models,rule,cfg):
    if rule["kind"] in {"reference","threshold"}:
        return run(pipe,texts,cfg["route"],tokenize)
    tick=time.perf_counter()
    p,tot=infer(pipe,texts,encoding=tokenize)
    begin=time.perf_counter()
    active=np.zeros(len(p),dtype=bool)
    if rule["kind"]!="raw":
        x=features(texts,p[:,0])
        active=order(rule["kind"],x,models,cfg["anchor"])>rule["eligibility"]
    spent=time.perf_counter()-begin
    if active.any():
        begin=time.perf_counter()
        canonical=normalize([s for s,b in zip(texts,active,strict=True) if b])
        spent+=time.perf_counter()-begin
        q,other=infer(pipe,canonical,encoding=tokenize)
        p[active,0]=(1-rule["weight"])*p[active,0]+rule["weight"]*q[:,0]
        tot["forward_seconds"]+=other["forward_seconds"]
        tot["analytic_gflops"]+=other["analytic_gflops"]
    tot.update(wall_seconds=time.perf_counter()-tick,forward_seconds=tot["forward_seconds"]+spent,routed=int(active.sum()))
    return p,tot


def public():
    cfg=json_at(OFFICIAL/"policy.json")
    asset=OFFICIAL/"router.joblib"
    if checksum(asset)!=cfg["sha256"]:
        raise ValueError("Router weights changed")
    models=joblib.load(asset)
    pipe=open_view("official",42,"reference",small.tokenizer())
    names=list(dict.fromkeys(("reference",cfg["selected"])))
    benchmark(OFFICIAL,names,cfg,lambda name,t:measured(pipe,t,models,cfg["modes"][name],cfg))


def oracle():
    """Measure which errors two views could repair at the already frozen threshold."""
    texts,y,_=load(PUBLIC)
    _,anchor=pinned()
    pipe=open_view("official",42,"reference",small.tokenizer())
    p,q,_,_=views(pipe,texts)
    with (HERE.parent/"round16/context_public/full_public.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=list(range(len(y))) or not np.array_equal(y[:,0],[int(r["label_vulnerability"]) for r in rows]):
        raise ValueError("The archived routed decisions differ in rows or labels")
    old=np.asarray([r["vulnerability"]=="1" for r in rows])
    pick=np.column_stack((np.column_stack((p[:,0],q,(p[:,0]+q)/2))>anchor,old))
    correct=(pick==y[:,0,None].astype(bool)).any(1)
    gold=y[:,0].astype(bool)
    best=gold^~correct
    cost=json_at(HERE.parent/"round16/context_public/results.json")["full"]["cost"]
    quality=metric(y,np.column_stack((best,p[:,1]>.5)))
    report=dict(threshold=anchor,rows=len(y),baseline=counts(y[:,0],old),oracle=counts(y[:,0],best),
                mean_f1=quality["mean"]["f1"],fixed_cost_score=competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                candidates=["raw","canonical","average","current_selective"],
                assumption="labels select the best binary decision per row at the same frozen cutoff; reference cost held fixed")
    write_json(OFFICIAL/"oracle.json",report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("project","tune","public","oracle"))
    {"project":project,"tune":tune,"public":public,"oracle":oracle}[p.parse_args().stage]()
