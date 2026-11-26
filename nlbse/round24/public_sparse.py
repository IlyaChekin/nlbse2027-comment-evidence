"""Measure the internally frozen sparse correction against the current validated tool."""
import csv
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np

from internal_sparse import HERE,ROOT,OUT,f,REFERENCE,recipe,json_at,write_json,checksum
from veto import proposals,restrict,revise,calibrate_sparse
from sparse_model import aligned
from fuse_sparse import correction
from union_scanner import scan,PROGRAM
from layout_official import recover,load,PUBLIC,ARCHIVE

REPORT=HERE/"public"


def measured_text(model,texts,reference,gate,heads,variant=None,package=None):
    p,aux,cost=f.infer_student(model,texts,f.pinned()[0],columns=22,return_all=True)
    t=time.perf_counter()
    values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
    est=f.estimate(heads,values)
    score,no,yes,ask=f.choose(values,est,gate,reference["gate"])
    flag=np.zeros(len(texts),dtype=bool)
    known=np.zeros_like(flag)
    if ask.any():
        outcome=scan([s for s,on in zip(texts,ask,strict=True) if on])
        known[ask]=outcome>=0
        flag[ask]=outcome==1
        score[known]=np.where(flag[known],yes[known],no[known])
    score=score.astype(np.float32)
    tag=recover(texts,p[:,1]>.5)
    start=time.perf_counter()
    threshold=reference["threshold"]
    queried=nnz=0
    changed=np.zeros(len(texts),dtype=bool)
    bits=score>threshold
    if variant is not None:
        state=dict(p=score,w=est["w"],no=no,yes=yes,known=known,flag=flag)
        bits,score,changed,nnz=correction(texts,values,state,threshold,variant,package)
        queried=int(changed.sum())
    end=time.perf_counter()
    cost["wall_seconds"]+=end-t
    cost["forward_seconds"]+=end-t
    cost["analytic_gflops"]+=4*nnz/1e9
    cost.update(extra_calls=queried,exact_calls=int(ask.sum()),sparse_seconds=end-start)
    return np.column_stack((bits,tag)),score,cost,changed


def public():
    cfg=json_at(OUT/"policy.json")
    selected=cfg["selected"]
    if selected==REFERENCE:
        raise ValueError("No candidate passed internal calibration")
    if checksum(OUT/"models.joblib")!=cfg["models_sha256"]:
        raise ValueError("The learned sparse correction changed")
    variant=cfg["candidates"][selected]
    package=joblib.load(OUT/"models.joblib")[variant["mode"]]
    reference,gate,heads=recipe()
    texts,gold,_=load(PUBLIC)
    with (ARCHIVE/"profile_marker.csv").open(newline="") as file:
        archive=np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")] for r in csv.DictReader(file)],dtype=bool)
    model=f.pipe("official",f.small.tokenizer())
    arms=(REFERENCE,selected)
    times={name:[] for name in arms}
    def call(name,rows):
        return measured_text(model,rows,reference,gate,heads,None if name==REFERENCE else variant,package)
    for name in arms:
        call(name,texts[:2048])
    result={}
    for run in range(5):
        for name in arms if run%2==0 else reversed(arms):
            bits,p,cost,active=call(name,texts)
            if (name==REFERENCE and not np.array_equal(bits,archive)) or not np.array_equal(bits[:,1],archive[:,1]):
                raise ValueError("Archived reference predictions changed")
            if name in result and not np.array_equal(bits,result[name][0]):
                raise ValueError("A repeated run changed decisions")
            result[name]=bits,p,active
            times[name].append(cost)
            print(name,cost,flush=True)
    REPORT.mkdir(exist_ok=True)
    budget=int((archive[:,0]&(gold[:,0]==0)).sum())
    stats={}
    for name,(bits,p,active) in result.items():
        cost={k:median(x[k] for x in times[name]) for k in times[name][0]}
        quality=f.metric(gold,bits)
        score=f.competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])
        if name==REFERENCE:
            matched=archive[:,0]
        elif variant["cohort_only"]:
            matched=archive[:,0].copy()
            inside_budget=int((matched[active]&(gold[active,0]==0)).sum())
            matched[active],_=f.exact_budget(p[active],gold[active,0],inside_budget)
        else:
            matched,_=f.exact_budget(p,gold[:,0],budget)
        file=REPORT/f"{name.replace(chr(58),chr(95))}.csv"
        with file.open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","label_vulnerability","label_mat","pred_vulnerability","pred_MAT","probability","at_same_fp"))
            w.writerows((i,*y,*b,q,s) for i,(y,b,q,s) in enumerate(zip(gold,bits.astype(int),p,matched.astype(int),strict=True)))
        stats[name]=dict(score=score,cost=cost,quality=quality,trials=times[name],
                frozen=f.comparison(gold[:,0],archive[:,0],bits[:,0]),
                matched=f.comparison(gold[:,0],archive[:,0],matched),predictions_sha256=checksum(file))
    write_json(REPORT/"results.json",dict(selected=selected,results=stats,public_sha256=checksum(PUBLIC),
               internal_policy_sha256=checksum(OUT/"policy.json"),scanner_sha256=checksum(PROGRAM)))
    print({k:v["score"] for k,v in stats.items()},flush=True)


if __name__=="__main__":
    public()
