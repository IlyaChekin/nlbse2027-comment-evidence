"""Use two identifier views selectively, with one compact checkpoint and a shared MAT scan."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace
import time

import numpy as np

from fast_mat import HERE, ROOT, infer
from backbones import frozen
from normalize_code import batch as canonicalize
from mat_replay import predict as tag_rule
from students import dataset, group_codes, metric, predict_bits, save
from pipeline import PUBLIC, load, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT=HERE/"efficient_aug"


def startup():
    net,tokenizer=frozen("canonical_official")
    return SimpleNamespace(encoder=net,tok=tokenizer,student={"thresholds":(.5,.5)})


def fit():
    _,_,texts,y,ids,_=dataset(full=True)
    pipe=startup()
    raw,c0=infer(pipe,texts,probabilities=True)
    start=time.perf_counter()
    norm=canonicalize(texts)
    conversion=time.perf_counter()-start
    post,c1=infer(pipe,norm,probabilities=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if f["rows"].tolist()!=ids:
            raise ValueError("Different official calibration row IDs")
        folds=f["folds"]
    cal,review=folds==3,folds==4
    a,b=raw[:,0],post[:,0]
    labels=raw[:,1].astype(bool)
    shape=np.asarray([max(1,min(len(text),2100)) for text in norm])
    scale=30_655/len(y)
    records=[]

    def compare(view,p,gflops,forward,**kw):
        for threshold in np.linspace(.23,.76,54):
            inference=np.column_stack((p>threshold,labels))
            value=metric(y[cal],inference[cal])["mean"]["f1"]
            records.append(dict(view=view,threshold=float(threshold),f1=value,
                proxy=competition_formula(value,forward*scale,gflops*scale),review=metric(y[review],inference[review]),**kw))

    compare("raw",a,c0["analytic_gflops"],c0["forward_seconds"])
    compare("norm",b,c1["analytic_gflops"],c1["forward_seconds"]+conversion)
    compare("mean",.5*(a+b),c0["analytic_gflops"]+c1["analytic_gflops"],
            c0["forward_seconds"]+c1["forward_seconds"]+conversion)
    for lower in (.02,.055,.1,.15,.2,.25,.35):
        for upper in (.65,.76,.85,.91,.96,.99):
            active=(a>lower)&(a<upper)
            fraction=shape[cal&active].sum()/shape[cal].sum()
            p=np.where(active,.5*(a+b),a)
            compare("selective",p,c0["analytic_gflops"]+fraction*c1["analytic_gflops"],
                    c0["forward_seconds"]+fraction*(c1["forward_seconds"]+conversion),
                    lower=lower,upper=upper,calibration_fraction=float(active[cal].mean()))
    best={view:max((r for r in records if r["view"]==view),key=lambda r:r["proxy"]) for view in ("raw","norm","mean","selective")}
    choose=max((k for k in best if k not in {"raw","selective"}),key=lambda k:best[k]["proxy"])
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"policy.json",dict(selected=best,extra_control=choose,grid=records))
    print(best,flush=True)


def run(pipe,texts,cfg):
    start=time.perf_counter()
    view=cfg["view"]
    p=None
    if view!="norm":
        p,c= infer(pipe,texts,probabilities=True)
    used=0
    if view!="raw":
        active=(p[:,0]>cfg["lower"])&(p[:,0]<cfg["upper"]) if view=="selective" else np.ones(len(texts),dtype=bool)
        raw=[text for text,s in zip(texts,active,strict=True) if s]
        used=len(raw)
        if raw:
            began=time.perf_counter()
            norm=canonicalize(raw)
            duration=time.perf_counter()-began
            other,d=infer(pipe,norm,probabilities=True)
            if p is None:
                p,c=other,d
                t=time.perf_counter()
                p[:,1]=tag_rule(raw)
                c["forward_seconds"]+=time.perf_counter()-t
            else:
                p[active,0]=.5*(p[active,0]+other[:,0])
                for k in ("forward_seconds","analytic_gflops"):
                    c[k]+=d[k]
            c["forward_seconds"]+=duration
    c.update(wall_seconds=time.perf_counter()-start,routed=used)
    return p,c


def public():
    catalog=json.loads((OUT/"policy.json").read_text())
    variants=("raw","selective",catalog["extra_control"])
    pipe=startup()
    texts,y,meta=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    groups=group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    base=predict_bits(ROOT/"round11/cold_shift/predictions.csv",ids,{"thresholds":(.5,.5)})
    trials={k:[] for k in variants}; predictions={}
    for key in variants:
        run(pipe,texts[:2048],catalog["selected"][key])
    for i in range(3):
        for key in variants if i%2==0 else reversed(variants):
            cfg=catalog["selected"][key]
            p,cost=run(pipe,texts,cfg)
            t=(cfg["threshold"],.5)
            bits=p>t
            if key in predictions and not np.array_equal(predictions[key]>t,bits):
                raise ValueError("One official decision changed between repetitions")
            if not np.array_equal(base[:,1],bits[:,1]):
                raise ValueError("Replay of MAT tags differs from the official reference")
            predictions[key]=p
            trials[key].append(cost)
            print(key,cost,flush=True)
    result={}
    for key,obs in trials.items():
        t=(catalog["selected"][key]["threshold"],.5)
        p=predictions[key]
        cost={field:median(r[field] for r in obs) for field in obs[0]}
        m=metric(y,p>t)
        result[key]=dict(metrics=m,cost=cost,local_formula=competition_formula(m["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                         wall_charged=competition_formula(m["mean"]["f1"],cost["wall_seconds"],cost["analytic_gflops"]),
                         interval_vs_round11=bootstrap_delta(y,base,p>t,groups),trials=obs)
        save(OUT/key,y,ids,np.zeros(len(y),dtype=bool),p,cost,t)
    write_json(OUT/"analysis.json",result)


def predict(source,destination):
    entry=json.loads((HERE/"entry.json").read_text())
    policy=json.loads((OUT/"policy.json").read_text())["selected"][entry["backend"]]
    with source.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function column")
        texts=[row["Function"] for row in reader]
    if not texts or any(t is None for t in texts):
        raise ValueError("Every input row must supply a function")
    p,_=run(startup(),texts,policy)
    with destination.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>(policy["threshold"],.5)).astype(np.int8).tolist())


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","public","predict"))
    ap.add_argument("--input",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.stage=="tune":
        fit()
    elif args.stage=="public":
        public()
    else:
        if args.input is None or args.output is None:
            ap.error("predict requires --input and --output")
        predict(args.input,args.output)
