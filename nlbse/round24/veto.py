"""Combine the sparse and compact classifiers at a fixed false-positive operating point."""
import argparse
import csv
from itertools import product
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from crossfit_sparse import HERE,ROOT,f,checksum,json_at,write_json
from project_eval import OP,PARTS,REFERENCE,calibrate
from fuse_sparse import combination_inputs,combined_inputs,proposals,restrict,revise,calibrate_sparse

R22=ROOT/"round22"
BASE_FIELDS=("p","w","no","yes","known","flag")
CONFIGS=tuple(product(("blend","veto","promote","two_gate"),("exact","near","wide"),(.5,1.)))


def scalar(x,y,c=.3):
    return LogisticRegression(C=c,solver="liblinear",max_iter=300,random_state=42).fit(x,y)


def exact_state(frame,heads,gate,policy=OP):
    est=f.estimate(heads,frame[REFERENCE])
    p,no,yes,asks=f.choose(frame[REFERENCE],est,policy,gate)
    sure=asks&f.valid(frame["raw"])
    fire=f.weak(frame["raw"])
    p[sure]=np.where(fire[sure],yes[sure],no[sure])
    return dict(p=p.astype(np.float32),w=est["w"],no=no,yes=yes,known=sure,flag=fire,asked=asks)


def sparse_calibration(folder,mode,recipe):
    path=folder/f"{mode}_oof.npz"
    if checksum(path)!=recipe["predictions_sha256"]:
        raise ValueError("Outside-file predictions changed")
    with np.load(path) as x:
        weak=x["weak"]
        return [scalar(x["logits"][weak==k,k,None],x["y"][weak==k],1.) for k in (0,1)]


def fit_heads(frame,state,sparse,fit,threshold):
    valid=f.valid(frame["raw"])
    w=f.weak(frame["raw"])
    combos=combination_inputs(frame[REFERENCE],sparse)
    learned=[]
    for k,x in enumerate(combos):
        use=fit&valid&(w==bool(k))
        learned.append(scalar(x[use],frame["y"][use]))
    two=combined_inputs(state,sparse)
    split=[]
    positive=state["p"]>threshold
    for k in (False,True):
        use=fit&(positive==k)
        split.append(scalar(two[use],frame["y"][use]))
    return learned,split


def load_sparse(folder,name,phase,frame,recipe):
    source=folder/f"{name}_{phase}.npz"
    if checksum(source)!=recipe[phase]:
        raise ValueError("Sparse scores changed")
    with np.load(source) as x:
        if not np.array_equal(x["ids"],frame["ids"]):
            raise ValueError("Sparse scores do not cover the conditional rows")
        return x["logits"]


def records(rotation):
    source=HERE/rotation
    dest=source/"comparison"
    dest.mkdir(exist_ok=True)
    a,b=(dict(np.load(R22/rotation/f"{k}.npz")) for k in ("dev","test"))
    frames={"dev":a,"test":b}
    train=json_at(f.WEAK/rotation/"policy.json")
    fit=np.isin(a["ids"],train["fit_rows"])
    cal=np.isin(a["ids"],train["calibration_rows"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,cut,baseline_dev,_=calibrate(frames,REFERENCE,fit,cal,budget)
    u,v=exact_state(a,heads,gate),exact_state(b,heads,gate)
    assert np.array_equal(u["p"][cal]>cut,baseline_dev)
    y=b["y"]
    ref=v["p"]>cut
    report={}
    models={}
    for mode,recipe in json_at(source/"fit.json")["models"].items():
        oof=sparse_calibration(source,mode,recipe)
        r,s=[calibrate_sparse(oof,load_sparse(source,mode,phase,frame,recipe)) for phase,frame in frames.items()]
        model=fit_heads(a,u,r,fit,cut)
        models[mode]=dict(sparse=oof,combiner=model)
        da=proposals(a,u,r,model,cut)
        dt=proposals(b,v,s,model,cut)
        for name,scope,mix in CONFIGS:
            key=f"{mode}_{name}_{scope}_{mix:g}"
            p,avail=revise(u,da,cut,name,scope,mix)
            at,th=f.strict_budget(p[cal],a["y"][cal],budget)
            q,active=revise(v,dt,cut,name,scope,mix)
            d,per=f.per_project(y,b["project"],v["p"],q,ref)
            group=np.zeros(len(ref),dtype=bool)
            for project in per:
                which=b["project"]==project
                fp=int((ref[which]&(y[which]==0)).sum())
                group[which],_=f.exact_budget(q[which],y[which],fp)
            field=group^ref
            changes={"fn_recovered":int((field&group&(y==1)).sum()),
                     "fp_removed":int((field&ref&(y==0)).sum()),
                     "tp_lost":int((field&ref&(y==1)).sum()),
                     "fp_added":int((field&group&(y==0)).sum())}
            info=dict(dev=f.comparison(a["y"][cal],baseline_dev,at),project_tp=d,project=per,
                        frozen=f.comparison(y,ref,q>th),if_same_fp=changes,threshold=th,
                        cal_rows=int((cal&avail).sum()),test_queries=int(active.sum()))
            report[key]=info
            # Keep every variant in one compact aligned prediction matrix.
            np.savez_compressed(dest/f"{key}.npz",ids=b["ids"],gold=y,score=q,baseline=ref,frozen=q>th,
                                at_same_fp=group,queries=active)
    joblib.dump(models,dest/"combiners.joblib",compress=3)
    top=max(report,key=lambda x:(report[x]["dev"]["delta_tp"],-report[x]["cal_rows"]))
    select=top if report[top]["dev"]["delta_tp"]>0 else REFERENCE
    write_json(dest/"results.json",dict(selected=select,reference_cutoff=cut,dev_fp=budget,options=report,
                                    combinations_sha256=checksum(dest/"combiners.joblib")))
    print(rotation,"selected",select,flush=True)
    print([(k,v["dev"]["delta_tp"],v["project_tp"],v["test_queries"]) for k,v in report.items()],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=PARTS)
    records(p.parse_args().rotation)
