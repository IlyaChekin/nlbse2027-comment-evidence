"""Test syntax-only rankers as conditional evidence on rule-negative functions."""
import argparse
import csv
from itertools import product
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.special import logit
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from features import HERE,ROOT,OUT,DENSE,f,checksum,json_at,write_json,load
from project_eval import OP,PARTS,REFERENCE,calibrate
from train import cohorts

R22=ROOT/"round22"
VIEWS=("counts","scoped_counts","flat","linked","combined")
FLAT_COUNTS=tuple(DENSE.index(name) for name in (
    "chars","nodes","depth","parse_error","block_count","branch_count","call_count",
    "call_arg_total","call_arg_max","node_kinds","function_count"))
MODES=tuple(product(("observed","prior"),(.5,1.)))


def fit_local(y,dense,flat,linked,ids):
    matrices={
        "counts":dense[:,FLAT_COUNTS],"scoped_counts":dense,
        "flat":flat,"linked":linked,"combined":hstack((flat,linked),format="csr"),
    }
    models={}
    results={}
    for name,x in matrices.items():
        started=time.perf_counter()
        if name.endswith("counts"):
            model=HistGradientBoostingClassifier(max_iter=120,learning_rate=.07,max_leaf_nodes=23,min_samples_leaf=50,
                                                 l2_regularization=2.,early_stopping=False,random_state=42)
        else:
            model=LogisticRegression(C=2.,solver="liblinear",max_iter=100,random_state=42)
        model.fit(x[ids],y[ids])
        models[name]=model
        results[name]=(x,model)
        print("trained",name,round(time.perf_counter()-started,2),flush=True)
    return results,models


def channel(values, local):
    return np.column_stack((f.features(values,"negative"),logit(local.clip(1e-5,1-1e-5))))


def baseline(frame,heads,gate,policy=OP):
    values=frame[REFERENCE]
    est=f.estimate(heads,values)
    prior,no,yes,asked=f.choose(values,est,policy,gate)
    base=prior.astype(np.float32)
    sure=asked&f.valid(frame["raw"])
    positive=sure&f.weak(frame["raw"])
    negative=sure&~positive
    base[negative]=no[negative]
    base[positive]=yes[positive]
    return dict(p=base,w=est["w"],no=no,sure=sure,negative=negative,asked=asked)


def blend(reference,signal,where,weight):
    p=reference["p"].copy()
    effect=signal-reference["no"]
    if where=="prior":
        unseen=~reference["sure"]
        p[unseen]+=weight*(1-reference["w"][unseen])*effect[unseen]
    k=reference["negative"]
    p[k]+=weight*effect[k]
    return p


def writer(target,frame,reference,pg,cutoff,score,selected,codec):
    with target.open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("row","project","gold","weak","reference","score","base_pred","fixed","at_same_fp","local"))
        old=pg["p"]>codec
        w.writerows(zip(frame["ids"],frame["project"],frame["y"],f.weak(frame["raw"]).astype(int),pg["p"],
                        score,old.astype(int),(score>cutoff).astype(int),selected.astype(int),reference,strict=True))


def run(rotation,cache):
    path=HERE/rotation
    path.mkdir(exist_ok=True)
    fitrecipe=json_at(f.R19/"partition.json")["rotations"][rotation]
    split=f.R19/fitrecipe["partition_file"]
    if checksum(split)!=fitrecipe["partition_sha256"]:
        raise ValueError("Project rotation changed")
    with np.load(split) as a:
        fitids=a["fit"]
    y,dense,flat,linked=cache
    weak,gold,cohort_hash=cohorts()
    if not np.array_equal(gold,y):
        raise ValueError("The structural labels differ from the official cohort")
    tr=fitids[weak[fitids]==0]
    classifiers,models=fit_local(y,dense,flat,linked,tr)
    joblib.dump(models,path/"models.joblib",compress=3)
    frames={}
    for phase in ("dev","test"):
        source=R22/rotation/f"{phase}.npz"
        with np.load(source) as z:
            frames[phase]=dict(z)
    internal,test=frames["dev"],frames["test"]
    old_cfg=json_at(f.WEAK/rotation/"policy.json")
    dfit=np.isin(internal["ids"],old_cfg["fit_rows"])
    cal=np.isin(internal["ids"],old_cfg["calibration_rows"])
    dmask=dfit&f.valid(internal["raw"])&~f.weak(internal["raw"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,ref_cut,base_bits,_=calibrate(frames,REFERENCE,dfit,cal,budget)
    a=baseline(internal,heads,gate)
    b=baseline(test,heads,gate)
    base_test=b["p"]>ref_cut
    raw_negative=f.valid(test["raw"])&~f.weak(test["raw"])
    test_report,dev_report,cal_models={},{},{}
    frozen_candidates={}
    for key,(x,model) in classifiers.items():
        d=model.predict_proba(x[internal["ids"]])[:,1]
        t=model.predict_proba(x[test["ids"]])[:,1]
        condition=LogisticRegression(C=.3,max_iter=350,random_state=42)
        df=channel(internal[REFERENCE],d)
        tf=channel(test[REFERENCE],t)
        condition.fit(df[dmask],internal["y"][dmask])
        cal_models[key]=condition
        candidate_train=condition.predict_proba(df)[:,1]
        candidate_test=condition.predict_proba(tf)[:,1]
        for where,weight in MODES:
            name=f"{key}_{where}_{weight:g}"
            score=blend(a,candidate_train,where,weight)
            dev_bits,cut=f.strict_budget(score[cal],internal["y"][cal],budget)
            dev_report[name]=dict(cutoff=cut,cal=f.comparison(internal["y"][cal],base_bits,dev_bits))
            frozen_candidates[name]=(key,where,weight,cut,t,candidate_test)
    champ=max(dev_report,key=lambda k:dev_report[k]["cal"]["delta_tp"])
    if dev_report[champ]["cal"]["delta_tp"]<=0:
        champ=REFERENCE
    write_json(path/"policy.json",dict(selected=champ,candidates=dev_report,gate=gate,reference_cutoff=ref_cut,
                   cohort_sha256=cohort_hash,features_sha256=checksum(OUT/"manifest.json"),partition_sha256=checksum(split),
                   fit_rows=len(tr),fit_positive=int(y[tr].sum()),dev_fit=int(dmask.sum()),dev_cal=int(cal.sum()),
                   project_fit_sha256=checksum(path/"models.joblib")))
    for name,(key,where,weight,cut,t,candidate) in frozen_candidates.items():
        score=blend(b,candidate,where,weight)
        delta,per=f.per_project(test["y"],test["project"],b["p"],score,base_test)
        matched=np.zeros(len(base_test),dtype=bool)
        for project in per:
            in_project=test["project"]==project
            nfp=int((base_test[in_project]&(test["y"][in_project]==0)).sum())
            matched[in_project],_=f.exact_budget(score[in_project],test["y"][in_project],nfp)
        fixed=f.comparison(test["y"],base_test,score>cut)
        test_report[name]=dict(project_tp=delta,frozen=fixed,per_project=per,
                 matched_cohort=f.comparison(test["y"][raw_negative],base_test[raw_negative],matched[raw_negative]),
                 standalone_ap=float(average_precision_score(test["y"][raw_negative],t[raw_negative])),
                 fused_ap=float(average_precision_score(test["y"][raw_negative],candidate[raw_negative])))
        writer(path/f"{name}_test.csv",test,t,b,cut,score,matched,ref_cut)
        print(rotation,name,delta,fixed["delta_tp"],fixed["delta_fp"],dev_report[name]["cal"]["delta_tp"],flush=True)
    joblib.dump(cal_models,path/"calibrators.joblib",compress=3)
    ap=float(average_precision_score(test["y"][raw_negative],b["no"][raw_negative]))
    write_json(path/"results.json",dict(selected=champ,baseline_weak_negative_ap=ap,
                      cohorts_test_negative=int(raw_negative.sum()),results=test_report,
                      policy_sha256=checksum(path/"policy.json"),calibrators_sha256=checksum(path/"calibrators.joblib")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=PARTS)
    a=p.parse_args()
    run(a.rotation,load())
