"""Calibrate on internal files and measure the actual cost of a selective teacher cascade."""

import argparse
import csv
import json
from statistics import median
import time

import joblib
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor

from crossfit import HERE, ROOT, checksum, write_json
from disagreement import cut_fraction, describe, mix, rank
from evaluate import counts, exact_budget
from precision import summary
from pipeline import PUBLIC, load, metric
from students import dataset, group_codes, load_selected, predict_bits, scores
import efficient_aug
from round2_report import bootstrap_delta
from score_runtime import competition_formula


OUT=HERE/"routed_public"
FAMILY="unixcoder9"
MODES=("uncertainty","loss_gain")


def release():
    return json.loads((efficient_aug.OUT/"policy.json").read_text())["selected"]["selective"]


def variables(texts,p):
    shape=np.asarray([describe(text) for text in texts],dtype=np.float32)
    return np.column_stack((logit(p.clip(1e-6,1-1e-6)),np.log1p(shape))),np.maximum(1,np.minimum(shape[:,0],2200))


def teacher_checkpoint():
    folder=ROOT/"round10/runs"/FAMILY
    recipe=json.loads((folder/"selected.json").read_text())
    return folder,recipe


def tune():
    OUT.mkdir(exist_ok=True)
    _,_,texts,y,ids,_=dataset(full=True)
    folder,cfg=teacher_checkpoint()
    src=folder/f"{cfg['checkpoint'].removesuffix('.pt')}_holdout.npz"
    with np.load(src) as f:
        if ids!=f["ids"].tolist() or not np.array_equal(y,f["y"]):
            raise ValueError("Teacher holdout differs from the reference")
        teacher,folds=f["p"][:,0],f["fold"].copy()
    ref=release()
    raw,cost=efficient_aug.run(efficient_aug.startup(),texts,ref)
    base=raw[:,0]
    mat=raw[:,1]>.5
    gold=y[:,0]
    x,length=variables(texts,base)
    train,cal,review=folds<=2,folds==3,folds==4
    target=lambda p: np.log(np.where(gold,p,1.-p).clip(1e-5,1.))
    delta=np.clip(target(mix(base,teacher,.55))-target(base),-2.5,2.5)
    route=HistGradientBoostingRegressor(max_leaf_nodes=9,learning_rate=.07,max_iter=130,
               min_samples_leaf=120,l2_regularization=2.,random_state=42).fit(x[train],delta[train])
    threshold=ref["threshold"]
    base_bits=base>threshold
    budget=counts(gold[cal],base_bits[cal])["fp"]
    before,_=exact_budget(base[cal],gold[cal],budget)
    scope=(gold==0)&(base>.9)&(teacher>.9)
    scaler=30655/len(y)
    main=lambda f: cost[f]*scaler
    extra=lambda f,share: cfg["settings"]["timing"][f]*scaler*share
    all_rec=[]
    for mode in MODES:
        route_score=rank(mode,route,x,base,threshold,length)
        for fraction in (.01,.02,.05,.1):
            cut=cut_fraction(route_score[cal],length[cal],fraction)
            use=route_score>cut
            share=float(length[use&cal].sum()/length[cal].sum())
            for weight in (.3,.55,.8,1.):
                q=np.where(use,mix(base,teacher,weight),base)
                proposed,new_cut=exact_budget(q[cal],gold[cal],budget)
                f1=metric(y[cal],np.column_stack((proposed,mat[cal])))["mean"]["f1"]
                projected=competition_formula(f1,main("forward_seconds")+extra("forward_seconds",share),
                                                 main("analytic_gflops")+extra("analytic_gflops",share))
                all_rec.append(dict(mode=mode,route_cutoff=cut,cutoff=new_cut,weight=weight,fraction=fraction,actual_proxy_share=share,
                                    projected_calibration_score=projected,development=summary(gold[cal],before,proposed,scope[cal]),
                                    review=summary(gold[review],base_bits[review],q[review]>new_cut,scope[review])))
    chosen={mode:max((r for r in all_rec if r["mode"]==mode),key=lambda r:r["projected_calibration_score"]) for mode in MODES}
    artifact=OUT/"route.joblib"
    joblib.dump(route,artifact,compress=3)
    write_json(OUT/"policy.json",dict(selected=chosen,candidates=all_rec,reference=ref,budget=budget,
              row_roles={str(f):int((folds==f).sum()) for f in np.unique(folds)},teacher_sha256=cfg["settings"]["sha256"],
              holdout_sha256=checksum(src),regressor_sha256=checksum(artifact)))
    print(chosen,flush=True)


def run(pipe,text,spec,router,teacher):
    start=time.perf_counter()
    p,c=efficient_aug.run(pipe,text,release())
    original=p.copy()
    routed=np.zeros(len(p),dtype=bool)
    if spec:
        began=time.perf_counter()
        x,l=variables(text,p[:,0])
        a=rank(spec["mode"],router,x,p[:,0],release()["threshold"],l)
        routed=a>spec["route_cutoff"]
        selector=time.perf_counter()-began
        if routed.any():
            q,d=scores(*teacher,[value for value,keep in zip(text,routed,strict=True) if keep],FAMILY)
            c["analytic_gflops"]+=d["analytic_gflops"]
            c["forward_seconds"]+=d["forward_seconds"]
            p[routed,0]=mix(p[routed,0],q[:,0],spec["weight"])
        c["forward_seconds"]+=selector
    c.update(wall_seconds=time.perf_counter()-start,teacher_rows=int(routed.sum()))
    return original,p,routed,c


def public():
    cfg=json.loads((OUT/"policy.json").read_text())
    reg=OUT/"route.joblib"
    if checksum(reg)!=cfg["regressor_sha256"] or teacher_checkpoint()[1]["settings"]["sha256"]!=cfg["teacher_sha256"]:
        raise ValueError("Research model changed")
    router=joblib.load(reg)
    ref=cfg["reference"]["threshold"]
    text,y,m=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    release_predictions=predict_bits(efficient_aug.OUT/"selective/predictions.csv",ids,{"thresholds":(ref,.5)})
    clusters=group_codes(zip(m["Projectname"],m["Filepath"],strict=True))
    pipe=efficient_aug.startup()
    teacher=load_selected(FAMILY)[:2]
    choices={"release":None,**cfg["selected"]}
    trials={name:[] for name in choices}
    predictions={}
    all_score={}
    for name,spec in choices.items():
        run(pipe,text[:2048],spec,router,teacher)
    budget=counts(y[:,0],release_predictions[:,0])["fp"]
    for phase in range(3):
        for name in (choices if phase%2==0 else reversed(choices)):
            spec=choices[name]
            original,p,routed,c=run(pipe,text,spec,router,teacher)
            if not np.array_equal(original>(ref,.5),release_predictions):
                raise ValueError("Public reference predictions changed")
            cut=spec["cutoff"] if spec else ref
            current=p>(cut,.5)
            if name in predictions and not np.array_equal(current,predictions[name]):
                raise ValueError("Nondeterministic public cascade")
            all_score[name]=(p,routed)
            predictions[name]=current
            trials[name].append(c)
            print(name,c,flush=True)
    result={}
    base, _=exact_budget(all_score["release"][0][:,0],y[:,0],budget)
    target=np.zeros(len(y),dtype=bool)
    for name, runs in trials.items():
        p,routed=all_score[name]
        matched,_=exact_budget(p[:,0],y[:,0],budget)
        med={key:median(r[key] for r in runs) for key in runs[0]}
        out=metric(y,predictions[name])
        result[name]=dict(metrics=out,cost=med,trials=runs,
                          score=competition_formula(out["mean"]["f1"],med["forward_seconds"],med["analytic_gflops"]),
                          matched_fp=summary(y[:,0],base,matched,target),
                          frozen=summary(y[:,0],release_predictions[:,0],predictions[name][:,0],target),
                          interval=bootstrap_delta(y,release_predictions,predictions[name],clusters))
        with (OUT/f"{name}.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(("row","label_vulnerability","label_mat","probability","vulnerability","mat","routed","vulnerability_equal_fp"))
            w.writerows(zip(ids,y[:,0],y[:,1],p[:,0],predictions[name][:,0].astype(int),predictions[name][:,1].astype(int),
                            routed.astype(int),matched.astype(int),strict=True))
    write_json(OUT/"results.json",result)
    print({key:(r["score"],r["frozen"]["delta_tp"],r["frozen"]["delta_fp"]) for key,r in result.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("tune","public"))
    tune() if p.parse_args().phase=="tune" else public()
