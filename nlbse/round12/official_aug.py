"""Test the project-validated augmentation against the official compact baseline."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np

from backbones import frozen
from project_split import HERE, ROOT
from official import base_pipe
from augmentation import DEST as AUGMENTATION
from pipeline import PUBLIC, load, write_json
from students import dataset, group_codes, metric, predict_bits, save
from normalize_code import parallel as canonicalize
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT=HERE/"official_aug"


def eligible():
    result=json.loads((AUGMENTATION/"analysis.json").read_text())
    tag=result["frozen"]
    trial=result["results"][tag]
    original=json.loads((HERE/"heads/analysis.json").read_text())["tests"]["compact/raw"]
    pooled=trial["scores"]["vulnerability"]["f1"]-original["scores"]["vulnerability"]["f1"]
    wins=sum(row["vulnerability"]["f1"]>original["by_project"][proj]["vulnerability"]["f1"]
             for proj,row in trial["by_project"].items())
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"eligibility.json",dict(view=tag,pooled_change=pooled,project_wins=wins,pass_gate=pooled>.003 and wins>=2))
    return pooled>.003 and wins>=2


def patch(pipe):
    net,tok=frozen("canonical_official")
    pipe.encoder,pipe.tok=net,tok
    return pipe


def infer(pipe,texts,view,normalized):
    from small import scores
    from comment_head import predict as predict_mat
    total=time.perf_counter()
    p,cost=None,None
    if view!="aug_norm":
        p,_,cost=scores(pipe.encoder,pipe.tok,texts)
    if view!="aug_raw":
        q,_,other=scores(pipe.encoder,pipe.tok,normalized)
        if p is None:
            p,cost=q,other
        else:
            p=.5*(p+q)
            cost={k:cost[k]+other[k] for k in cost}
    began=time.perf_counter()
    mat,_,n=predict_mat(texts,pipe.mat["chosen"],pipe.mat_weights,pipe.mat_bias)
    cost["forward_seconds"]+=time.perf_counter()-began
    cost["analytic_gflops"]+=2*n/1e9
    cost["wall_seconds"]=time.perf_counter()-total
    return np.column_stack((p[:,0],mat)),cost


def tune():
    pipe=patch(base_pipe())
    report=json.loads((OUT/"eligibility.json").read_text())
    view=report["view"]
    _,_,texts,y,ids,_=dataset(full=True)
    norm=canonicalize(texts) if view!="aug_raw" else None
    p,_=infer(pipe,texts,view,norm)
    with np.load(ROOT/"round8/frozen/holdout.npz") as saved:
        if saved["rows"].tolist()!=ids:
            raise ValueError("Official holdout order differs")
        folds=saved["folds"]
    cal,review=folds==3,folds==4
    mat=pipe.mat["chosen"]["threshold"]
    candidates=[]
    for t in np.linspace(.15,.85,71):
        pred=p>(t,mat)
        candidates.append((metric(y[cal],pred[cal])["mean"]["f1"],-abs(t-.5),float(t)))
    _,_,t=max(candidates)
    write_json(OUT/"policy.json",dict(view=view,thresholds=[t,mat],review=metric(y[review],p[review]>(t,mat))))


def public():
    cfg=json.loads((OUT/"policy.json").read_text())
    texts,y,fields=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    start=time.perf_counter()
    norm=canonicalize(texts) if cfg["view"]!="aug_raw" else None
    prep=time.perf_counter()-start if norm is not None else 0.
    cluster=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    expected=predict_bits(ROOT/"round11/cold_shift/predictions.csv",ids,{"thresholds":(.5,.5)})
    pipe=patch(base_pipe())
    trials=[]
    result=None
    for _ in range(3):
        p,cost=infer(pipe,texts,cfg["view"],norm)
        cost["wall_seconds"]+=prep
        cost["forward_seconds"]+=prep
        bits=p>cfg["thresholds"]
        if result is not None and not np.array_equal(result,bits):
            raise ValueError("Augmentation prediction changed on repeat")
        result=bits
        trials.append(cost)
        print(cost,flush=True)
    c={k:median(q[k] for q in trials) for k in trials[0]}
    m=metric(y,result)
    save(OUT/"public",y,ids,np.zeros(len(y),dtype=bool),p,c,cfg["thresholds"])
    write_json(OUT/"analysis.json",dict(metrics=m,cost=c,trials=trials,
                     local_formula=competition_formula(m["mean"]["f1"],c["forward_seconds"],c["analytic_gflops"]),
                     interval_vs_round11=bootstrap_delta(y,expected,result,cluster)))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("eligible","tune","public"))
    args=ap.parse_args()
    if args.stage=="eligible":
        raise SystemExit(0 if eligible() else 3)
    tune() if args.stage=="tune" else public()
