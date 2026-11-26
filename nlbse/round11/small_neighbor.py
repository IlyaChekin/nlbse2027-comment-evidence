"""Fit a neighbor-aware readout for the single lightweight neural encoder."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import small
from neighbor_eval import (OUT, ROOT, features, neighbors, table, cutoff, control, MAT_REFERENCE,
                           dataset, metric, save, sha256, write_json, competition_formula)


def study(width):
    source=small.run_directory(width)
    out=OUT/f"small{width}"
    out.mkdir(parents=True,exist_ok=True)
    record=json.loads((source/"selected.json").read_text())
    path=source/f"{Path(record['checkpoint']).stem}_holdout.npz"
    data=np.load(path)
    folds=data["folds"]
    p=data["p"][:,0]
    y=data["y"]
    fit,cal,review=folds<=2,folds==3,folds==4
    x=features(neighbors(OUT/"holdout.f32",len(y)),p)
    candidates=[]
    for name,model in (("logistic",LogisticRegression(C=.2,max_iter=400)),
                       ("trees",HistGradientBoostingClassifier(learning_rate=.075,max_iter=155,
                        max_leaf_nodes=12,min_samples_leaf=90,l2_regularization=1.,random_state=42))):
        model.fit(x[fit],y[fit,0])
        q=model.predict_proba(x)[:,1]
        t=cutoff(y[cal],q[cal])
        candidates.append((dict(name=name,threshold=t,calibration=control(y[cal],q[cal],t),
                                review=control(y[review],q[review],t)),model))
    chosen,model=max(candidates,key=lambda item:item[0]["calibration"])
    freeze=dict(selected=chosen,reevaluation=[t[0] for t in candidates],
                baseline_review=control(y[review],p[review],record["thresholds"][0]),
                checkpoint_sha256=record["sha256"])
    write_json(out/"policy.json",freeze)
    _,_,_,y,ids,cal=dataset(public=True)
    reference=table(MAT_REFERENCE)
    neural=np.load(source/"public.npz")
    if neural["ids"].tolist()!=ids or reference[:,0].tolist()!=ids or not np.array_equal(neural["y"],y):
        raise ValueError("Labels or indices are misaligned")
    start=time.perf_counter()
    q=model.predict_proba(features(neighbors(OUT/"public.f32",len(y)),neural["p"][:,0]))[:,1]
    readout=time.perf_counter()-start
    profile=json.loads((source/"public/analysis.json").read_text())["cost"]
    query=float((OUT/"query.log").read_text().splitlines()[-1].split()[-1])
    cost=dict(profile,query_seconds=query,readout_seconds=readout)
    cost["wall_seconds"]+=query+readout
    cost["forward_seconds"]+=query+readout
    p=np.column_stack((q,reference[:,3]))
    t=(chosen["threshold"],.5)
    metrics=metric(y,p>t)
    save(out,y,ids,cal,p,cost,t)
    result=dict(metrics=metrics,cost=cost,score=competition_formula(metrics["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]))
    write_json(out/"analysis.json",result)
    print(freeze,result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,default=128,choices=(128,256))
    study(parser.parse_args().width)
