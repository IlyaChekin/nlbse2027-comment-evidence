"""Fit a small counterfactual router on the known outcomes of frozen readouts."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round9"))
from cache_hybrid import ACTIONS
from frontier import metric
from pipeline import write_json
from tune_hybrid import costs,marginal_target

BASE=HERE.parent/"round9"
PENALTIES=(0,.05,.1,.2,.3,.4,.6,.8,1.,1.25,1.6,2.,3.,4.,6.)


def features(p,width,tags):
    return np.column_stack((p,p*(1-p),np.log1p(width),tags))


def fit():
    output=HERE/"risk_frozen"
    output.mkdir(exist_ok=True)
    rules=json.loads((BASE/"frozen/asymmetric.json").read_text())["options"]["asym_score"]["rule"]
    cut=json.loads((BASE/"frozen/policy.json").read_text())["thresholds"]
    with np.load(BASE/"frozen/holdout.npz") as first,np.load(BASE/"cache/holdout.npz") as tail:
        if not np.array_equal(first["rows"],tail["ids"]):
            raise ValueError("Mismatched examples")
        fold,y=first["folds"],first["y"]
        early=first["early"]
        tags=tail["features"][:,-5:].astype(np.float32)
        x=features(early,first["width"],tags)
        base,prices=costs(first["width"])
        p=np.stack((early>cut,*(tail[k]>.5 for k in ACTIONS)),axis=1)
    fit,cal,review=fold<=2,fold==3,fold==4
    utility=marginal_target(y[fit],p[fit])
    tree=[HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=12,min_samples_leaf=60,
                                       l2_regularization=2,learning_rate=.075,random_state=42)
          .fit(x[fit],utility[:,i]) for i in (1,2)]
    g=np.column_stack((np.zeros(len(x)),*(h.predict(x) for h in tree)))
    scale=float(prices[cal,1].mean())
    indices=np.arange(len(x))
    origin=np.where((early[:,0]>rules["lower"])&(early[:,0]<rules["upper"]),2,0)
    comparator=metric(y[review],p[indices,origin][review])
    results={}
    for penalty in PENALTIES:
        route=(g-penalty*prices/scale).argmax(1)
        out=p[indices,route]
        gflops=base+prices[indices,route]
        cal_f1=metric(y[cal],out[cal])["mean"]["f1"]
        normalized=gflops[cal].sum()/(base[cal]+prices[cal,1]).sum()
        results[penalty]=dict(mean=cal_f1,review=metric(y[review],out[review]),
                             stop=float((route[cal]==0).mean()),cal_cost_ratio=normalized,
                             objective=.6*cal_f1-.2*normalized*485224/3e6)
    selected=max(results,key=lambda t:results[t]["objective"])
    headroom=metric(y[cal],p[indices,origin][cal])["mean"]["f1"]-.0015
    guarded=[t for t,s in results.items() if s["mean"]>=headroom]
    safe=max(guarded,key=lambda t:results[t]["objective"]) if guarded else selected
    joblib.dump(tree,output/"gate.joblib")
    write_json(output/"policy.json",dict(score=selected,quality=safe,scale=scale,grid=results,
                                        baseline_review=comparator,
                                        notes="Oracle targets use only the fit fold of the official train holdout"))
    print("risk review base",comparator["mean"]["f1"],"choice",
          {t:results[t] for t in (selected,safe)},flush=True)


if __name__=="__main__":
    fit()
