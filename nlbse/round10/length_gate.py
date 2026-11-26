"""Test whether the confidence interval should depend on the number of tokens."""

import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round9"))
from cache_hybrid import ACTIONS
from frontier import metric
from pipeline import write_json
from tune_hybrid import costs

BASE=HERE.parent/"round9"


def main():
    previous=json.loads((BASE/"frozen/asymmetric.json").read_text())["options"]["asym_score"]["rule"]
    gate=json.loads((BASE/"frozen/policy.json").read_text())
    with np.load(BASE/"frozen/holdout.npz") as a,np.load(BASE/"cache/holdout.npz") as b:
        if not np.array_equal(a["rows"],b["ids"]):
            raise ValueError("Misaligned cached rows")
        y=a["y"]
        widths=a["width"]
        p=a["early"]
        folds=a["folds"]
        probability=np.stack((p>gate["thresholds"],*(b[k]>.5 for k in ACTIONS)),axis=1)
    prefix,late=costs(widths)
    cal,review=folds==3,folds==4
    power=np.clip(np.log2((widths+16)/100),-1.5,2.)
    all_rows=np.arange(len(y))
    ref=(p[:,0]>previous["lower"])&(p[:,0]<previous["upper"])
    ref_scores=metric(y[review],probability[all_rows,ref.astype(np.int8)*2][review])
    baseline_cost=np.mean(prefix[cal]+late[cal,1])
    result=[]
    for index in (1,2):
        for slope in (-.10,-.055,-.025,0.,.025,.055,.1):
            for lo in (.02,.035,.06,.09):
                for hi in (.8,.84,.86,.9,.93):
                    lower=np.clip(lo+slope*power,.005,.38)
                    upper=np.clip(hi-slope*power,.55,.995)
                    need=(p[:,0]>lower)&(p[:,0]<upper)
                    decisions=probability[all_rows,need.astype(int)*index]
                    quality=metric(y[cal],decisions[cal])["mean"]["f1"]
                    volume=prefix+late[all_rows,need.astype(int)*index]
                    estimate=.6*quality-.2*(volume[cal].mean()/baseline_cost)*485224/3e6
                    result.append(dict(index=index,slope=slope,low=lo,high=hi,quality=quality,estimate=estimate,
                                       review=metric(y[review],decisions[review]),
                                       review_cost=float(volume[review].mean())))
    best=max(result,key=lambda v:v["estimate"])
    print("baseline review",ref_scores["mean"]["f1"],"length route",best,flush=True)
    write_json(HERE/"length_gate.json",dict(reference_review=ref_scores,chosen=best,candidates=result))


if __name__=="__main__":
    main()
