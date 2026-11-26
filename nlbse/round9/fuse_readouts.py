"""Check whether the reused early logits improve decisions after a late update."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

from cache_hybrid import ACTIONS, HERE
from frontier import JointHead, as_common_scale, calibrated, metric
from pipeline import write_json


def features(early,late):
    a=logit(np.clip(early,1e-6,1-1e-6))
    b=logit(np.clip(late,1e-6,1-1e-6))
    return np.column_stack((a,b))


def analyze(output):
    stage=HERE/"frozen"
    gate=json.loads((stage/"policy.json").read_text())
    with np.load(stage/"holdout.npz") as fitted, np.load(HERE/"cache/holdout.npz") as frozen:
        y=fitted["y"]
        fit,cal,review=fitted["folds"]<=2,fitted["folds"]==3,fitted["folds"]==4
        early=as_common_scale(fitted["early"],gate["thresholds"])
        results={}
        for name in ACTIONS:
            late=frozen[name]
            x=features(early,late)
            choices={}
            for strength in (.015,.06,.25):
                head=JointHead.fit(x[fit],y[fit],strength)
                p=head(x)
                t=calibrated(y[cal],p[cal])
                choices[strength]=dict(
                    head=head.to_dict(),thresholds=t,
                    calibration=metric(y[cal],p[cal]>t),
                    review=metric(y[review],p[review]>t))
            selected=max(choices,key=lambda k:choices[k]["calibration"]["mean"]["f1"])
            results[name]=dict(
                selected=selected,candidates=choices,
                baseline_calibration=metric(y[cal],late[cal]>.5),
                baseline_review=metric(y[review],late[review]>.5))
            print(name, "baseline",results[name]["baseline_review"]["mean"]["f1"],
                  "fused",choices[selected]["review"]["mean"]["f1"],flush=True)
    write_json(output,results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=HERE/"readout_fusion.json")
    analyze(p.parse_args().output)
