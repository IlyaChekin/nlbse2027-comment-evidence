"""Select separate late-update budgets for false-negative and false-positive risk."""

import argparse
import json
from pathlib import Path

import numpy as np

from cache_hybrid import ACTIONS, HERE
from frontier import metric
from pipeline import write_json
from tune_hybrid import costs

NEGATIVE = (.025,.05,.075,.1,.125,.15,.175,.2,.225,.25)
POSITIVE = (.63,.68,.72,.75,.78,.82,.86,.90,.94,.97)


def actions(early, lower, upper, index):
    return np.where((early[:,0]<=lower)|(early[:,0]>=upper),0,index)


def main(output):
    gate=json.loads((HERE/"frozen/policy.json").read_text())
    with np.load(HERE/"frozen/holdout.npz") as states, np.load(HERE/"cache/holdout.npz") as targets:
        if not np.array_equal(states["rows"],targets["ids"]):
            raise ValueError("Cached groups do not match")
        cal,review=states["folds"]==3,states["folds"]==4
        y=states["y"]
        early=states["early"]
        z=np.stack((early>gate["thresholds"],*(targets[key]>.5 for key in ACTIONS)),axis=1)
        prefix,late=costs(states["width"])
        ids=np.arange(len(y))
        calibration_volume=(prefix[cal]+late[cal,1]).mean()
        full={"calibration":metric(y[cal],z[cal,1]),"review":metric(y[review],z[review,1])}
        floor=full["calibration"]["mean"]["f1"]
        table=[]
        for index in (1,2):
            for lower in NEGATIVE:
                for upper in POSITIVE:
                    action=actions(early[cal],lower,upper,index)
                    p=z[cal][np.arange(action.size),action]
                    f1=metric(y[cal],p)["mean"]["f1"]
                    op=(prefix[cal]+late[cal][np.arange(action.size),action]).mean()
                    adjusted=.6*f1-.2*(op/calibration_volume)*485224/3e6
                    table.append(dict(kind="asymmetric",lower=lower,upper=upper,late=index,score=adjusted,
                                      f1=f1,stop=float((action==0).mean()),cost_ratio=op/calibration_volume))
        scoring=max(table,key=lambda k:k["score"])
        near=[row for row in table if row["f1"]>=floor-.001 and row["stop"]>=.1]
        options={"asym_score":scoring}
        if near:
            options["asym_quality"]=max(near,key=lambda k:k["score"])
        else:
            options["asym_quality"]=max(table,key=lambda k:(k["f1"],k["score"]))
        heldout={}
        for name,cfg in options.items():
            action=actions(early,cfg["lower"],cfg["upper"],cfg["late"])
            block=z[ids,action]
            heldout[name]=dict(rule=cfg,review=metric(y[review],block[review]),
                               counts=np.bincount(action[review],minlength=3).tolist())
            print(name,cfg,heldout[name]["review"]["mean"]["f1"],flush=True)
        write_json(output,dict(options=heldout,reference=full,grid=table))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=HERE/"frozen/asymmetric.json")
    main(p.parse_args().output)
