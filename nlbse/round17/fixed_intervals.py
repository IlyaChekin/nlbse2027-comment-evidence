"""Estimate fixed-FP uncertainty without resampling individual functions independently."""

import csv

import numpy as np

from view_train import HERE,SEEDS,inputs,json_at,write_json
from budget_interval import resampled_intervals
from evaluate import counts


def main():
    _,labels,projects,paths,split=inputs()
    ids=split["test"]
    y=labels[ids,0]
    files=[paths[i] for i in ids]
    detail={}
    for seed in SEEDS:
        folder=HERE/"projects"/str(seed)
        meta=json_at(folder/"results.json")
        for name in ("full","paired","consistent"):
            with (folder/f"{name}_test.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            with (folder/"reference_test.csv").open(newline="") as f:
                ref=list(csv.DictReader(f))
            if ([int(r["row"]) for r in rows]!=ids.tolist() or [int(r["row"]) for r in ref]!=ids.tolist()
                or [int(r["label"]) for r in rows]!=y.tolist()):
                raise ValueError("Fixed-budget bootstrap rows differ")
            fitted=[np.asarray([float(r["score"]) for r in t]) for t in (ref,rows)]
            baseline=np.asarray([r["before_equal"]=="1" for r in rows])
            if counts(y,baseline)["fp"]!=meta["budget"]:
                raise ValueError("Wrong fixed-FP baseline")
            interval=resampled_intervals(y,projects[ids],files,fitted,baseline,np.zeros(len(y),dtype=bool),draws=2000)
            detail[f"{seed}/{name}"]=interval
            print(seed,name,interval["delta_tp"],flush=True)
    write_json(HERE/"fixed_intervals.json",detail)


if __name__=="__main__":
    main()
