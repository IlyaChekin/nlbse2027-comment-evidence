"""Measure the zero-cost ceiling of switching among existing binary predictors."""
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round10"))
from students import ROOT, dataset, metric, write_json
from neighbor_eval import table


def main():
    _,_,_,y,ids,_=dataset(public=True)
    paths={"compact": ROOT/"round10/compact_exit/lexical_deferred/predictions.csv",
           "unix": ROOT/"round10/unix_context/unix_lexical/predictions.csv",
           "neighbor_unix": HERE/"neighbors/unixcoder9/predictions.csv",
           "tiny": HERE/"small/public/predictions.csv",
           "adaptive": HERE/"cascade/adaptive/predictions.csv"}
    candidates={}
    for name,path in paths.items():
        if not path.is_file():
            continue
        data=table(path)
        if data[:,0].tolist()!=ids:
            raise ValueError(f"Unexpected rows in {path}")
        candidates[name]=data[:,2].astype(bool)
    mat=table(paths["unix"])[:,3].astype(bool)
    sets=[("tiny","unix"),("compact","unix"),("compact","unix","neighbor_unix"),tuple(candidates)]
    result={}
    for names in sets:
        correct=np.stack([candidates[name]==y[:,0] for name in names]).any(axis=0)
        picked=np.logical_xor(~correct,y[:,0])
        values=metric(y,np.column_stack((picked,mat)))
        result["_".join(names)]=dict(optimistic_F1=values["mean"]["f1"],
            optimistic_vulnerability=values["vulnerability"]["f1"],
            zero_cost_ceiling=.4+.6*values["mean"]["f1"],
            vulnerability_FN_all=int((~correct&(y[:,0]==1)).sum()),
            vulnerability_FP_all=int((~correct&(y[:,0]==0)).sum()))
    result["minimum_F1_for_0_97"]={"mean":(.97-.4)/.6,
        "vulnerability_zero_cost_fixed_mat":2*(.97-.4)/.6-metric(y,np.column_stack((mat,mat)))["MAT"]["f1"]}
    write_json(HERE/"ceiling.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
