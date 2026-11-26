"""Bootstrap equal-FP improvements by file, with the project mix fixed."""

import csv
import json

import numpy as np

from crossfit import HERE, write_json
from budget_interval import resampled_intervals
from evaluate import counts, exact_budget, inputs


def main():
    directory=HERE/"disagreement"
    policy=json.loads((directory/"policy.json").read_text())
    entries=json.loads((directory/"results.json").read_text())
    _,labels,projects,paths,split=inputs()
    ids=split["test"]
    label=labels[ids,0]
    files=[paths[i] for i in ids]
    scope=np.zeros(len(ids),dtype=bool)
    results={}
    for name in entries:
        with (directory/f"{name.replace('/','_')}.csv").open(newline="") as stream:
            data=list(csv.DictReader(stream))
        if [int(r["row"]) for r in data]!=ids.tolist() or [int(r["label"]) for r in data]!=label.tolist():
            raise ValueError("Held-out rows changed before bootstrapping")
        p,q=[np.asarray([float(r[key]) for r in data]) for key in ("p_compact","p_fused")]
        base=p>policy["reference_threshold"]
        target=counts(label,base)["fp"]
        old,_=exact_budget(p,label,target)
        new,_=exact_budget(q,label,target)
        if [int(r["ref_equal"]) for r in data]!=old.astype(int).tolist() or [int(r["exact_fp"]) for r in data]!=new.astype(int).tolist():
            raise ValueError("Exact-FP decisions could not be replayed")
        result=resampled_intervals(label,projects[ids],files,[p,q],base,scope,draws=2000)
        results[name]=dict(interval=result,draws=2000)
        print(name,result["delta_tp"],result["delta_f1"],flush=True)
    write_json(directory/"intervals.json",results)


if __name__=="__main__":
    main()
