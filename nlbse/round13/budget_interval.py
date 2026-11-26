"""Recompute equal-FP recall on paired file bootstrap draws within each project."""
import csv
import json

import numpy as np

from evaluate import inputs, OUT
from rotate import HERE
from pipeline import write_json


def read(path,ids):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=ids.tolist():
        raise ValueError("Study CSV disagrees with the frozen rows")
    p=[np.asarray([float(r[k]) for r in rows]) for k in ("base_score","candidate_score")]
    baseline=np.asarray([int(r["base_pred"]) for r in rows],dtype=bool)
    target=np.asarray([int(r["joint_low"]) for r in rows],dtype=bool)
    short=np.asarray([int(r["short"]) for r in rows],dtype=bool)
    return p,baseline,target,short


def resampled_intervals(y,projects,paths,weights,baseline,target,draws=1500):
    sources={}
    cluster=[]
    for g,path in zip(projects,paths,strict=True):
        key=(g,path)
        cluster.append(sources.setdefault(key,len(sources)))
    by_project={g:np.asarray([i for (p,_),i in sources.items() if p==g]) for g in np.unique(projects)}
    cluster=np.asarray(cluster)
    ranks=[np.argsort(-p,kind="stable") for p in weights]
    indices=[(r,y[r]==0,target[r]) for r in ranks]
    incorrect=(y==0)&baseline
    rng=np.random.default_rng(42)
    sample=[]
    for _ in range(draws):
        draws_ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in by_project.values()])
        multiplicity=np.bincount(draws_ix,minlength=len(sources))[cluster]
        budget=multiplicity[incorrect].sum()
        positives=int(multiplicity[y==1].sum())
        summary=[]
        for r,neg,within in indices:
            ww=multiplicity[r]
            prefix=np.cumsum(ww*neg)
            last=np.searchsorted(prefix,budget+1,side="left")
            tp=int(ww[:last][~neg[:last]].sum())
            fixed=int(ww[:last][(~neg&within)[:last]].sum())
            f1=2*tp/max(tp+budget+positives,1)
            summary.append((tp,fixed,f1))
        sample.append(np.asarray(summary[1])-summary[0])
    limits=np.quantile(sample,[.025,.975],axis=0).T.tolist()
    return dict(zip(("delta_tp","delta_target_tp","delta_f1"),limits,strict=True))


def main():
    _,labels,project,paths,roles=inputs()
    rows=roles["test"]
    config=json.loads((OUT/"analysis.json").read_text())
    names=list(dict.fromkeys([*(f"{key}/{value}" for key,value in config["selections"].items()),
                              "joint_low/bindings","compact_low/bindings"]))
    inputs_to_check={name:OUT/(name.replace("/","_")+".csv") for name in names}
    inputs_to_check.update({f"reweight/{mode}":HERE/"reweight"/mode/"predictions.csv"
                            for mode in ("matched","focused")})
    files=np.asarray([paths[i] for i in rows])
    result={}
    for name,path in inputs_to_check.items():
        p,b,target,short=read(path,rows)
        result[name]={}
        for key,mask in (("all",np.ones(len(rows),dtype=bool)),("short_only",short)):
            result[name][key]=resampled_intervals(labels[rows[mask],0],project[rows[mask]],files[mask],
                                      [v[mask] for v in p],b[mask],target[mask])
    write_json(HERE/"fixed_budget_intervals.json",result)
    print(result,flush=True)

if __name__=="__main__":
    main()
