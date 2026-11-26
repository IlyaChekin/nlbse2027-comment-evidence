"""Collect project-disjoint and official results with explicit evaluation-scope labels."""
import csv
import json

import numpy as np

from project_heads import corpus
from project_split import HERE
from pipeline import metric, write_json


def read(path):
    return json.loads(path.read_text())


def verify(path,ids,y,expected):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=ids.tolist():
        raise ValueError(f"Unexpected prediction order in {path}")
    p=np.asarray([[int(r["pred_vulnerability"]),int(r["pred_MAT"])] for r in rows])
    for label,values in metric(y,p).items():
        if abs(values["f1"]-expected[label]["f1"])>1e-12:
            raise ValueError(f"Published F1 doesn't match {path}")


def main():
    source=read(HERE/"heads/analysis.json")
    policy=read(HERE/"heads/policy.json")
    _,all_y,_,_,split=corpus()
    ids=split["test"]
    y=all_y[ids]
    rows=[]

    def add(scope,model,metrics,cost=None,score=None):
        rows.append(dict(scope=scope,model=model,vulnerability_f1=metrics["vulnerability"]["f1"],
                         mat_f1=metrics["MAT"]["f1"],mean_f1=metrics["mean"]["f1"],
                         wall_seconds=None if cost is None else cost["wall_seconds"],
                         forward_seconds=None if cost is None else cost["forward_seconds"],
                         gflops=None if cost is None else cost["analytic_gflops"],local_formula=score))

    for key,item in source["tests"].items():
        family,mode=key.split("/")
        verify(HERE/"heads"/f"{family}_{mode}_test.csv",ids,y,item["scores"])
        add("heldout_pooled",key,item["scores"])
        for name,project in item["by_project"].items():
            add(f"heldout_{name}",key,project)
    for folder,namespace in (("confident","confident"),("retrieval","retrieval"),("crossfit","crossfit")):
        data=read(HERE/folder/"analysis.json")
        for k,v in data.items():
            if "by_project" not in v:
                continue
            scores=v.get("metrics",v.get("scores",v.get("f1")))
            add("heldout_pooled",f"{namespace}/{k}",scores)
            for group,project in v["by_project"].items():
                add(f"heldout_{group}",f"{namespace}/{k}",project)
    aug=read(HERE/"augmentation/analysis.json")
    for name,trial in aug["results"].items():
        verify(HERE/"augmentation"/f"{name}.csv",ids,y,trial["scores"])
        add("heldout_pooled",name,trial["scores"],trial["costs"])
        for g,r in trial["by_project"].items():
            add(f"heldout_{g}",name,r)
    for folder in ("official","official_aug","fast_mat","efficient_aug"):
        path=HERE/folder/"analysis.json"
        if not path.is_file():
            continue
        data=read(path)
        records={folder:data} if folder=="official_aug" else data
        for name,q in records.items():
            add("official_public",f"{folder}/{name}",q["metrics"],q["cost"],q["local_formula"])
    with (HERE/"results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    selected=policy["choices"]["compact"]["frozen"]
    best=source["tests"][f"compact/{selected}"]
    record=dict(holdout_rows=len(y),shared_errors=source["common_errors"],
        selected_compact_feature=selected,baseline_compact=source["tests"]["compact/raw"]["scores"],
        augmented_compact=best["scores"],
        public_results=[r for r in rows if r["scope"]=="official_public"])
    write_json(HERE/"summary.json",record)
    print(f"Verified {len(y)} labels per saved project comparison. Exported {len(rows)} metric rows.",flush=True)


if __name__=="__main__":
    main()
