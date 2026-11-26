"""Check that train-only similarity adds value when every evaluation project is unseen."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from project_heads import OUT as HEADS, calibration, corpus, primary
from project_split import HERE, ROOT, checksum
from build_index import export
from neighbor_eval import features, neighbors
from students import group_codes, metric
from pipeline import write_json
from round2_report import bootstrap_delta

OUT=HERE/"retrieval"


def extract(parts):
    texts,y,_,_,rows=corpus()
    OUT.mkdir(parents=True,exist_ok=True)
    executable=OUT/"score"
    if not executable.is_file():
        subprocess.run(["c++","-std=c++17","-O3",str(ROOT/"round11/neighbors.cpp"),"-o",str(executable)],check=True)
    fit=OUT/"fit.bin"
    if not fit.is_file():
        export(fit,texts,rows["fit"],y)
    command=[str(executable),str(fit)]
    for part in parts:
        q=OUT/f"{part}.bin"
        export(q,texts,rows[part])
        command.extend((str(q),str(OUT/f"{part}.f32")))
    log=subprocess.run(command,check=True,text=True,capture_output=True).stdout
    (OUT/("_".join(parts)+"_timing.log")).write_text(log)
    print(log,end="",flush=True)


def tune():
    _,y,projects,_,rows=corpus()
    names=("compact","unix")
    p0,p1=(primary(part) for part in ("probe","dev"))
    a,b=(neighbors(OUT/f"{part}.f32",len(rows[part])) for part in ("probe","dev"))
    models={}
    cfg={}
    for name in names:
        train=features(a,p0[name][:,0])
        dev=features(b,p1[name][:,0])
        head=LogisticRegression(C=.2,max_iter=400).fit(train,y[rows["probe"],0])
        dev_p=head.predict_proba(dev)[:,1]
        models[name]=head
        cfg[name]=calibration(y[rows["dev"],0],dev_p,projects[rows["dev"]])
    file=OUT/"weights.joblib"
    joblib.dump(models,file)
    write_json(OUT/"policy.json",dict(choices=cfg,checkpoint_sha256=checksum(file)))


def test():
    _,y,projects,paths,rows=corpus()
    ids=rows["test"]
    policy=json.loads((OUT/"policy.json").read_text())
    file=OUT/"weights.joblib"
    if checksum(file)!=policy["checkpoint_sha256"]:
        raise ValueError("Transfer weights changed")
    models=joblib.load(file)
    x=neighbors(OUT/"test.f32",len(ids))
    p=primary("test")
    raw_baselines=json.loads((HEADS/"policy.json").read_text())["choices"]
    base_v={k:p[k][:,0]>raw_baselines[k]["candidates"]["raw"]["cutoff"] for k in p}
    common=(base_v["compact"]!=y[ids,0])&(base_v["unix"]!=y[ids,0])
    group=projects[ids]
    cluster=group_codes(zip(group,[paths[i] for i in ids],strict=True))
    result={}
    for name,model in models.items():
        with (HEADS/f"{name}_raw_test.csv").open(newline="") as f:
            prediction=list(csv.DictReader(f))
        if [int(row["row"]) for row in prediction]!=ids.tolist():
            raise ValueError("Wrong holdout rows in comparator")
        mat=np.asarray([int(row["pred_MAT"]) for row in prediction],dtype=bool)
        q=model.predict_proba(features(x,p[name][:,0]))[:,1]
        v=q>policy["choices"][name]["cutoff"]
        bits=np.column_stack((v,mat))
        comparator=np.column_stack((base_v[name],mat))
        result[name]=dict(scores=metric(y[ids],bits), interval=bootstrap_delta(y[ids],comparator,bits,cluster),
             by_project={g:metric(y[ids[group==g]],bits[group==g]) for g in np.unique(group)},
             common_fixed=int((common&(v==y[ids,0])).sum()))
        with (OUT/f"{name}_test_predictions.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(("row","project","p_vulnerability","pred_vulnerability","pred_MAT"))
            w.writerows(zip(ids,group,q,v.astype(int),mat.astype(int),strict=True))
    result["nearest_coverage"]={str(t):float((x[:,4:6].max(1)>t).mean()) for t in (.5,.75,.9)}
    write_json(OUT/"analysis.json",result)
    print({k:v.get("scores",v) for k,v in result.items()},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("features","tune","test"))
    args=ap.parse_args()
    if args.stage=="features":
        extract(("probe","dev"))
    elif args.stage=="tune":
        tune()
    else:
        extract(("test",))
        test()
