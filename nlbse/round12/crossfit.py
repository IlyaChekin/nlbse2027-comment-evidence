"""Test project-balanced feature fusion with leave-one-development-project-out calibration."""
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from project_heads import OUT as HEADS, calibration, channels, corpus, model_for, primary, structural
from project_split import HERE, checksum
from students import group_codes, metric
from pipeline import write_json
from round2_report import bootstrap_delta

OUT=HERE/"crossfit"
MODES=("logit","token_motifs","ast_motifs","all")
ENCODERS=("compact","unix")


def weighting(projects):
    values,number=np.unique(projects,return_counts=True)
    weights={k:len(projects)/len(values)/n for k,n in zip(values,number,strict=True)}
    return np.asarray([weights[k] for k in projects])


def tune():
    _,y,project,_,roles=corpus()
    probe,dev=roles["probe"],roles["dev"]
    a,b=(primary(part) for part in ("probe","dev"))
    da,sa=structural("probe")
    db,sb=structural("dev")
    groups=project[dev]
    y0,y1=y[probe,0],y[dev,0]
    condensed=[]
    heads={}
    results={}
    for name in ENCODERS:
        results[name]={}
        for feature in MODES:
            x0=channels(a[name][:,0],da,sa,feature)
            x1=channels(b[name][:,0],db,sb,feature)
            p=np.empty(len(dev),dtype=np.float32)
            for omitted in np.unique(groups):
                keep=groups!=omitted
                x=np.vstack((x0,x1[keep]))
                labels=np.r_[y0,y1[keep]]
                weight=weighting(np.r_[project[probe],groups[keep]])
                model=model_for("logistic").fit(x,labels,logisticregression__sample_weight=weight)
                p[~keep]=model.predict_proba(x1[~keep])[:,1]
            record=calibration(y1,p,groups)
            record["average_precision"]=average_precision_score(y1,p)
            all_x=np.vstack((x0,x1))
            weight=weighting(np.r_[project[probe],groups])
            model=model_for("logistic").fit(all_x,np.r_[y0,y1],logisticregression__sample_weight=weight)
            heads[f"{name}/{feature}"]=model
            results[name][feature]=record
        ranked=max(MODES[1:],key=lambda k:results[name][k]["objective"])
        condensed.append((name,"logit",ranked))
    OUT.mkdir(parents=True,exist_ok=True)
    artifact=OUT/"weights.joblib"
    joblib.dump(heads,artifact)
    write_json(OUT/"policy.json",dict(selected=condensed,table=results,sha256=checksum(artifact)))


def test():
    cfg=json.loads((OUT/"policy.json").read_text())
    if checksum(OUT/"weights.joblib")!=cfg["sha256"]:
        raise ValueError("Cross-fitted weights mismatch")
    heads=joblib.load(OUT/"weights.joblib")
    _,y,projects,paths,roles=corpus()
    i=roles["test"]
    group=projects[i]
    labels=y[i]
    files=group_codes(zip(group,[paths[j] for j in i],strict=True))
    primary_p=primary("test")
    dense,motifs=structural("test")
    raw_cfg=json.loads((HEADS/"policy.json").read_text())["choices"]
    raw={name:primary_p[name][:,0]>raw_cfg[name]["candidates"]["raw"]["cutoff"] for name in ENCODERS}
    common=(raw["compact"]!=labels[:,0])&(raw["unix"]!=labels[:,0])
    result={}
    for name,*selected in cfg["selected"]:
        with (HEADS/f"{name}_raw_test.csv").open(newline="") as stream:
            saved=list(csv.DictReader(stream))
        if [int(r["row"]) for r in saved]!=i.tolist():
            raise ValueError("Unexpected heldout order")
        mat=np.asarray([int(r["pred_MAT"]) for r in saved],dtype=bool)
        primary_reference=np.column_stack((raw[name],mat))
        new=[]
        for feature in selected:
            x=channels(primary_p[name][:,0],dense,motifs,feature)
            p=heads[f"{name}/{feature}"].predict_proba(x)[:,1]
            bits=np.column_stack((p>cfg["table"][name][feature]["cutoff"],mat))
            path=OUT/f"{name}_{feature}.csv"
            with path.open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(("row","project","p_vulnerability","pred_vulnerability","pred_MAT"))
                w.writerows(zip(i,group,p,bits[:,0].astype(int),mat.astype(int),strict=True))
            record=dict(f1=metric(labels,bits),by_project={g:metric(labels[group==g],bits[group==g]) for g in np.unique(group)},
                        average_precision=average_precision_score(labels[:,0],p),
                        common_fixed=int((common&(bits[:,0]==labels[:,0])).sum()),
                        interval_vs_base=bootstrap_delta(labels,primary_reference,bits,files))
            if new:
                record["interval_vs_matched_logit"]=bootstrap_delta(labels,new[0],bits,files)
            new.append(bits)
            result[f"{name}/{feature}"]=record
            print(name,feature,record["f1"]["vulnerability"]["f1"],flush=True)
    write_json(OUT/"analysis.json",result)


if __name__=="__main__":
    tune()
    test()
