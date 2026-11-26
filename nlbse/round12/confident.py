"""Learn an extra readout for confident negatives using only probe and development labels."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logit

from project_heads import OUT as HEADS, binary_f1, corpus, model_for, primary, structural
from project_split import HERE, checksum
from students import group_codes, metric
from round2_report import bootstrap_delta
from pipeline import write_json

OUT=HERE/"confident"


def inputs(p,dense,motif):
    probabilities=logit(np.column_stack((p,motif)).clip(1e-7,1-1e-7))
    return np.column_stack((probabilities,dense[:,:20]))


def target_score(y,p,project):
    unweighted=np.mean([binary_f1(y[project==g],p[project==g]) for g in np.unique(project)])
    return .5*(binary_f1(y,p)+unweighted)


def tune():
    _,y,project,_,split=corpus()
    probe,dev=split["probe"],split["dev"]
    d0,m0=structural("probe")
    d1,m1=structural("dev")
    model_policy=json.loads((HEADS/"policy.json").read_text())["choices"]
    p0,p1=(primary(role) for role in ("probe","dev"))
    artifacts={}
    evidence={}
    for name in ("compact","unix"):
        a,b=p0[name][:,0],p1[name][:,0]
        limit=model_policy[name]["candidates"]["raw"]["cutoff"]
        select=(a<.25)&(a<=limit)
        x=inputs(a,d0,m0)
        head=model_for("logistic").fit(x[select],y[probe[select],0])
        q=head.predict_proba(inputs(b,d1,m1))[:,1]
        base=b>limit
        base_score=target_score(y[dev,0],base,project[dev])
        rows=[]
        for cap in (.035,.05,.1,.2,.25):
            for quantile in (.25,.5,.75,1.):
                length=float(np.quantile(d0[:,0],quantile)) if quantile<1. else None
                selected=(b<cap)&~base
                if length is not None:
                    selected&=d1[:,0]<=length
                for threshold in np.r_[np.linspace(.025,.90,36),1.0]:
                    v=base|(selected&(q>threshold))
                    s=target_score(y[dev,0],v,project[dev])
                    rows.append(dict(cap=cap,cutoff=float(threshold),length=length,dev_score=s,
                                     active=int(selected.sum()),added=int((v&~base).sum())))
        chosen=max(rows,key=lambda x:(x["dev_score"],-x["added"],-x["active"]))
        evidence[name]=dict(baseline=base_score,selection=chosen, candidates=rows,base_threshold=limit)
        artifacts[name]=head
    OUT.mkdir(parents=True,exist_ok=True)
    f=OUT/"weights.joblib"
    joblib.dump(artifacts,f)
    write_json(OUT/"policy.json",dict(engines=evidence,sha256=checksum(f)))


def test():
    cfg=json.loads((OUT/"policy.json").read_text())
    if checksum(OUT/"weights.joblib")!=cfg["sha256"]:
        raise ValueError("The confident-negative readout changed")
    models=joblib.load(OUT/"weights.joblib")
    _,labels,g,paths,split=corpus()
    ids=split["test"]
    y=labels[ids]
    p=primary("test")
    x,m=structural("test")
    groups=g[ids]
    files=group_codes(zip(groups,[paths[i] for i in ids],strict=True))
    raw={n:p[n][:,0]>v["base_threshold"] for n,v in cfg["engines"].items()}
    common=(y[:,0]==1)&~raw["compact"]&~raw["unix"]
    report={}
    for name,head in models.items():
        info=cfg["engines"][name]
        row=info["selection"]
        confidence=p[name][:,0]
        q=head.predict_proba(inputs(confidence,x,m))[:,1]
        mask=(confidence<row["cap"])&~raw[name]
        if row["length"] is not None:
            mask&=x[:,0]<=row["length"]
        added=mask&(q>row["cutoff"])
        with (HEADS/f"{name}_raw_test.csv").open(newline="") as f:
            saved=list(csv.DictReader(f))
        if [int(z["row"]) for z in saved]!=ids.tolist():
            raise ValueError("Wrong cohort rows")
        mat=np.asarray([int(z["pred_MAT"]) for z in saved],dtype=bool)
        base=np.column_stack((raw[name],mat))
        v=raw[name]|added
        bits=np.column_stack((v,mat))
        report[name]=dict(metrics=metric(y,bits),by_project={z:metric(y[groups==z],bits[groups==z]) for z in np.unique(groups)},
                    interval=bootstrap_delta(y,base,bits,files),recovered_common=int((added&common).sum()),
                    additional_false_positives=int((added&(y[:,0]==0)).sum()),active=int(mask.sum()))
        with (OUT/f"{name}_test.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(("row","project","p_aux","pred_vulnerability","pred_MAT"))
            writer.writerows(zip(ids,groups,q,v.astype(int),mat.astype(int),strict=True))
    write_json(OUT/"analysis.json",report)
    print(report,flush=True)


if __name__=="__main__":
    tune()
    test()
