"""Verify the measured predictions and calculate a paired interval and remaining errors."""
import csv
from statistics import median

import numpy as np

from length_eval import HERE,ROOT,R24,f,checksum,json_at,write_json
from pipeline import PUBLIC,load
from round2_report import bootstrap_delta
from students import group_codes


def read(path):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=list(range(len(rows))):
        raise ValueError("Predictions were reordered")
    return np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")] for r in rows],dtype=bool),np.asarray(
        [float(r["probability"]) for r in rows],dtype=np.float32)


def main():
    out=HERE/"views/public"
    result=json_at(out/"results.json")
    policy=json_at(out/"policy.json")
    name=result["selected"]
    option=policy["support"][name]
    source=json_at(R24/"official/tuned/policy.json")
    cut=source["candidates"][source["selected"]]["threshold"]
    paths={key:out/f"{key}.csv" for key in ("reference",name)}
    for key,file in paths.items():
        info=result["results"][key]
        if checksum(file)!=info["predictions_sha256"]:
            raise ValueError("Public decision archive changed")
        if any(v!=median(r[k] for r in info["trials"]) for k,v in info["cost"].items()):
            raise ValueError("Paired medians differ from the trials")
    a,orig=read(paths["reference"])
    b,_=read(paths[name])
    texts,y,meta=load(PUBLIC,audit=True)
    if checksum(PUBLIC)!=result["public_sha256"]:
        raise ValueError("Official public data changed")
    src=ROOT/"round22/union_public/scanner_predictions.npz"
    if checksum(src)!=json_at(ROOT/"round22/union_public/results.json")["scanner_predictions_sha256"]:
        raise ValueError("Archived exact decisions changed")
    with np.load(src) as z:
        eligibility=z["gate"].copy()
        eligibility[eligibility]=z["flag"]==0
    chosen=eligibility.copy()
    if option["branch"]!="all":
        length=np.fromiter((len(s) for s in texts),dtype=np.int32)
        chosen &= (length<=policy["length_cutoff"])==(option["branch"]=="short")
    if option["distance"] is not None:
        chosen &= abs(orig-cut)<option["distance"]
    if np.any(a[~chosen]!=b[~chosen]) or np.any(a[:,1]!=b[:,1]):
        raise ValueError("An unselected function or its MAT output changed")
    p=result["results"][name]
    if int(chosen.sum())!=p["cost"]["segment_calls"]:
        raise ValueError("Audited branch differs from measured requests")
    q=f.metric(y,b)
    if any(abs(q[k]["f1"]-p["quality"][k]["f1"])>1e-12 for k in q):
        raise ValueError("Measured F1 does not reproduce")
    fwd=p["cost"]["forward_seconds"]
    ops=p["cost"]["analytic_gflops"]
    if abs(f.competition_formula(q["mean"]["f1"],fwd,ops)-p["score"])>1e-12:
        raise ValueError("Saved score differs from measured costs")
    groups=group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    receipt=dict(interval=bootstrap_delta(y,a,b,groups),selected_rows=int(chosen.sum()),
                 altered_decisions=int((a[:,0]!=b[:,0]).sum()),full_exact_negative=int(eligibility.sum()),
                 inside_rule_negative=f.comparison(y[:,0][eligibility],a[eligibility,0],b[eligibility,0]),
                 required_vulnerability_f1_if_others_fixed=q["vulnerability"]["f1"]+(.97-p["score"])/.3,
                 gap=.97-p["score"],profile_sha256=checksum(out/"results.json"))
    write_json(out/"audit.json",receipt)
    print(receipt,flush=True)


if __name__=="__main__":
    main()
