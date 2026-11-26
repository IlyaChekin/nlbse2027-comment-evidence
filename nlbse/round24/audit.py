"""Recount the paired public change and audit untouched rows and remaining errors."""
import csv
from pathlib import Path

import numpy as np

from internal_sparse import HERE,ROOT,f,checksum,json_at,write_json
from pipeline import PUBLIC,load
from round2_report import bootstrap_delta
from students import group_codes


def public_rows(path):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=list(range(len(rows))):
        raise ValueError("Public decision order changed")
    return (np.asarray([[int(r[k]) for k in ("label_vulnerability","label_mat")] for r in rows]),
            np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")] for r in rows],dtype=bool),
            np.asarray([int(r["at_same_fp"]) for r in rows],dtype=bool))


def main():
    folder=HERE/"public"
    receipt=json_at(folder/"results.json")
    current=receipt["selected"]
    results=receipt["results"]
    base=folder/"reference.csv"
    choice=folder/f"{current.replace(':','_')}.csv"
    if any(checksum(file)!=results[name]["predictions_sha256"] for name,file in (("reference",base),(current,choice))):
        raise ValueError("Archived predictions changed")
    y,a,_=public_rows(base)
    target,b,diag=public_rows(choice)
    _,gold,meta=load(PUBLIC,audit=True)
    if not np.array_equal(y,target) or not np.array_equal(y,gold) or checksum(PUBLIC)!=receipt["public_sha256"]:
        raise ValueError("The report no longer matches public labels")
    info=json_at(ROOT/"round22/union_public/results.json")
    trace=ROOT/"round22/union_public/scanner_predictions.npz"
    if checksum(trace)!=info["scanner_predictions_sha256"]:
        raise ValueError("Archive rule outcomes changed")
    with np.load(trace) as z:
        actual=z["gate"].copy()
        actual[actual]=z["flag"]==0
    if len(y)!=30655 or actual.sum()!=results[current]["cost"]["extra_calls"]:
        raise ValueError("Unexpected prediction or extra call count")
    if (a[~actual]!=b[~actual]).any() or (a[:,1]!=b[:,1]).any():
        raise ValueError("The correction touched a non-target decision")
    if np.any(diag[~actual]!=a[~actual,0]):
        raise ValueError("The fixed-FP diagnostic changed decisions outside its cohort")
    scores=[f.metric(y,z) for z in (a,b)]
    if any(abs(m["mean"]["f1"]-results[key]["quality"]["mean"]["f1"])>1e-12
           for m,key in zip(scores,("reference",current),strict=True)):
        raise ValueError("Paired metric does not reproduce")
    with (ROOT/"round20/conditional_audit.csv").open(newline="") as stream:
        records=list(csv.DictReader(stream))
    if [int(r["row"]) for r in records]!=list(range(len(y))):
        raise ValueError("Full rule pass differs in row order")
    exact_all=np.asarray([int(r["any_rule"]) for r in records],dtype=bool)
    mode={"queried_rule_absent":actual,"not_queried_rule_absent":~actual&~exact_all,"rule_fired":exact_all}
    counts={k:f.comparison(y[rows,0],a[rows,0],b[rows,0]) for k,rows in mode.items()}
    fp_before=f.counts(y[:,0],a[:,0])["fp"]
    fp_diag=f.counts(y[:,0],diag)["fp"]
    if fp_before!=fp_diag:
        raise ValueError("Matched diagnostic did not preserve FP")
    groups=group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    ci=bootstrap_delta(y,a,b,groups)
    cost=results[current]["cost"]
    score=results[current]["score"]
    need=(.97-score)/.3
    result=dict(scope="full public",file_cluster_95_percent_delta=ci,remaining=counts,
                hash=checksum(choice),required_vulnerability_f1_if_all_else_fixed=scores[1]["vulnerability"]["f1"]+need,
                score_with_zero_time_and_operations=.4+.6*scores[1]["mean"]["f1"],
                goal_gap=.97-score)
    write_json(HERE/"public/audit.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
