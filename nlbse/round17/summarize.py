"""Reconstruct fixed-budget tables and measured scores from the archived predictions."""

import csv
import json
import math
from pathlib import Path
from statistics import median
import sys


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round16"))
from summarize import compare,counts,f1,rows


def read(path):
    return json.loads(path.read_text())


def write(name, fields):
    with (HERE/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields[0])
        w.writeheader()
        w.writerows(fields)


def project_suite(directories, target, routed=False):
    table=[]
    for folder in directories:
        data=read(folder/"results.json")
        for name,row in data["modes"].items():
            frame=rows(folder/f"{name}_test.csv")
            if len(frame)!=data["rows"]:
                raise ValueError("Incomplete project-held-out decisions")
            equal=compare(frame,"label","before_equal","after_equal",row["matched" if routed else "equal"])
            frozen=compare(frame,"label","before_frozen","after_frozen",row["frozen"])
            count=sum(x["routed"]=="1" for x in frame)
            if count!=row.get("n_routed",row.get("routed")) or equal["fp"]!=data["budget"] or equal["delta_fp"]:
                raise ValueError("Project routing or false-positive figure changed")
            result=dict(seed=folder.name,mode=name,selected=name==data["selected"],rows=len(frame),fp=equal["fp"],
                        matched_tp=equal["delta_tp"],matched_f1=equal["f1"],frozen_tp=frozen["delta_tp"],
                        frozen_fp=frozen["delta_fp"],routed=count)
            if not routed:
                longer=[v for v in frame if int(v["length"])>512]
                shorter=[v for v in frame if int(v["length"])<=512]
                long=compare(longer,"label","before_equal","after_equal",row["long"])
                short=compare(shorter,"label","before_equal","after_equal",row["short"])
                result.update(long_tp=long["delta_tp"],short_tp=short["delta_tp"],long_rows=len(longer),
                              eligible=row["gap"]["eligible"],mean_view_gap=row["gap"]["abs_gap"],raw_only_tp=row["raw_only"]["delta_tp"])
            table.append(result)
    if table:
        write(target,table)
        print(f"Verified {len(table)} comparisons for {target}",flush=True)


def measured(folder,target):
    result=read(folder/"results.json")
    policy=read(folder/"policy.json")
    if set(result)!={"reference",policy["selected"]}:
        raise ValueError("The publicly measured model was not frozen internally")
    ref=rows(folder/"reference_public.csv")
    if len(ref)!=30655 or [int(r["row"]) for r in ref]!=list(range(30655)):
        raise ValueError("Incomplete public split")
    report=[]
    for name,row in result.items():
        frame=rows(folder/f"{name}_public.csv")
        for a,b in zip(ref,frame,strict=True):
            if any(a[k]!=b[k] for k in ("row","label_vulnerability","label_mat","mat")):
                raise ValueError("Public labels or MAT decisions changed")
            b["old"]=a["vulnerability"]
        m=compare(frame,"label_vulnerability","reference_equal_fp","equal_fp",row["matched"])
        d=compare(frame,"label_vulnerability","old","vulnerability",row["frozen"])
        if m["delta_fp"]:
            raise ValueError("Public matched threshold changed the FP budget")
        vuln=d["f1"]
        mat=f1(counts(frame,"label_mat","mat"))
        avg=(vuln+mat)/2
        med={k:median(v[k] for v in row["trials"]) for k in row["cost"]}
        score=.6*avg+.2*max(1-med["forward_seconds"]/1000,0)+.2*max(1-med["analytic_gflops"]/3000000,0)
        if med!=row["cost"] or len(row["trials"])!=3 or not math.isclose(score,row["score"],abs_tol=1e-12):
            raise ValueError("Public paired timing receipt differs")
        report.append(dict(mode=name,rows=len(frame),vulnerability_f1=vuln,mat_f1=mat,mean_f1=avg,score=score,**med,
                           fixed_budget=m["fp"],matched_tp=m["delta_tp"],frozen_tp=d["delta_tp"],frozen_fp=d["delta_fp"]))
    write(target,report)
    print(f"Verified {len(report)} public scores for {target}",flush=True)


def latent_probe():
    folder=HERE/"latent_probe"
    record=read(folder/"results.json")
    frame=rows(folder/"predictions.csv")
    if len(frame)!=record["rows"]:
        raise ValueError("Latent-neighbor predictions do not cover the held-out rows")
    equal=compare(frame,"label","before_equal","after_equal",record["matched"])
    frozen=compare(frame,"label","before_frozen","after_frozen",record["frozen"])
    if equal["delta_fp"] or equal["fp"]!=record["fp"]:
        raise ValueError("Latent-neighbor comparison changed the FP budget")
    write("latent_probe.csv",[dict(rows=len(frame),fp=equal["fp"],matched_tp=equal["delta_tp"],matched_f1=equal["f1"],
        frozen_tp=frozen["delta_tp"],frozen_fp=frozen["delta_fp"],routed=record["routed"])])
    print("Verified the latent-neighbor comparison",flush=True)


if __name__=="__main__":
    for source,target in (("projects","views.csv"),("representation","representation.csv")):
        folders=sorted(p.parent for p in (HERE/source).glob("*/results.json"))
        project_suite(folders,target)
    routed=HERE/"router/projects"
    if (routed/"results.json").is_file():
        project_suite([routed],"router.csv",routed=True)
    for source,target in (("official/evaluation","public_views.csv"),("router/official","public_router.csv")):
        folder=HERE/source
        if (folder/"results.json").is_file():
            measured(folder,target)
    if (HERE/"latent_probe/results.json").is_file():
        latent_probe()
