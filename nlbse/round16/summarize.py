"""Recalculate reported F1, fixed budgets and profiled scores from saved CSVs."""

import csv
import json
import math
from pathlib import Path
from statistics import median


HERE=Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def counts(frame,label,prediction):
    y,p=[[r[c]=="1" for r in frame] for c in (label,prediction)]
    return dict(tp=sum(v and w for v,w in zip(y,p,strict=True)),fp=sum(not v and w for v,w in zip(y,p,strict=True)),
                fn=sum(v and not w for v,w in zip(y,p,strict=True)),tn=sum(not v and not w for v,w in zip(y,p,strict=True)))


def f1(c):
    return 2*c["tp"]/max(2*c["tp"]+c["fp"]+c["fn"],1)


def compare(frame,label,ref,candidate,report):
    a,b=counts(frame,label,ref),counts(frame,label,candidate)
    if report["before"]!=a or report["after"]!=b:
        raise ValueError("Saved confusion counts disagree with predictions")
    if report["delta_tp"]!=b["tp"]-a["tp"] or report["delta_fp"]!=b["fp"]-a["fp"]:
        raise ValueError("Saved confusion-count deltas are incorrect")
    return dict(delta_tp=report["delta_tp"],delta_fp=report["delta_fp"],f1=f1(b),fp=b["fp"])


def write(name,table):
    with (HERE/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=table[0])
        w.writeheader()
        w.writerows(table)


def project(directory="projects",target="project_results.csv"):
    table=[]
    for output in sorted((HERE/directory).glob("*/results.json")):
        report=load(output)
        dev=load(output.parent/"policy.json")["dev"]
        for name,item in report["modes"].items():
            frame=rows(output.parent/f"test_{name}.csv")
            match=compare(frame,"label","reference_equal","candidate_equal",item["matched"])
            control=compare(frame,"label","control_equal","candidate_equal",item["vs_control"])
            frozen=compare(frame,"label","reference_frozen","candidate_frozen",item["frozen"])
            if match["delta_fp"] or control["delta_fp"] or match["fp"]!=report["fp"]:
                raise ValueError("Invalid project-disjoint false-positive budget")
            table.append(dict(seed=output.parent.name,mode=name,dp=dev[name]["matched"]["delta_tp"],fp=match["fp"],
                              matched_delta_tp=match["delta_tp"],delta_tp_vs_control=control["delta_tp"],
                              matched_f1=match["f1"],frozen_delta_tp=frozen["delta_tp"],frozen_delta_fp=frozen["delta_fp"]))
    write(target,table)
    print(f"Verified {len(table)} runs in {directory}",flush=True)


def context():
    folder=HERE/"context"
    table=[]
    data=load(folder/"results.json")
    for name,item in data["modes"].items():
        frame=rows(folder/f"{name}_test.csv")
        matched=compare(frame,"label","equal_before","equal_after",item["matched"])
        frozen=compare(frame,"label","frozen_before","frozen_after",item["frozen"])
        if matched["delta_fp"] or matched["fp"]!=data["fp"]:
            raise ValueError("Invalid long-context false-positive budget")
        table.append(dict(mode=name,selected=name==data["selected"],affected=data["affected"],fp=matched["fp"],
                          matched_delta_tp=matched["delta_tp"],matched_f1=matched["f1"],
                          frozen_delta_tp=frozen["delta_tp"],frozen_delta_fp=frozen["delta_fp"],
                          context_gflops=item["cost"]["analytic_gflops"]))
    write("context_results.csv",table)
    print(f"Verified {len(table)} long-context comparisons",flush=True)


def ablation():
    table=[]
    for source in sorted((HERE/"ablation").glob("*/results.json")):
        for model,entry in load(source).items():
            for gain,result in entry["test"].items():
                frame=rows(source.parent/f"{model}_{float(gain):g}.csv")
                matched=compare(frame,"label","off_equal","candidate_equal",result["matched"])
                if matched["delta_fp"]:
                    raise ValueError("A head ablation changed the false-positive budget")
                table.append(dict(seed=source.parent.name,mode=model,gain=gain,selected=float(gain)==entry["selected"],
                                  delta_tp_from_same_trunk=matched["delta_tp"],
                                  delta_tp_from_reference=result["against_reference"]["delta_tp"],matched_f1=matched["f1"],fp=matched["fp"]))
    write("ablation_results.csv",table)
    print(f"Verified {len(table)} head ablations",flush=True)


def official(directory="official",target="public_results.csv"):
    folder=HERE/directory
    data=load(folder/"results.json")
    selected=load(folder/"policy.json")["selected"]
    if set(data)!={"reference",selected}:
        raise ValueError("Final public run differs from frozen selection")
    baseline=rows(folder/"reference_public.csv")
    same=("row","label_vulnerability","label_mat","mat")
    if [int(r["row"]) for r in baseline]!=list(range(30655)):
        raise ValueError("Missing public data in measured run")
    table=[]
    for name,item in data.items():
        frame=rows(folder/f"{name}_public.csv")
        if any(any(a[k]!=b[k] for k in same) for a,b in zip(baseline,frame,strict=True)):
            raise ValueError("Public labels, rows or MAT predictions changed")
        v=f1(counts(frame,"label_vulnerability","vulnerability"))
        m=f1(counts(frame,"label_mat","mat"))
        cost={k:median(q[k] for q in item["trials"]) for k in item["cost"]}
        score=.6*(v+m)/2+.2*max(1-cost["forward_seconds"]/1000,0)+.2*max(1-cost["analytic_gflops"]/3e6,0)
        if len(item["trials"])!=3 or cost!=item["cost"] or not math.isclose(score,item["score"],abs_tol=1e-12):
            raise ValueError("Invalid public benchmark")
        matched=compare(frame,"label_vulnerability","reference_equal_fp","equal_fp",item["matched"])
        if matched["delta_fp"]:
            raise ValueError("Invalid public false-positive budget")
        for row,ref in zip(frame,baseline,strict=True):
            row["frozen_reference"]=ref["vulnerability"]
        frozen=compare(frame,"label_vulnerability","frozen_reference","vulnerability",item["frozen"])
        table.append(dict(mode=name,rows=len(frame),vulnerability_f1=v,mat_f1=m,mean_f1=(v+m)/2,score=score,**cost,
                          matched_tp=matched["delta_tp"],frozen_tp=frozen["delta_tp"],frozen_fp=frozen["delta_fp"]))
    write(target,table)
    print(f"Verified {len(table)} public score calculations in {directory}",flush=True)


if __name__=="__main__":
    project()
    if list((HERE/"memory_projects").glob("*/results.json")):
        project("memory_projects","memory_results.csv")
    if list((HERE/"ranking_projects").glob("*/results.json")):
        project("ranking_projects","ranking_results.csv")
    if (HERE/"context/results.json").is_file():
        context()
    if list((HERE/"ablation").glob("*/results.json")):
        ablation()
    if (HERE/"official/results.json").is_file():
        official()
    if (HERE/"context_public/results.json").is_file():
        official("context_public","context_public_results.csv")
