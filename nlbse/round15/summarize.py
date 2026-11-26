"""Verify paired project-held-out decisions and regenerate the precision table."""

import csv
import json
import math
from pathlib import Path
from statistics import median
import sys


HERE=Path(__file__).resolve().parent


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def measure(rows,reference,candidate):
    tp=fp=fn=tn=0
    n_tp=n_fp=n_fn=n_tn=0
    suppressed=0
    for row in rows:
        y,b,p=(row[key]=="1" for key in ("label",reference,candidate))
        tp+=y and b
        fp+=not y and b
        fn+=y and not b
        tn+=not y and not b
        n_tp+=y and p
        n_fp+=not y and p
        n_fn+=y and not p
        n_tn+=not y and not p
        suppressed+=b and not p and row["shared_high_fp"]=="1"
    return ({"tp":tp,"fp":fp,"fn":fn,"tn":tn},{"tp":n_tp,"fp":n_fp,"fn":n_fn,"tn":n_tn},suppressed)


def validate(rows, result, columns):
    before,after,suppressed=measure(rows,*columns)
    if (before,after,suppressed)!=(result["before"],result["after"],result["corrected_confident_fp"]):
        raise ValueError("Archived prediction decisions differ from reported metrics")


def summarize(suite,file,name,columns,meta,chosen):
    rows=read(file)
    exact,locked=meta["equal"],meta["frozen"]
    validate(rows,exact,columns["equal"])
    validate(rows,locked,columns["frozen"])
    if exact["delta_fp"]!=0:
        raise ValueError("Equal-FP comparison changed the budget")
    return dict(suite=suite,mode=name,selected_on_dev=chosen,rows=len(rows),matched_fps=exact["before"]["fp"],
                vulnerability_f1_matched=exact["f1_after"],matched_delta_tp=exact["delta_tp"],
                matched_hard_fp_removed=exact["corrected_confident_fp"],frozen_delta_tp=locked["delta_tp"],
                frozen_delta_fp=locked["delta_fp"],frozen_vulnerability_f1=locked["f1_after"],
                frozen_ci_lower=meta["fixed_cutoff_interval"][0] if "fixed_cutoff_interval" in meta else meta["fixed_interval"][0],
                frozen_ci_upper=meta["fixed_cutoff_interval"][1] if "fixed_cutoff_interval" in meta else meta["fixed_interval"][1])


def write_table(name,rows):
    with (HERE/name).open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def summarize_disagreement():
    folder=HERE/"disagreement"
    data=json.loads((folder/"results.json").read_text())
    base=json.loads((folder/"policy.json").read_text())["reference_threshold"]
    table=[]
    for mode,output in data.items():
        rows=read(folder/f"{mode.replace('/','_')}.csv")
        for r in rows:
            c,u=float(r["p_compact"]),float(r["p_unix"])
            r["base_frozen"]=str(int(c>base))
            r["shared_high_fp"]=str(int(c>.9 and u>.9 and r["label"]=="0"))
        validate(rows,output["exact"],("ref_equal","exact_fp"))
        validate(rows,output["frozen"],("base_frozen","frozen"))
        if output["exact"]["delta_fp"] or sum(r["routed"]=="1" for r in rows)!=output["routed"]:
            raise ValueError("Disagreement budget or number of teacher queries changed")
        table.append(dict(mode=mode,rows=len(rows),teacher_rows=output["routed"],proxy_share=output["proxy_share"],
                          fp_budget=output["exact"]["before"]["fp"],matched_delta_tp=output["exact"]["delta_tp"],
                          matched_f1=output["exact"]["f1_after"],frozen_delta_tp=output["frozen"]["delta_tp"],
                          frozen_delta_fp=output["frozen"]["delta_fp"]))
    write_table("disagreement_table.csv",table)
    print(f"Verified {len(table)} selective teacher comparisons.",flush=True)


def checked_score(rows,record,metrics,labels):
    label_scores=[]
    for key,label,predictor in zip(("vulnerability","MAT"),labels,("vulnerability","mat"),strict=True):
        gold,pred=[[row[col]=="1" for row in rows] for col in (label,predictor)]
        tp=sum(y and p for y,p in zip(gold,pred,strict=True))
        f1=2*tp/max(sum(gold)+sum(pred),1)
        if not math.isclose(f1,metrics[key]["f1"],abs_tol=1e-12):
            raise ValueError("Public F1 disagrees with archived decisions")
        label_scores.append(f1)
    profile={field:median(x[field] for x in record["trials"]) for field in record["cost"]}
    if len(record["trials"])!=3 or profile!=record["cost"]:
        raise ValueError("Incomplete public runtime measurements")
    mean=sum(label_scores)/2
    expected=.6*mean+.2*max(1-profile["forward_seconds"]/1000,0)+.2*max(1-profile["analytic_gflops"]/3_000_000,0)
    if not math.isclose(expected,record["score"],abs_tol=1e-12):
        raise ValueError("Archived score does not match the measured formula")
    return dict(rows=len(rows),vulnerability_f1=label_scores[0],mat_f1=label_scores[1],mean_f1=mean,
                score=expected,gflops=profile["analytic_gflops"],wall_seconds=profile["wall_seconds"],
                forward_seconds=profile["forward_seconds"])


def summarize_public():
    folder=HERE/"routed_public"
    reported=json.loads((folder/"results.json").read_text())
    release=read(folder/"release.csv")
    if len(release)!=30655 or [int(r["row"]) for r in release]!=list(range(len(release))):
        raise ValueError("Public test does not contain the full reference split")
    table=[]
    for name,r in reported.items():
        rows=read(folder/f"{name}.csv")
        stable=("row","label_vulnerability","label_mat","mat")
        if any(any(a[k]!=b[k] for k in stable) for a,b in zip(release,rows,strict=True)):
            raise ValueError("Changed public rows, labels or reference MAT decisions")
        budget=lambda records:sum(k["label_vulnerability"]=="0" and k["vulnerability_equal_fp"]=="1" for k in records)
        if budget(release)!=budget(rows):
            raise ValueError("Public equal-FP reconstruction changed the budget")
        count=sum(row["routed"]=="1" for row in rows)
        if count!=r["cost"]["teacher_rows"]:
            raise ValueError("Incorrect allocation of teacher queries")
        numeric=checked_score(rows,r,r["metrics"],("label_vulnerability","label_mat"))
        table.append(dict(mode=name,**numeric,teacher_rows=count,
                          equal_fp_delta_tp=r["matched_fp"]["delta_tp"],frozen_delta_tp=r["frozen"]["delta_tp"],
                          frozen_delta_fp=r["frozen"]["delta_fp"]))
    write_table("routed_public_table.csv",table)
    print(f"Verified {len(table)} public score calculations.",flush=True)


def summarize_distillation():
    folder=HERE/"official_distill"
    reported=json.loads((folder/"results.json").read_text())
    policy=json.loads((folder/"policy.json").read_text())
    if set(reported)!={"release",policy["selected"]}:
        raise ValueError("The publicly run model differs from the frozen selection")
    release=read(folder/"release_public.csv")
    if len(release)!=30655 or [int(r["row"]) for r in release]!=list(range(len(release))):
        raise ValueError("Public test does not contain the full reference split")
    table=[]
    stable=("row","truth_vulnerability","truth_mat","mat")
    fixed_path=folder/"fixed_fp.json"
    fixed=json.loads(fixed_path.read_text()) if fixed_path.is_file() else None
    for name,r in reported.items():
        rows=read(folder/f"{name}_public.csv")
        if any(any(a[k]!=b[k] for k in stable) for a,b in zip(release,rows,strict=True)):
            raise ValueError("Distilled comparison changed public rows, labels or MAT decisions")
        t=checked_score(rows,r,r["quality"],("truth_vulnerability","truth_mat"))
        matched_delta=""
        if fixed:
            aligned=read(folder/f"{name}_equal_fp.csv")
            for current, archived in zip(aligned,rows,strict=True):
                if any(current[k]!=archived[k] for k in ("row","truth_vulnerability")):
                    raise ValueError("Public equal-FP rows differ from the measured run")
                current["label"]=current["truth_vulnerability"]
                current["shared_high_fp"]="0"
            report=fixed["matched"][name]
            validate(aligned,report,("reference_matched","matched"))
            if report["delta_fp"] or report["before"]["fp"]!=fixed["budget"]:
                raise ValueError("Single encoder used a different matched FP budget")
            matched_delta=report["delta_tp"]
        table.append(dict(mode=name,**t,delta_tp=r["vulnerability"]["delta_tp"],delta_fp=r["vulnerability"]["delta_fp"],
                          equal_fp_delta_tp=matched_delta))
    write_table("distilled_public_table.csv",table)
    print(f"Verified {len(table)} single-encoder public score calculations.",flush=True)


def main():
    summarize_disagreement()
    if sys.argv[1:]==["--only-disagreement"]:
        return
    results=[]
    for path in sorted((HERE/"precision").glob("results_*.json")):
        data=json.loads(path.read_text())
        seed=data["seed"]
        tag=data.get("tag",str(seed))
        selected=json.loads((path.parent/f"policy_{tag}.json").read_text())["chosen"]
        csv_name="test_advisor.csv" if data.get("study")=="advisor" else "test.csv"
        for name,r in data["modes"].items():
            cols={"equal":("reference_equal","candidate_equal"),"frozen":("reference_frozen","candidate_frozen")}
            item=summarize(f"neural_{tag}",path.parent/f"{seed}_{name}"/csv_name,name,cols,r,selected==name)
            item["matched_delta_tp_vs_continuation"]=r["against_control"]["delta_tp"]
            results.append(item)
    veto=HERE/"veto"
    for name,r in json.loads((veto/"results.json").read_text())["modes"].items():
        cols={"equal":("base_equal","new_equal"),"frozen":("base_frozen","new_frozen")}
        item=summarize("sparse_veto",veto/f"{name}.csv",name,cols,r,True)
        item["matched_delta_tp_vs_continuation"]=""
        results.append(item)
    if not any(r["suite"].startswith("neural") for r in results):
        raise ValueError("Missing neural precision comparisons")
    write_table("results.csv",results)
    print(f"Verified {len(results)} matched and frozen comparisons.",flush=True)
    if (HERE/"routed_public/results.json").is_file():
        summarize_public()
    if (HERE/"official_distill/results.json").is_file():
        summarize_distillation()


if __name__=="__main__":
    main()
