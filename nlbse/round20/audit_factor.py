"""Audit the frozen policy with and without exact rules on the same public functions."""
import csv
import time

import joblib
import numpy as np

from factor import (HERE, SOURCE, Policy, base, choose, comparison, counts, estimate,
                    exact_budget, factor_data, json_at, scanner, small, valid, weak, write_json)
from pipeline import PUBLIC,load,metric
from score_runtime import competition_formula


def main():
    folder=HERE/"factor_numeric/official"
    cfg=json_at(folder/"policy.json")
    entry=cfg["modes"][cfg["selected"]]
    program=HERE/"scan_extended.mjs"
    if base.checksum(program)!=cfg["program_sha256"] or base.checksum(HERE/"scan.mjs")!=cfg["scanner_sha256"]:
        raise ValueError("The frozen static scanner changed")
    profile=json_at(folder/"results.json")[cfg["selected"]]
    texts,gold,_=load(PUBLIC)
    values,mat,_=factor_data("official",texts,small.tokenizer())
    model=joblib.load(folder/"heads.joblib")
    calibrated=estimate(model,values)
    policy=Policy(**entry["policy"])
    prior,no,yes,asked=choose(values,calibrated,policy,entry["gate"])
    start=time.perf_counter()
    columns,raw=scanner(texts,program=program)
    runtime=time.perf_counter()-start
    expected=json_at(SOURCE/"policy.json")["rule_columns"]+["rule_195"]
    if columns!=expected or len(raw)!=len(gold):
        raise ValueError("The complete static scan returned different rows or rules")
    answer=np.where(valid(raw),np.where(weak(raw),yes,no),prior)
    guess=prior.astype(np.float32)
    exact=answer.astype(np.float32)
    frozen=np.where(asked,exact,guess)
    y=gold[:,0]
    cut=entry["threshold"]
    binding=frozen>cut
    with (folder/f"{cfg['selected']}_public.csv").open(newline="") as f:
        reference=list(csv.DictReader(f))
    if [int(r["row"]) for r in reference]!=list(range(len(gold))):
        raise ValueError("Saved decisions changed order")
    if not np.array_equal(binding,[r["vulnerability"]=="1" for r in reference]):
        raise ValueError("The audit failed to reproduce the actual public predictions")
    if not np.array_equal(mat,[r["mat"]=="1" for r in reference]):
        raise ValueError("The audit failed to reproduce the actual MAT predictions")
    fp=counts(y,binding)["fp"]
    p,_=exact_budget(frozen,y,fp)
    known,_=exact_budget(exact,y,fp)
    without,_=exact_budget(guess,y,fp)
    choice=np.where((exact>cut)==y,exact>cut,guess>cut)
    summary=dict(rows=len(y),scanned_by_frozen=int(asked.sum()),complete_scan_seconds=runtime,
                 budget=fp,all_rules_at_frozen_threshold=comparison(y,binding,exact>cut),
                 no_queries_at_same_fp=comparison(y,p,without),
                 all_queries_at_same_fp=comparison(y,p,known),
                 oracle_correct_branch_at_frozen_threshold=comparison(y,binding,choice))
    estimate_score=metric(gold,np.column_stack((choice,mat)))["mean"]["f1"]
    cost=profile["cost"]
    summary["oracle_score_at_frozen_cost"]=competition_formula(estimate_score,cost["forward_seconds"],cost["analytic_gflops"])
    out=HERE/"conditional_audit.csv"
    with out.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("row","vulnerability_gold","asked","any_rule","valid","p_unqueried","p_known","p_frozen"))
        w.writerows(zip(range(len(y)),y,asked.astype(int),weak(raw).astype(int),valid(raw).astype(int),
                         guess,exact,frozen,strict=True))
    summary["predictions_sha256"]=base.checksum(out)
    write_json(HERE/"conditional_audit.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    main()
