"""Verify cohort budgets and recount the calibrated public result from saved decisions."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def groups():
    variants=[]
    for phase in ("test_a","test_b"):
        for source in ("comparison","conditioned"):
            rec=read(HERE/phase/source/"results.json")
            for name,model in rec["options"].items():
                by=model["project"] if source=="comparison" else model["per_project"]
                assert all(x["before"]["fp"]==x["after"]["fp"] for x in by.values())
                assert model["project_tp"]==sum(x["delta_tp"] for x in by.values())
                dev=model.get("calibration",model.get("dev"))
                variants.append((phase,source,name,model["project_tp"],dev["delta_tp"],
                    model["frozen"]["delta_tp"],model["frozen"]["delta_fp"],model["test_queries"]))
    with (HERE/"experiments.csv").open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("rotation","family","name","tp_same_fp","tp_calibration",
                    "tp_frozen","fp_frozen","sparse_calls"))
        w.writerows(variants)


def public():
    report=read(HERE/"public/results.json")
    coef=report["results"][report["selected"]]
    for name,r in report["results"].items():
        path=HERE/"public"/f"{name.replace(':','_')}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r["predictions_sha256"]
        cost=r["cost"]
        assert all(cost[k]==median(x[k] for x in r["trials"]) for k in cost)
        num=.6*r["quality"]["mean"]["f1"]+.2*(1-cost["forward_seconds"]/1000)+.2*(1-cost["analytic_gflops"]/3e6)
        assert abs(r["score"]-num)<1e-12
    now=coef["frozen"]["after"]
    forecast=read(HERE/"public/audit.json")
    scoped=forecast["remaining"]["queried_rule_absent"]["after"]
    n=now["tp"]+now["fn"]
    extra=scoped["fn"]
    removable=scoped["fp"]
    tp=now["tp"]+extra
    fp=now["fp"]-removable
    best_f1=2*tp/(tp+fp+n)
    best_score=coef["score"]+.3*(best_f1-coef["quality"]["vulnerability"]["f1"])
    target=forecast["required_vulnerability_f1_if_all_else_fixed"]
    needed_tp=math.ceil(target*(n+fp)/(2-target))-now["tp"]
    diag=dict(score=coef["score"],correctable_fn_in_observed_negative=extra,
              correctable_fp_in_observed_negative=removable,
              fixed_cost_oracle_score=best_score,
              target_f1_if_all_else_fixed=target,
              remaining_tp_needed_after_removing_all_scope_fp=needed_tp)
    (HERE/"target_gap.json").write_text(json.dumps(diag,indent=2)+"\n")
    print(diag,flush=True)


if __name__=="__main__":
    groups()
    public()
