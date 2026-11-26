"""Quantify score ceilings for the requested short-function repair budget."""
import csv
import json
import math

import numpy as np

from rotate import HERE, PLAN
from evaluate import counts
from pipeline import PUBLIC,load,write_json,metric
from score_runtime import competition_formula


def main():
    texts,y,_=load(PUBLIC)
    observed=json.loads((HERE/"official/analysis.json").read_text())["baseline"]
    costs=observed["cost"]
    with (HERE/"official/baseline/predictions.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    if [int(r["row"]) for r in rows]!=list(range(len(y))):
        raise ValueError("Official prediction order differs")
    p=np.asarray([float(r["p_vulnerability"]) for r in rows])
    v=np.asarray([int(r["pred_vulnerability"]) for r in rows],dtype=bool)
    mat=np.asarray([int(r["pred_MAT"]) for r in rows],dtype=bool)
    short=np.fromiter(map(len,texts),dtype=int)<=json.loads(PLAN.read_text())["short_chars_max"]
    fn=(y[:,0]==1)&~v
    def score(bits,mat_bits=mat):
        m=metric(y,np.column_stack((bits,mat_bits)))
        return dict(vulnerability_f1=m["vulnerability"]["f1"],mat_f1=m["MAT"]["f1"],mean_f1=m["mean"]["f1"],
                    score=competition_formula(m["mean"]["f1"],costs["forward_seconds"],costs["analytic_gflops"]))
    b=counts(y[:,0],v)
    mat_counts=counts(y[:,1],mat)
    required=(.97-competition_formula(0,costs["forward_seconds"],costs["analytic_gflops"]))/0.6
    mat_f1=observed["metrics"]["MAT"]["f1"]
    vulnerability_target=2*required-mat_f1
    mat_target=2*required-observed["metrics"]["vulnerability"]["f1"]
    mat_required_tp=math.ceil((mat_counts["tp"]+mat_counts["fn"]+mat_counts["fp"])*mat_target/(2-mat_target))
    result=dict(observed=b,mat_f1=mat_f1,required_mean_f1=required,required_vulnerability_f1_if_fixed_mat=vulnerability_target,
           confident_short_false_negatives=int((fn&short&(p<.05)).sum()),all_short_false_negatives=int((fn&short).sum()),
           all_confident_short_repaired=score(v|(fn&short&(p<.05))),all_short_repaired=score(v|(fn&short)),
           all_vulnerability_fn_repaired_fixed_fp=score(v|fn),
           max_fp_at_perfect_vulnerability_recall=math.floor(2*int(y[:,0].sum())*(1/vulnerability_target-1)),
           mat_observed=mat_counts,required_mat_f1_if_fixed_vulnerability=mat_target,
           mat_fn_to_recover_at_fixed_fp=mat_required_tp-mat_counts["tp"],
           all_mat_fn_repaired_fixed_fp=score(v,mat|(y[:,1]==1)))
    write_json(HERE/"limits.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
