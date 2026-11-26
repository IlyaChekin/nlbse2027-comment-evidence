"""Combine the train-selected length limit with the frozen adaptive routes."""

import argparse
from functools import partial
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from hybrid import (HERE, dataset, save, predict_bits, bootstrap_delta, group_codes,
                    load, PUBLIC)
from length_batches import length_buckets
from packed_suffix import PackedPipeline
from pipeline import write_json
from score_runtime import competition_formula

NAMES = ("balanced_fixed", "fast_fixed", "balanced_length", "fast_length")


def benchmark(output, repeat):
    config=json.loads((HERE/"length_batches/analysis.json").read_text())
    budget=config["selected_budget"]
    if not budget:
        raise ValueError("No variable-length batch was selected on the timing pilot")
    original=PackedPipeline()
    helper=partial(length_buckets,budget=budget)
    resized=PackedPipeline(batcher=helper)
    asymmetric=json.loads((HERE/"frozen/asymmetric.json").read_text())["options"]["asym_score"]["rule"]
    modes={
        "balanced_fixed":(original,asymmetric),
        "fast_fixed":(original,original.gate["options"]["score"]["rule"]),
        "balanced_length":(resized,asymmetric),
        "fast_length":(resized,resized.gate["options"]["score"]["rule"]),
    }
    _,_,t,y,ids,cal=dataset(public=True)
    _,_,columns=load(PUBLIC,audit=True)
    groups=group_codes(zip(columns["Projectname"],columns["Filepath"],strict=True))
    earlier=HERE.parent/"round8/tokens_final/query_saliency_50/predictions.csv"
    base=predict_bits(earlier,ids,{"thresholds":(.5,.5)})
    timings={k:[] for k in modes}
    prediction={}
    for runner,rule in modes.values():
        runner.adaptive(t[:2048],rule)
    for i in range(repeat):
        for key in NAMES if i%2==0 else NAMES[::-1]:
            runner,rule=modes[key]
            torch.cuda.reset_peak_memory_stats()
            p,cost=runner.adaptive(t,rule)
            cost["peak_gpu_mib"]=torch.cuda.max_memory_allocated()/2**20
            if key in prediction and not np.array_equal(p>.5,prediction[key]>.5):
                raise ValueError(f"Modifying batch lengths changed decisions between repeats of {key}")
            prediction[key]=p
            timings[key].append(cost)
            print(i+1,key,cost,flush=True)
    results={}
    for key,p in prediction.items():
        runs=timings[key]
        cost={name:median(r[name] for r in runs)
              for name in ("wall_seconds","forward_seconds","analytic_gflops","peak_gpu_mib")}
        cost["actions"]=runs[0]["actions"]
        save(output/key,y,ids,cal,p,cost,(.5,.5))
        row=json.loads((output/key/"metrics.json").read_text())
        row["local_formula"]=competition_formula(row["scores"]["mean"]["f1"],cost["forward_seconds"],
                                                cost["analytic_gflops"])
        row["paired_interval_95_vs_release"]=bootstrap_delta(y,base,p>.5,groups)
        row["binary_changes_vs_fixed"]=int(np.count_nonzero((p>.5)!=(prediction[key.split("_")[0]+"_fixed"]>.5)))
        results[key]=row
    write_json(output/"analysis.json",dict(budget=budget,results=results,trials=timings,
                                           device=torch.cuda.get_device_name(),repeats=repeat))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=HERE/"combined")
    ap.add_argument("--repeat",type=int,default=2)
    args=ap.parse_args()
    benchmark(args.output,args.repeat)
