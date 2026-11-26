"""Confirm front-runner modes with the same CPU/GPU overlap engine."""
import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from frontier import (BASE, HERE, PORTABLE, THRESHOLDS, dataset, load_checkpoint,
                      predict_bits, run, run_policy, save, write_json)
from score_runtime import competition_formula
from tokenizer_speed import DeferredOffsets

MODES = ("exact", "six", "adaptive", "wide")


def make_prediction(mode, model, tokenizer, texts, config):
    if mode=="exact":
        p,cost=run_policy(model,tokenizer,texts,"cuda",BASE,prefetch=True)
        return p,cost,THRESHOLDS
    p,cost=run(model,tokenizer,texts,config,mode,prefetch=True)
    return p,cost,(.5,.5)


def main(output, repeats):
    config=json.loads((HERE/"frozen/policy.json").read_text())
    model,tokenizer=load_checkpoint(PORTABLE,"cuda")
    tokenizer=DeferredOffsets(tokenizer)
    _,_,texts,y,ids,cal=dataset(public=True)
    def reference(mode,t):
        name="baseline" if mode=="exact" else mode
        return predict_bits(HERE/"public"/name/"predictions.csv",ids,{"thresholds":t})
    trials={mode:[] for mode in MODES}
    predictions={}
    for mode in MODES:
        make_prediction(mode,model,tokenizer,texts[:2048],config)
    for repeat in range(repeats):
        for mode in MODES if repeat%2==0 else MODES[::-1]:
            p,cost,thresholds=make_prediction(mode,model,tokenizer,texts,config)
            target=reference(mode,thresholds)
            if not np.array_equal(p>thresholds,target):
                raise ValueError(f"Overlapping CPU preparation changed {mode} decisions")
            trials[mode].append(cost)
            predictions[mode]=(p,thresholds)
            print(repeat+1, mode, cost["wall_seconds"],cost["forward_seconds"], flush=True)
    results={}
    for mode,(p,t) in predictions.items():
        cost={k:median(x[k] for x in trials[mode]) for k in ("wall_seconds","forward_seconds","analytic_gflops")}
        save(output/mode,y,ids,cal,p,cost,t)
        result=json.loads((output/mode/"metrics.json").read_text())
        result["local_formula"]=competition_formula(result["scores"]["mean"]["f1"],
                                                     cost["forward_seconds"],cost["analytic_gflops"])
        results[mode]=result
    write_json(output/"analysis.json",dict(apparatus=torch.cuda.get_device_name(),repetitions=repeats,
                                          results=results,trials=trials,ordered_bits_unchanged=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat",type=int,default=2)
    ap.add_argument("--output",type=Path,default=HERE/"combined")
    args=ap.parse_args()
    main(args.output,args.repeat)
