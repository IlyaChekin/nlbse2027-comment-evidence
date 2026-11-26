"""Optimize fused attention by sizing batches to their padded sequence lengths."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from cache_hybrid import TOKEN_CONFIG
from fused_attention import (HERE, PORTABLE, DeferredOffsets, dataset, load_checkpoint,
                             predict, save)
from frontier import metric
from pipeline import TRAIN, load, select_indices, write_json
from score_runtime import competition_formula
from windows import chunks

SETTINGS = (0, 3072, 4096, 5120, 6144, 8192)
QUADRATIC_LIMIT = 32*512*512
MODE = "query_saliency_50"


def length_buckets(sequences, rows, budget):
    ordered=sorted(rows,key=lambda i:len(sequences[i]))
    group=[]
    for i in ordered:
        n=len(group)+1
        width=len(sequences[i])
        if group and (n>160 or n*width>budget or n*width*width>QUADRATIC_LIMIT):
            yield group
            group=[]
        group.append(i)
    if group:
        yield group


class Engine:
    def __init__(self):
        self.model,tok=load_checkpoint(PORTABLE,"cuda")
        self.tok=DeferredOffsets(tok)
        self.cache={}
        self.head=json.loads(TOKEN_CONFIG.read_text())["configurations"][MODE]

    def run(self,texts,budget):
        def batcher(seq,rows):
            return length_buckets(seq,rows,budget) if budget else chunks(seq,rows)

        torch.cuda.reset_peak_memory_stats()
        p,cost=predict(self.model,self.tok,texts,self.head,MODE,True,self.cache,batcher=batcher)
        cost["peak_cuda_mib"]=torch.cuda.max_memory_allocated()/2**20
        return p,cost


def benchmark(output):
    runner=Engine()
    t,y,_=load(TRAIN)
    fit,_=select_indices(pilot=False)
    ids=np.sort(np.random.default_rng(42).choice(fit,8192,replace=False))
    pilot=[t[i] for i in ids]
    del t
    labels=y[ids]
    trials={k:[] for k in SETTINGS}
    pred={}
    runner.run(pilot[:2048],0)
    for i in range(2):
        for budget in SETTINGS if i==0 else SETTINGS[::-1]:
            p,cost=runner.run(pilot,budget)
            if budget in pred and not np.array_equal(p>.5,pred[budget]>.5):
                raise ValueError(f"Unstable results for {budget}")
            trials[budget].append(cost)
            pred[budget]=p
            print("pilot",i+1,budget,cost,flush=True)
    yhat=pred[0]>.5
    baseline=metric(labels,yhat)["mean"]["f1"]
    summary={}
    for budget,p in pred.items():
        values=trials[budget]
        summary[budget]=dict(
            f1=metric(labels,p>.5)["mean"]["f1"],
            changed=int(np.count_nonzero(yhat!=(p>.5))),
            wall_seconds=median(x["wall_seconds"] for x in values), trials=values)
    eligible=[k for k,row in summary.items() if row["f1"]>=baseline-.0002]
    picked=min(eligible,key=lambda k:(summary[k]["wall_seconds"],summary[k]["changed"]))
    _,_,x,y,rows,cal=dataset(public=True)
    specs={"reference":0,"selected":picked}
    time={k:[] for k in specs}
    probabilities={}
    for i in range(3):
        for key in specs if i%2==0 else tuple(specs)[::-1]:
            p,cost=runner.run(x,specs[key])
            if key in probabilities and not np.array_equal(p>.5,probabilities[key]>.5):
                raise ValueError(f"Non-reproducible full predictions for {key}")
            probabilities[key]=p
            time[key].append(cost)
            print("public",i+1,key,cost,flush=True)
    results={}
    for key,p in probabilities.items():
        cost={field:median(x[field] for x in time[key]) for field in
              ("wall_seconds","forward_seconds","analytic_gflops","peak_cuda_mib")}
        save(output/key,y,rows,cal,p,cost,(.5,.5))
        scores=json.loads((output/key/"metrics.json").read_text())
        scores["local_formula"]=competition_formula(scores["scores"]["mean"]["f1"],
                                                     cost["forward_seconds"],cost["analytic_gflops"])
        results[key]=scores
    write_json(output/"analysis.json",dict(
        mode=MODE,device=torch.cuda.get_device_name(),selected_budget=picked,
        pilot_rows=ids.tolist(),pilot=summary,results=results,trials=time,
        binary_changes=int(np.count_nonzero((probabilities["reference"]>.5)!=(probabilities["selected"]>.5)))))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=HERE/"length_batches")
    benchmark(ap.parse_args().output)
