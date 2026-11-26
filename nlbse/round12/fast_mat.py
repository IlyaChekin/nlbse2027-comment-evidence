"""Reuse the compact encoder's full-comment features for the MAT decision."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"round11"),str(ROOT/"round10")]
import cascade
import small
from students import PUBLIC, batch, batch_plan, lexical_features, load, metric, ordered_batches, predict_bits, save, write_json
from score_runtime import competition_formula

OUT=HERE/"fast_mat"


@torch.inference_mode()
def infer(pipe,texts,*,probabilities=False):
    model,tok=pipe.encoder,pipe.tok
    model.eval()
    start=time.perf_counter()
    fwd,gflops=0.,0.
    output=np.empty((len(texts),2),dtype=np.float32 if probabilities else np.int8)
    threshold=pipe.student["thresholds"][0]

    def prepare(offset):
        rows=texts[offset:offset+2048]
        features=lexical_features(rows)
        return small.encode(rows,tok),features

    with torch.autocast("cuda",dtype=torch.float16):
        for offset,(sequence,features) in ordered_batches(range(0,len(texts),2048),prepare,True):
            output[offset:offset+len(sequence),1]=features[:,:4].any(axis=1)
            extra=torch.as_tensor(features,device="cuda")
            for rows in batch_plan(list(range(len(sequence))),sequence,small.BATCH):
                x,mask=batch(rows,sequence,tok.pad_token_id)
                step=time.perf_counter()
                p=model(x,mask,extra[rows])[:,0].float().sigmoid().cpu().numpy()
                output[offset+np.asarray(rows),0]=p if probabilities else p>threshold
                fwd+=time.perf_counter()-step
                gflops+=len(rows)*small.operations(1,x.shape[1],model.width)
    return output,dict(wall_seconds=time.perf_counter()-start,forward_seconds=fwd,analytic_gflops=gflops)


def runner():
    return cascade.Pipeline("small_lexical",256,"gold_disagreement",shifted=True)


def public():
    texts,y,_=load(PUBLIC)
    ids=list(range(len(y)))
    base=predict_bits(ROOT/"round11/cold_shift/predictions.csv",ids,{"thresholds":(.5,.5)})
    pipe=runner()
    variants={"learned":pipe.run,"reuse_tags":lambda values:infer(pipe,values)}
    trials={k:[] for k in variants}
    results={}
    for run in variants.values():
        run(texts[:2048])
    for i in range(3):
        for name in variants if i%2==0 else reversed(variants):
            pred,cost=variants[name](texts)
            if not np.array_equal(pred,base):
                raise ValueError(f"{name} changed a public decision")
            results[name]=pred
            trials[name].append(cost)
            print(name,cost,flush=True)
    output={}
    for name,obs in trials.items():
        fields=("wall_seconds","forward_seconds","analytic_gflops")
        cost={k:median(r[k] for r in obs) for k in fields}
        m=metric(y,results[name])
        output[name]=dict(metrics=m,cost=cost,trials=obs,identical_binary_decisions=2*len(y),
                     local_formula=competition_formula(m["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                     formula_charging_whole_wall=competition_formula(m["mean"]["f1"],cost["wall_seconds"],cost["analytic_gflops"]))
        save(OUT/name,y,ids,np.zeros(len(y),dtype=bool),results[name],cost,(.5,.5))
    write_json(OUT/"analysis.json",output)


def predict(source,destination):
    with source.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function input column")
        functions=[r["Function"] for r in reader]
    if not functions or any(t is None for t in functions):
        raise ValueError("Every row must supply a function")
    p,_=infer(runner(),functions)
    with destination.open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("vulnerability","MAT"))
        w.writerows(p.tolist())


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("public","predict"))
    ap.add_argument("--input",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.mode=="public":
        public()
    else:
        if args.input is None or args.output is None:
            ap.error("predict requires --input and --output")
        predict(args.input,args.output)
