"""Select the batch size and tokenizer threading on a train-only timing pilot."""

import argparse
import json
import os
from pathlib import Path
from statistics import median

import numpy as np
import torch

from hybrid import Pipeline, HERE, dataset, save
from pipeline import TRAIN, load, select_indices, write_json
from score_runtime import competition_formula
import windows

SETTINGS = ((32,False),(32,True),(64,False),(64,True),(96,True),(128,True),(192,True))


def run(proc,texts,mode,spec):
    batch,parallel=spec
    windows.BATCH=batch
    os.environ["TOKENIZERS_PARALLELISM"]="true" if parallel else "false"
    torch.cuda.reset_peak_memory_stats()
    result,cost=proc.predict(texts,mode)
    cost["peak_gpu_mib"]=torch.cuda.max_memory_allocated()/2**20
    return result,cost


def benchmark(output,mode):
    from tune_hybrid import metric
    os.environ.setdefault("RAYON_NUM_THREADS","4")
    old_batch=windows.BATCH
    original_env=os.environ.get("TOKENIZERS_PARALLELISM")
    try:
        proc=Pipeline()
        X,y,_=load(TRAIN)
        fit,_=select_indices(pilot=False)
        sample=np.sort(np.random.default_rng(42).choice(fit,size=8192,replace=False))
        texts=[X[i] for i in sample]
        y=y[sample]
        del X
        run(proc,texts[:2048],mode,SETTINGS[0])
        trials={spec:[] for spec in SETTINGS}
        scores={}
        for cycle in range(2):
            for spec in SETTINGS if cycle==0 else SETTINGS[::-1]:
                try:
                    p,cost=run(proc,texts,mode,spec)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("too much memory",spec,flush=True)
                    continue
                if spec in scores and not np.array_equal(p>.5,scores[spec]>.5):
                    raise ValueError(f"Unstable decisions for batch setting {spec}")
                scores[spec]=p
                trials[spec].append(cost)
                print("pilot",cycle+1,spec,cost,flush=True)
        baseline=scores[SETTINGS[0]]
        pilot={}
        f1_reference=metric(y,baseline>.5)["mean"]["f1"]
        for spec,p in scores.items():
            report=trials[spec]
            if len(report)!=2:
                continue
            pilot[spec]=dict(f1=metric(y,p>.5)["mean"]["f1"],changes=int(np.count_nonzero((baseline>.5)!=(p>.5))),
                             wall_seconds=median(t["wall_seconds"] for t in report),
                             peak_gpu_mib=max(t["peak_gpu_mib"] for t in report),trials=report)
        eligible=[spec for spec in pilot if pilot[spec]["f1"]>=f1_reference-.0002]
        selected=min(eligible,key=lambda spec:(pilot[spec]["wall_seconds"],pilot[spec]["changes"]))
        _,_,public,y,ids,cal=dataset(public=True)
        commands={"reference":SETTINGS[0],"selected":selected}
        times={key:[] for key in commands}
        predictions={}
        for i in range(3):
            for variant in commands if i%2==0 else tuple(commands)[::-1]:
                p,cost=run(proc,public,mode,commands[variant])
                if variant in predictions and not np.array_equal(p>.5,predictions[variant]>.5):
                    raise ValueError(f"Repeated output changed with {variant}")
                predictions[variant]=p
                times[variant].append(cost)
                print("public",i+1,variant,cost,flush=True)
        results={}
        for key,p in predictions.items():
            row={col:median(r[col] for r in times[key]) for col in
                 ("wall_seconds","forward_seconds","analytic_gflops","peak_gpu_mib")}
            save(output/key,y,ids,cal,p,row,(.5,.5))
            metrics=json.loads((output/key/"metrics.json").read_text())
            metrics["local_formula"]=competition_formula(metrics["scores"]["mean"]["f1"],row["forward_seconds"],
                                                         row["analytic_gflops"])
            results[key]=metrics
        write_json(output/"analysis.json",dict(
            device=torch.cuda.get_device_name(),mode=mode,pilot_seed=42,pilot_rows=sample.tolist(),
            pilot=[dict(batch=k[0],parallel=k[1],**v) for k,v in pilot.items()],
            selected=dict(batch=selected[0],parallel=selected[1],rayon_threads=int(os.environ["RAYON_NUM_THREADS"])),
            results=results,trials=times,binary_changes=int(np.count_nonzero(
                (predictions["reference"]>.5)!=(predictions["selected"]>.5)))))
    finally:
        windows.BATCH=old_batch
        if original_env is None:
            os.environ.pop("TOKENIZERS_PARALLELISM",None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"]=original_env


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",default="fused50")
    ap.add_argument("--output",type=Path,default=HERE/"batch")
    args=ap.parse_args()
    benchmark(args.output,args.mode)
