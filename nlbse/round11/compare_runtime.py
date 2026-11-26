"""Alternate the leading variants on identical public rows and thread settings."""
import json
from pathlib import Path
from statistics import median


import numpy as np

import cascade
from students import ROOT, dataset, metric, write_json
from neighbor_eval import table
import compact_exit
import deep_exit
from score_runtime import competition_formula

HERE=Path(__file__).resolve().parent


def prepare(mode):
    if mode=="round10_compact":
        model,tok,rule,comments=compact_exit.pipeline(lexical=True)
        def run(texts):
            p,cost=compact_exit.evaluate(model,tok,texts,rule,comments,deferred_offsets=True)
            return p>.5,cost
        return run, ROOT/"round10/compact_exit/lexical_deferred/predictions.csv"
    if mode=="deep_exit_selected":
        model,tok,config=deep_exit.inference_model()
        def run(texts):
            p,_,cost=deep_exit.evaluate(model,tok,texts,config["selected"])
            return p>.5,cost
        return run, HERE/"deep_exit/selected/predictions.csv"
    prefix,key=mode.rsplit("256",1) if "256" in mode else mode.rsplit("128",1)
    width=256 if "256" in mode else 128
    continuation=key.removeprefix("_") or None
    pipeline=cascade.Pipeline(prefix,width,continuation)
    path=cascade.run_directory(width,continuation)/prefix/"predictions.csv"
    return pipeline.run,path


def main():
    summary=json.loads((HERE/"summary.json").read_text())
    modes=tuple(dict.fromkeys(["round10_compact",summary["score_winner_single_encoder"],summary["score_winner"]]))
    _,_,texts,y,ids,_=dataset(public=True)
    trials={k:[] for k in modes}
    runners={k:prepare(k) for k in modes}
    baseline={}
    for name,(run,path) in runners.items():
        saved=table(path)
        if saved[:,0].tolist()!=ids:
            raise ValueError(f"Prediction order differs for {name}")
        baseline[name]=saved[:,2:].astype(bool)
        run(texts[:2048])
    for repeat in range(3):
        for name in modes if repeat%2==0 else reversed(modes):
            bits,cost=runners[name][0](texts)
            mismatches=int(np.count_nonzero(bits!=baseline[name]))
            if mismatches:
                raise ValueError(f"Incompatible decisions for {name}: {mismatches}")
            trials[name].append(cost)
            print(name,cost,flush=True)
    results={}
    for name,records in trials.items():
        m=metric(y,baseline[name])["mean"]["f1"]
        cost={k:median(r[k] for r in records) for k in ("wall_seconds","forward_seconds","analytic_gflops")}
        results[name]=dict(mean_f1=m, cost=cost, trials=records, decisions_verified=2*len(ids),
                           local_formula=competition_formula(m,cost["forward_seconds"],cost["analytic_gflops"]),
                           wall_charged_formula=competition_formula(m,cost["wall_seconds"],cost["analytic_gflops"]))
    write_json(HERE/"matched_runtime.json",results)


if __name__=="__main__":
    main()
