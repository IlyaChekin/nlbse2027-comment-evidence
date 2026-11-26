"""Profile rearranged token layouts with the fixed compact checkpoint."""

import argparse
import csv
from statistics import median

import numpy as np

from context import NAMES, OUT as STOCK, layout, tokenize
from experiment import HERE, ROOT, checksum, comparison, json_at, open_model, small, strict_budget
from evaluate import counts, exact_budget
from local_public import heldout, old_policy, run, write_public
from pipeline import PUBLIC, load, metric, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from students import group_codes


OUT=HERE/"context_public"
NAME=("reference","threshold",*NAMES)


def encoding(mode,idf):
    if mode in {"reference","threshold"}:
        return None

    def prepare(texts,tok):
        return [layout(vec,mode,idf) for vec in tokenize(texts,tok)]

    return prepare


def tune():
    OUT.mkdir(exist_ok=True)
    route,threshold=old_policy()
    text,y,ids,cal,rev=heldout(include_test=False)
    source=STOCK/"idf.npy"
    idf=np.load(source)
    pipe=open_model("official",OUT,"reference",small.tokenizer())
    base,old=run(pipe,text,route)
    base_bits=base[:,0]>threshold
    fp=counts(y[cal,0],base_bits[cal])["fp"]
    ref,_=strict_budget(base[cal,0],y[cal,0],fp)
    settings={}
    for mode in NAME:
        p,cost=(base,old) if mode in {"reference","threshold"} else run(pipe,text,route,encoding(mode,idf))
        if not np.array_equal(p[:,1],base[:,1]):
            raise ValueError("Context rearrangement changed the MAT rule")
        equal,cut=strict_budget(p[cal,0],y[cal,0],fp)
        cut=threshold if mode=="reference" else cut
        bits=p[cal]>(cut,.5)
        mean=metric(y[cal],bits)["mean"]["f1"]
        record=dict(cutoff=cut,development=comparison(y[cal,0],ref,equal),
                    review=comparison(y[rev,0],base_bits[rev],p[rev,0]>cut),cost=cost,
                    proxy=competition_formula(mean,0.,cost["analytic_gflops"]*30655/len(y)))
        settings[mode]=record
        print(mode,record,flush=True)
    selected=max(NAME,key=lambda n:settings[n]["proxy"])
    write_json(OUT/"policy.json",dict(selected=selected,modes=settings,route=route,dev_fp=fp,cal_rows=ids[cal].tolist(),
                                      review_rows=ids[rev].tolist(),idf_sha256=checksum(source)))


def public():
    cfg=json_at(OUT/"policy.json")
    source=STOCK/"idf.npy"
    if checksum(source)!=cfg["idf_sha256"]:
        raise ValueError("Tile-frequency data changed")
    idf=np.load(source)
    chosen=cfg["selected"]
    modes=list(dict.fromkeys(("reference",chosen)))
    text,y,meta=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    with (ROOT/"round15/official_distill/original_teacher_public.csv").open(newline="") as f:
        archive=list(csv.DictReader(f))
    if [int(r["row"]) for r in archive]!=ids:
        raise ValueError("Public reference does not cover the full split")
    original=np.asarray([[int(r[k]) for k in ("vulnerability","mat")] for r in archive],dtype=bool)
    files=group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    pipe=open_model("official",OUT,"reference",small.tokenizer())
    caches={}
    obs={m:[] for m in modes}
    for mode in modes:
        run(pipe,text[:2048],cfg["route"],encoding(mode,idf))
    for i in range(3):
        for mode in modes if i%2==0 else reversed(modes):
            probability,cost=run(pipe,text,cfg["route"],encoding(mode,idf))
            bits=probability>(cfg["modes"][mode]["cutoff"],.5)
            if mode=="reference" and not np.array_equal(bits,original):
                raise ValueError("Measured pipeline differs from the round 15 reference")
            if mode in caches and not np.array_equal(bits,caches[mode][1]):
                raise ValueError("Predictions changed between runtime repeats")
            if not np.array_equal(bits[:,1],original[:,1]):
                raise ValueError("MAT differs from reference after changing the input layout")
            caches[mode]=(probability,bits)
            obs[mode].append(cost)
            print(mode,cost,flush=True)
    results={}
    fp=counts(y[:,0],original[:,0])["fp"]
    reference,_=exact_budget(caches["reference"][0][:,0],y[:,0],fp)
    for name,times in obs.items():
        prob,bits=caches[name]
        equal,_=exact_budget(prob[:,0],y[:,0],fp)
        q=metric(y,bits)
        cost={k:median(x[k] for x in times) for k in times[0]}
        results[name]=dict(quality=q,cost=cost,score=competition_formula(q["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                           matched=comparison(y[:,0],reference,equal),frozen=comparison(y[:,0],original[:,0],bits[:,0]),
                           trials=times,paired_interval=bootstrap_delta(y,original,bits,files))
        write_public(OUT/f"{name}_public.csv",ids,y,prob,bits,equal,reference)
    write_json(OUT/"results.json",results)
    print({k:(r["score"],r["matched"]["delta_tp"]) for k,r in results.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("tune","public"))
    {"tune":tune,"public":public}[p.parse_args().stage]()
