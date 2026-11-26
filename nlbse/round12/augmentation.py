"""Compare identifier-role augmentation against an otherwise matched compact encoder."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

from project_split import HERE, ROOT
from backbones import frozen, score
from project_heads import OUT, calibration, corpus, mat_scores, primary
from normalize_code import parallel as canonicalize
from pipeline import write_json, metric
from students import group_codes
from round2_report import bootstrap_delta

DEST=HERE/"augmentation"


def variants(functions):
    start=time.perf_counter()
    normalized=canonicalize(functions)
    prep=time.perf_counter()-start
    model,tok=frozen("canonical")
    raw,a=score("canonical",model,tok,functions)
    fancy,b=score("canonical",model,tok,normalized)
    b["wall_seconds"]+=prep
    b["normalization_seconds"]=prep
    return {"aug_raw":raw[:,0],"aug_norm":fancy[:,0],"aug_mean":.5*(raw[:,0]+fancy[:,0])},dict(aug_raw=a,aug_norm=b,
                     aug_mean={k:a[k]+b[k] for k in a})


def main(stage):
    texts,y,projects,paths,split=corpus()
    ids=split[stage]
    p,costs=variants([texts[i] for i in ids])
    gold=y[ids]
    groups=projects[ids]
    DEST.mkdir(parents=True,exist_ok=True)
    if stage=="dev":
        recipe={k:calibration(gold[:,0],q,groups) for k,q in p.items()}
        choice=max(recipe,key=lambda k:recipe[k]["objective"])
        write_json(DEST/"policy.json",dict(recipes=recipe,frozen=choice,development_costs=costs))
        print(recipe,choice,flush=True)
        return
    cfg=json.loads((DEST/"policy.json").read_text())
    main_policy=json.loads((OUT/"policy.json").read_text())
    weights=joblib.load(OUT/"heads.joblib")
    mat=mat_scores([texts[i] for i in ids],main_policy["mat"],weights)
    mat_bits=mat>main_policy["mat"]["cutoff"]
    base={}
    for name,posterior in primary("test").items():
        settings=main_policy["choices"][name]["candidates"]["raw"]
        base[name]=posterior[:,0]>settings["cutoff"]
    common=(base["compact"]!=gold[:,0])&(base["unix"]!=gold[:,0])
    old=np.column_stack((base["compact"],mat_bits))
    cluster=group_codes(zip(groups,[paths[i] for i in ids],strict=True))
    results={}
    for name,q in p.items():
        v=q>cfg["recipes"][name]["cutoff"]
        bits=np.column_stack((v,mat_bits))
        results[name]=dict(scores=metric(gold,bits),by_project={g:metric(gold[groups==g],bits[groups==g]) for g in np.unique(groups)},
                           costs=costs[name],common_errors_fixed=int((common&(v==gold[:,0])).sum()),
                           interval_vs_compact=bootstrap_delta(gold,old,bits,cluster))
        with (DEST/f"{name}.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(("row","project","p_vulnerability","p_MAT","pred_vulnerability","pred_MAT"))
            writer.writerows(zip(ids,groups,q,mat,v.astype(int),mat_bits.astype(int),strict=True))
    write_json(DEST/"analysis.json",dict(results=results,frozen=cfg["frozen"]))
    print({k:v["scores"]["mean"]["f1"] for k,v in results.items()},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("dev","test"))
    main(ap.parse_args().stage)
