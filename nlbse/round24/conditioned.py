"""Calibrate a sparse correction inside one routing cohort at a time."""
import argparse
from itertools import product
from pathlib import Path
import sys

import joblib
import numpy as np

from veto import (HERE,ROOT,R22,f,REFERENCE,OP,PARTS,checksum,json_at,write_json,exact_state,
                  sparse_calibration,calibrate_sparse,load_sparse,fit_heads,proposals,calibrate)

from fuse_sparse import apply,eligible

SCOPES=("no","yes","unqueried")
VARIANTS=tuple(product(SCOPES,("blend","veto","promote"),(.5,1.)))


def match(y,groups,base,inside,q):
    bits=base.copy()
    for name in sorted(set(groups)):
        m=groups[inside]==name
        if not m.any():
            continue
        old=base[inside][m]
        yp=y[inside][m]
        n=int((old&(yp==0)).sum())
        current,_=f.exact_budget(q[m],yp,n)
        bits[np.flatnonzero(inside)[m]]=current
    if np.any(bits[~inside]!=base[~inside]):
        raise ValueError("A cohort correction changed the other selected examples")
    records={name:f.comparison(y[groups==name],base[groups==name],bits[groups==name])
             for name in sorted(set(groups))}
    if any(item["delta_fp"] for item in records.values()):
        raise ValueError("False positives changed on a held-out project")
    return bits,records


def evaluate(rotation):
    folder=HERE/rotation
    out=folder/"conditioned"
    out.mkdir(exist_ok=True)
    frames={}
    for phase in ("dev","test"):
        with np.load(R22/rotation/f"{phase}.npz") as a:
            frames[phase]=dict(a)
    dev,test=frames["dev"],frames["test"]
    known=json_at(f.WEAK/rotation/"policy.json")
    fit=np.isin(dev["ids"],known["fit_rows"])
    cal=np.isin(dev["ids"],known["calibration_rows"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,cut,_,_=calibrate(frames,REFERENCE,fit,cal,budget)
    old=exact_state(dev,heads,gate)
    ref=exact_state(test,heads,gate)
    a,b=old["p"]>cut,ref["p"]>cut
    variants={}
    for name,recipe in json_at(folder/"fit.json")["models"].items():
        oof=sparse_calibration(folder,name,recipe)
        dd,tt=[calibrate_sparse(oof,load_sparse(folder,name,k,frame,recipe)) for k,frame in frames.items()]
        model=fit_heads(dev,old,dd,fit,cut)
        d=proposals(dev,old,dd,model,cut)[0]
        t=proposals(test,ref,tt,model,cut)[0]
        for scope,kind,mix in VARIANTS:
            mask,score=apply(old,d,cut,kind,scope,mix)
            on=cal[mask]
            target=dev["y"][mask][on]
            n=int((a[mask][on]&(target==0)).sum())
            bits,th=f.strict_budget(score[on],target,n)
            development=f.comparison(target,a[mask][on],bits)
            selected,q=apply(ref,t,cut,kind,scope,mix)
            fixed=b.copy()
            fixed[selected]=q>th
            relative,pers=match(test["y"],test["project"],b,selected,q)
            net=sum(t["delta_tp"] for t in pers.values())
            label=f"{name}:{scope}:{kind}:{mix:g}"
            variants[label]=dict(calibration=development,matched=f.comparison(test["y"],b,relative),
                frozen=f.comparison(test["y"],b,fixed),per_project=pers,project_tp=net,
                threshold=th,cal_queries=int((cal&mask).sum()),test_queries=int(selected.sum()))
            np.savez_compressed(out/f"{name}_{scope}_{kind}_{mix:g}_pred.npz",ids=test["ids"],truth=test["y"],
                       baseline=b,frozen=fixed,matched=relative,selected=selected,score=q)
    picks={scope:max((k for k in variants if k.split(":")[1]==scope),
                  key=lambda k:(variants[k]["calibration"]["delta_tp"],-variants[k]["cal_queries"]))
           for scope in SCOPES}
    rows={scope:{k:variants[key][k] for k in ("project_tp","test_queries")} for scope,key in picks.items()}
    write_json(out/"results.json",dict(options=variants,selected=picks,source_policy_sha256=checksum(f.WEAK/rotation/"policy.json")))
    print(rotation,rows,flush=True)
    print("top",sorted([(v["project_tp"],k,v["calibration"]["delta_tp"],v["test_queries"])
                for k,v in variants.items()],reverse=True)[:12],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=PARTS)
    evaluate(p.parse_args().rotation)
