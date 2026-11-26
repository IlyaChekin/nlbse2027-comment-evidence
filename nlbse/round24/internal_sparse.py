"""Choose a sparse correction from internal files using measured extra inference cost."""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np

from veto import (HERE,ROOT,f,REFERENCE,checksum,json_at,write_json,exact_state,fit_heads,proposals,
                  restrict,revise,load_sparse,calibrate_sparse,sparse_calibration)
from sparse_model import aligned
from conditioned import apply as cohort_apply
from project_eval import PARTS,raw_values

ORIGIN=ROOT/"round20/factor_numeric/official"
FOLDER=HERE/"official"
OUT=FOLDER/"tuned"


def supported():
    tables=[
        [json_at(HERE/r/file)["options"] for r in PARTS]
        for file in ("comparison/results.json","conditioned/results.json")
    ]
    result={}
    for left,right in tables:
        for k in sorted(set(left)&set(right)):
            gain=[x[k]["project_tp"] for x in (left,right)]
            if min(gain)>=0 and max(gain)>0:
                result[k]=gain
    return result


def recipe():
    cfg=json_at(ORIGIN/"policy.json")
    current=cfg["modes"][cfg["selected"]]
    return current,f.Policy(**current["policy"]),joblib.load(ORIGIN/"heads.joblib")


def tune():
    OUT.mkdir(parents=True,exist_ok=True)
    options=supported()
    if not options:
        write_json(OUT/"policy.json",dict(selected=REFERENCE,reason="No variant improved the paired project comparison"))
        print("No project-supported variant",flush=True)
        return
    _,_,texts,label,ids,_=f.dataset(full=True)
    ids=np.asarray(ids)
    with np.load(ROOT/"round8/frozen/holdout.npz") as frozen:
        if not np.array_equal(ids,frozen["rows"]):
            raise ValueError("The internal calibration rows changed")
        fold=frozen["folds"]
    fit,cal,review=fold<3,fold==3,fold==4
    ref,policy,core=recipe()
    v,_,_=f.factor_data("official",texts,f.small.tokenizer())
    frame={REFERENCE:v,"raw":raw_values(ids),"y":label[:,0],"ids":ids}
    state=exact_state(frame,core,ref["gate"],policy=policy)
    baseline=state["p"]>ref["threshold"]
    budget=int((baseline&cal&(frame["y"]==0)).sum())
    print("options",options,"baseline_fp",budget,flush=True)
    sources=json_at(FOLDER/"fit.json")["models"]
    local={}
    decisions={}
    timings={}
    models={}
    for name in options:
        cohort= ":" in name
        if cohort:
            model,scope,kind,weight=name.split(":")
        else:
            model,kind,scope,weight=name.replace("two_gate","twogate").split("_")
            kind="two_gate" if kind=="twogate" else kind
        mix=float(weight)
        if model not in local:
            rec=sources[model]
            path=FOLDER/f"{model}.joblib"
            if checksum(path)!=rec["weights_sha256"]:
                raise ValueError("The projected sparse expert changed after training")
            expert=joblib.load(path)
            raw=load_sparse(FOLDER,model,"dev",frame,rec)
            oof=sparse_calibration(FOLDER,model,rec)
            p=calibrate_sparse(oof,raw)
            fit_models=fit_heads(frame,state,p,fit,ref["threshold"])
            ideas=proposals(frame,state,p,fit_models,ref["threshold"])
            models[model]=dict(expert=expert,oof=oof,combiner=fit_models)
            local[model]=(ideas,raw)
        record,raw=local[model]
        if cohort:
            mask,q=cohort_apply(state,record[0],ref["threshold"],kind,scope,mix)
            cal_inside=cal[mask]
            y_in=frame["y"][mask][cal_inside]
            fp_inside=int((baseline[mask][cal_inside]&(y_in==0)).sum())
            _,cut=f.strict_budget(q[cal_inside],y_in,fp_inside)
            revised=state["p"].copy()
            revised[mask]=q
            frozen=baseline.copy()
            frozen[mask]=q>cut
            at=frozen[cal]
        else:
            revised,mask=revise(state,record,ref["threshold"],kind,scope,mix)
            at,cut=f.strict_budget(revised[cal],frame["y"][cal],budget)
            frozen=revised>cut
        if (model,scope) not in timings:
            use=np.flatnonzero(cal&mask)
            sample=[texts[i] for i in use]
            expert=models[model]["expert"]
            elapsed=[]
            if sample:
                aligned(expert,sample,model)
                for _ in range(4):
                    t=time.perf_counter()
                    check,nnz=aligned(expert,sample,model)
                    _=calibrate_sparse(models[model]["oof"],check)
                    elapsed.append(time.perf_counter()-t)
                if not np.allclose(check,raw[use],rtol=0,atol=1e-6):
                    raise ValueError("Online vectorization differs from frozen sparse features")
            else:
                elapsed=[0.]
                nnz=0
            timings[model,scope]=dict(seconds=median(elapsed),trials=elapsed,nnz=nnz,rows=len(use))
        cost=timings[model,scope]
        dev=f.comparison(frame["y"][cal],baseline[cal],at)
        hold=f.comparison(frame["y"][review],baseline[review],frozen[review])
        boost=.3*(dev["f1_after"]-dev["f1_before"])
        scale=f.PUBLIC_N/cal.sum()
        debit=.2*scale*(cost["seconds"]/1000+4*cost["nnz"]/3e15)
        decisions[name]=dict(cohort_only=cohort,mode=model,kind=kind,scope=scope,mix=mix,threshold=cut,calibration=dev,
                        review=hold,proxy=float(boost-debit),quality_gain=boost,
                        measured_cost=cost,project_tp=options[name])
        np.savez_compressed(OUT/f"{name.replace(chr(58),chr(95))}_held.npz",ids=ids[cal|review],fold=fold[cal|review],
            gold=frame["y"][cal|review],baseline=baseline[cal|review],proposal=revised[cal|review],frozen=frozen[cal|review],active=mask[cal|review])
        print(name,dev["delta_tp"],hold["delta_tp"],hold["delta_fp"],"proxy",boost-debit,cost["rows"],flush=True)
    best=max(decisions,key=lambda n:decisions[n]["proxy"])
    selected=best if decisions[best]["proxy"]>0 else REFERENCE
    dst=OUT/"models.joblib"
    joblib.dump(models,dst,compress=3)
    write_json(OUT/"policy.json",dict(selected=selected,best=best,candidates=decisions,cal_rows=int(cal.sum()),
               source_policy_sha256=checksum(ORIGIN/"policy.json"),heldout_sha256=checksum(ROOT/"round8/frozen/holdout.npz"),
               models_sha256=checksum(dst),script_sha256=checksum(Path(__file__))))
    print("selected",selected,flush=True)


if __name__=="__main__":
    tune()
