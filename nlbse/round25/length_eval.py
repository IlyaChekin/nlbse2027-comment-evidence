"""Calibrate length-aware corrections inside the exact negative cohort."""
import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round24"))
from veto import f,REFERENCE,exact_state,sparse_calibration,load_sparse,fit_heads,proposals,calibrate
from fuse_sparse import apply as sparse_apply
from conditioned import match
from internal_sparse import recipe
from project_eval import PARTS,raw_values,checksum,json_at,write_json

R22=ROOT/"round22"
R24=ROOT/"round24"
SLICES=("all","short","long")
RATIOS=(.5,1.)
VIEWS=("pooled","interaction","split","directional")


@dataclass
class Context:
    texts: list
    frame: dict
    state: dict
    active: np.ndarray
    inputs: np.ndarray
    current: np.ndarray
    bits: np.ndarray
    cutoff: float
    lengths: np.ndarray


def log_odds(p):
    return logit(p.clip(1e-5,1-1e-5))


def length_info(text):
    return np.fromiter((len(s) for s in text),dtype=np.int32)


def base_context(frame,state,sparse,heads,ref_cut,cutoff,texts):
    prior=proposals(frame,state,sparse,heads,ref_cut)[0]
    active,p=sparse_apply(state,prior,ref_cut,"blend","no",.5)
    full=state["p"].copy()
    full[active]=p
    y=state["p"]>ref_cut
    y[active]=p>cutoff
    x=np.column_stack((f.features(frame[REFERENCE],"negative"),log_odds(sparse[:,0]),log_odds(full)))
    return Context(texts,frame,state,active,x,full,y,cutoff,length_info(texts))


def source_contexts(rotation):
    texts,_,_,_,_=f.inputs()
    controls={}
    if rotation=="official":
        _,_,text,y,ids,_=f.dataset(full=True)
        with np.load(ROOT/"round8/frozen/holdout.npz") as split:
            if not np.array_equal(ids,split["rows"]):
                raise ValueError("Official internal calibration rows changed")
            fold=split["folds"]
        fit,cal,review=fold<3,fold==3,fold==4
        ref,policy,core=recipe()
        original=R24/"official/tuned"
        values,_,_=f.factor_data("official",text,f.small.tokenizer())
        frame={REFERENCE:values,"raw":raw_values(ids),"y":y[:,0],"ids":np.asarray(ids)}
        state=exact_state(frame,core,ref["gate"],policy=policy)
        frozen=json_at(original/"policy.json")
        if checksum(original/"models.joblib")!=frozen["models_sha256"]:
            raise ValueError("Frozen sparse calibration weights changed")
        variant=frozen["candidates"][frozen["selected"]]
        if (variant["scope"],variant["kind"],variant["mix"],variant["mode"])!=("no","blend",.5,"token"):
            raise ValueError("Expected the measured round 24 variant")
        models=joblib.load(original/"models.joblib")["token"]
        rec=json_at(R24/"official/fit.json")["models"]["token"]
        sparse=load_sparse(R24/"official","token","dev",frame,rec)
        from fuse_sparse import calibrate_sparse
        current=base_context(frame,state,calibrate_sparse(models["oof"],sparse),models["combiner"],
                             ref["threshold"],variant["threshold"],text)
        paired=original/f"{frozen['selected'].replace(':','_')}_held.npz"
        with np.load(paired) as z:
            if not np.array_equal(z["frozen"],current.bits[cal|review]):
                raise ValueError("Round 24 control did not reproduce its internal decisions")
        return current,None,fit,cal,review
    for phase in ("dev","test"):
        with np.load(R22/rotation/f"{phase}.npz") as z:
            controls[phase]=dict(z)
    dev,test=controls["dev"],controls["test"]
    split=json_at(f.WEAK/rotation/"policy.json")
    fit=np.isin(dev["ids"],split["fit_rows"])
    cal=np.isin(dev["ids"],split["calibration_rows"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,ref_cut,_,_=calibrate(controls,REFERENCE,fit,cal,budget)
    state=[exact_state(fr,heads,gate) for fr in (dev,test)]
    folder=R24/rotation
    rec=json_at(folder/"fit.json")["models"]["token"]
    from fuse_sparse import calibrate_sparse
    oof=sparse_calibration(folder,"token",rec)
    sparse=[calibrate_sparse(oof,load_sparse(folder,"token",k,fr,rec)) for k,fr in controls.items()]
    merged=fit_heads(dev,state[0],sparse[0],fit,ref_cut)
    config=json_at(folder/"conditioned/results.json")["options"]["token:no:blend:0.5"]
    cutoff=config["threshold"]
    d,t=[base_context(fr,s,spec,merged,ref_cut,cutoff,[texts[i] for i in fr["ids"]])
         for fr,s,spec in zip((dev,test),state,sparse,strict=True)]
    with np.load(folder/"conditioned/token_no_blend_0.5_pred.npz") as z:
        if not np.array_equal(t.bits,z["frozen"]) or not np.array_equal(t.active,z["selected"]):
            raise ValueError("Round 24 control did not reproduce its frozen project decisions")
    return d,t,fit,cal,None


def inputs(z,lengths,size,view):
    if view=="pooled":
        return z
    l=np.clip(np.log1p(lengths)-np.log1p(size),-2.,2.)
    return np.column_stack((z,l,np.abs(l),l[:,None]*z))


def train(mode,dev,fit,size):
    data=inputs(dev.inputs,dev.lengths,size,mode)
    if mode=="split":
        side=dev.lengths>size
    elif mode=="directional":
        side=dev.bits
    else:
        side=np.zeros(len(dev.lengths),dtype=bool)
    train=fit&dev.active
    models={}
    for key in sorted(set(side[train])):
        use=train&(side==key)
        models[bool(key)]=LogisticRegression(C=.25,solver="liblinear",max_iter=250,random_state=42).fit(
                    data[use],dev.frame["y"][use])
    return models


def score(mode,frames,models,size):
    data=inputs(frames.inputs,frames.lengths,size,mode)
    if mode=="split":
        side=frames.lengths>size
    elif mode=="directional":
        side=frames.bits
    else:
        side=np.zeros(len(frames.lengths),dtype=bool)
    p=frames.current.copy()
    for key,model in models.items():
        mask=frames.active&(side==key)
        if mask.any():
            p[mask]=model.predict_proba(data[mask])[:,1]
    return p


def band(context,name,size):
    use=context.active.copy()
    if name=="short":
        use &= context.lengths<=size
    elif name=="long":
        use &= context.lengths>size
    return use


def mix(ctx,prob,selected,ratio):
    score=ctx.current[selected]
    return ((1.-ratio)*score+ratio*prob[selected]).astype(np.float32)


def error_bins(frame,mask):
    return {name:dict(rows=int((slice_&mask&frame.active).sum()),
                 **f.counts(frame.frame["y"][slice_&mask&frame.active],frame.bits[slice_&mask&frame.active]))
            for name,slice_ in (("1-128",frame.lengths<=128),("129-256",(frame.lengths>128)&(frame.lengths<=256)),
                 ("257-512",(frame.lengths>256)&(frame.lengths<=512)),("513+",frame.lengths>512))}


def run(rotation):
    dest=HERE/"length"/rotation
    dest.mkdir(parents=True,exist_ok=True)
    dev,test,fit,cal,review=source_contexts(rotation)
    size=int(np.median(dev.lengths[fit&dev.active]))
    evalframe=dev if test is None else test
    keep=review if test is None else np.ones(len(evalframe.bits),dtype=bool)
    census=dict(train=error_bins(dev,fit),calibration=error_bins(dev,cal),eval=error_bins(evalframe,keep))
    modes={}
    records={}
    for name in VIEWS:
        models=train(name,dev,fit,size)
        modes[name]=models
        a=score(name,dev,models,size)
        b=score(name,evalframe,models,size)
        for scope,ratio in product(SLICES,RATIOS):
            selection=band(dev,scope,size)
            z=mix(dev,a,selection,ratio)
            here=cal[selection]
            label=dev.frame["y"][selection][here]
            budget=int((dev.bits[selection][here]&(label==0)).sum())
            current,cutoff=f.strict_budget(z[here],label,budget)
            cal_metrics=f.comparison(label,dev.bits[selection][here],current)
            active=band(evalframe,scope,size)
            p=mix(evalframe,b,active,ratio)
            pred=evalframe.bits.copy()
            pred[active]=p>cutoff
            fixed=f.comparison(evalframe.frame["y"][keep],evalframe.bits[keep],pred[keep])
            key=f"{name}_{scope}_{ratio:g}"
            rec=dict(calibration=cal_metrics,frozen=fixed,threshold=cutoff,eval_calls=int((active&keep).sum()))
            if test is not None:
                matched,each=match(test.frame["y"],test.frame["project"],test.bits,active,p)
                rec["same_fp"]=f.comparison(test.frame["y"],test.bits,matched)
                rec["per_project"]=each
            records[key]=rec
            np.savez_compressed(dest/f"{key}.npz",ids=evalframe.frame["ids"][keep],baseline=evalframe.bits[keep],
                       frozen=pred[keep],lengths=evalframe.lengths[keep],active=active[keep],score=p[keep[active]])
            print(rotation,key,cal_metrics["delta_tp"],rec.get("same_fp",fixed)["delta_tp"],fixed["delta_fp"],flush=True)
    joblib.dump(modes,dest/"models.joblib",compress=3)
    write_json(dest/"results.json",dict(length_split=size,length_coverage=census,results=records,
                                      weights_sha256=checksum(dest/"models.joblib"),
                                      code_sha256=checksum(Path(__file__))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    run(p.parse_args().rotation)
