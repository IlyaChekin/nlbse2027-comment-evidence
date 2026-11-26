"""Train sparse experts specializing in short or long rule-negative functions."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

from length_eval import HERE,ROOT,R24,PARTS,f,checksum,json_at,write_json,source_contexts,match
from train import cohorts
from sparse_model import WORDS
from veto import scalar

OUT=HERE/"specialists"
CHOICES=(.25,0.)


def fit_pair(x,y,short,idx,other):
    models=[]
    for branch in (True,False):
        rate=np.where(short[idx]==branch,1.,other)
        keep=rate>0
        model=LogisticRegression(C=2.,solver="liblinear",max_iter=150,random_state=42).fit(
                  x[idx[keep]],y[idx[keep]],sample_weight=rate[keep])
        models.append(model)
    return models


def predict_pair(models,x,selected,short):
    p=np.empty(len(selected),dtype=np.float32)
    for k,model in zip((True,False),models,strict=True):
        span=short[selected]==k
        if span.any():
            p[span]=model.decision_function(x[selected[span]])
    return p


def data_for(rotation):
    text,y,_,_,_=f.inputs()
    cache=R24/"features"
    source=cache/"matrix.npz"
    if checksum(source)!=json_at(cache/"manifest.json")["matrix_sha256"]:
        raise ValueError("Frozen lexical features changed")
    matrix=load_npz(source)[:,:WORDS.n_features].tocsr()
    part=R24/rotation
    archive=part/"token_oof.npz"
    if checksum(archive)!=json_at(part/"fit.json")["models"]["token"]["predictions_sha256"]:
        raise ValueError("Round 24 file-disjoint assignments changed")
    with np.load(archive) as z:
        idx=z["ids"]
        fold=z["fold"]
        dup=z["dup"]
        weak=z["weak"]
        if not np.array_equal(z["y"],y[idx,0]):
            raise ValueError("Training rows or labels changed")
    lengths=np.fromiter((len(s) for s in text),dtype=np.int32)
    return matrix,y[:,0],idx,fold,dup,weak,lengths,checksum(archive)


def evaluation(rotation):
    x,y,idx,fold,dup,weak,lengths,digest=data_for(rotation)
    d,b,fit,cal,review=source_contexts(rotation)
    if b is None:
        b=d
    target=idx[weak==0]
    labels=y[target]
    dev_size=int(np.median(d.lengths[fit&d.active]))
    short=lengths<=dev_size
    da=d.frame["ids"][d.active]
    ba=b.frame["ids"][b.active]
    dst=OUT/rotation
    dst.mkdir(parents=True,exist_ok=True)
    weights={}
    evidence={}
    for other in CHOICES:
        t=time.perf_counter()
        result=np.empty(len(target),dtype=np.float32)
        for n in sorted(set(fold)):
            held=fold[weak==0]==n
            allowed=target[(dup[weak==0]&(1<<n))==0]
            if np.any(np.isin(target[held],allowed)):
                raise ValueError("Out-of-file specialized fit overlaps")
            models=fit_pair(x,y,short,allowed,other)
            result[held]=predict_pair(models,x,target[held],short)
            print(rotation,other,"fold",n,round(time.perf_counter()-t,2),flush=True)
        models=fit_pair(x,y,short,target,other)
        calibration=[]
        for q in (True,False):
            use=short[target]==q
            calibration.append(scalar(result[use,None],labels[use],1.))
        sparse={}
        for phase,i in (("dev",da),("test",ba)):
            raw=predict_pair(models,x,i,short)
            pred=np.empty(len(raw))
            for q,classifier in zip((True,False),calibration,strict=True):
                keep=short[i]==q
                pred[keep]=classifier.predict_proba(raw[keep,None])[:,1]
            sparse[phase]=logit(pred.clip(1e-5,1-1e-5))
        train=fit[d.active]
        xx=np.column_stack((d.inputs[d.active],sparse["dev"]))
        yy=np.column_stack((b.inputs[b.active],sparse["test"]))
        h=LogisticRegression(C=.30,solver="liblinear",max_iter=250,random_state=42).fit(
                              xx[train],y[da][train])
        p=h.predict_proba(xx)[:,1]
        q=h.predict_proba(yy)[:,1]
        weights[str(other)]=dict(classifiers=models,calibration=calibration,head=h)
        np.savez_compressed(dst/f"oof_{other:g}.npz",ids=target,logits=result,y=labels,short=short[target])
        costs={}
        for scope,ratio,distance in product(("all","short","long"),(.4,.8),(None,.075)):
            am=short[da]==(scope=="short") if scope!="all" else np.ones(len(da),dtype=bool)
            bm=short[ba]==(scope=="short") if scope!="all" else np.ones(len(ba),dtype=bool)
            if distance is not None:
                am &= abs(d.current[d.active]-d.cutoff)<distance
                bm &= abs(b.current[b.active]-b.cutoff)<distance
            selected=cal[d.active]&am
            base=d.bits[d.active][selected]
            truth=y[da][selected]
            budget=int((base&(truth==0)).sum())
            v=(1-ratio)*d.current[d.active]+ratio*p
            zz=(1-ratio)*b.current[b.active]+ratio*q
            bits,cut=f.strict_budget(v[selected],truth,budget)
            train_stat=f.comparison(truth,base,bits)
            total=d.bits[cal].copy()
            cal_route=d.active.copy()
            cal_route[d.active]=am
            total[cal_route[cal]]=v[selected]>cut
            all_metrics=f.comparison(d.frame["y"][cal],d.bits[cal],total)
            if review is None:
                keep=np.ones(len(b.bits),dtype=bool)
            else:
                keep=review
            eligible=b.active.copy()
            eligible[b.active]=bm
            pred=b.bits.copy()
            pred[eligible]=zz[bm]>cut
            fixed=f.comparison(b.frame["y"][keep],b.bits[keep],pred[keep])
            key=f"{other:g}_{scope}_{ratio:g}_{'all' if distance is None else 'margin'}"
            record=dict(calibration=train_stat,cal_full=all_metrics,frozen=fixed,threshold=cut,branch=scope,ratio=ratio,
                        other_weight=other,distance=distance,rows=int((eligible&keep).sum()))
            if review is None:
                votes,where=match(b.frame["y"],b.frame["project"],b.bits,eligible,zz[bm])
                record["matched"]=f.comparison(b.frame["y"],b.bits,votes)
                record["project"]=where
            else:
                profile=(scope,distance)
                if profile not in costs:
                    chosen=da[selected]
                    times=[]
                    nnz=0
                    if len(chosen):
                        for _ in range(4):
                            start=time.perf_counter()
                            margin=predict_pair(models,x,chosen,short)
                            # Keep the old bag-of-tokens matrix and add one specialist dot and calibrator.
                            for branch_flag,estimator in zip((True,False),calibration,strict=True):
                                use=short[chosen]==branch_flag
                                if use.any():
                                    margin[use]=estimator.predict_proba(margin[use,None])[:,1]
                            h.predict_proba(np.column_stack((d.inputs[d.active][selected],
                                                       logit(margin.clip(1e-5,1-1e-5)))))
                            times.append(time.perf_counter()-start)
                        nnz=x[chosen].nnz
                    else:
                        times=[0.]
                    costs[profile]=dict(seconds=median(times),trials=times,nnz=nnz,rows=len(chosen))
                cost=costs[profile]
                impact=.3*(all_metrics["f1_after"]-all_metrics["f1_before"])
                record["proxy"]=float(impact-.2*f.PUBLIC_N/cal.sum()*(cost["seconds"]/1000+2*cost["nnz"]/3e15))
                record["cost"]=cost
            evidence[key]=record
            np.savez_compressed(dst/f"{key}.npz",ids=b.frame["ids"][keep],baseline=b.bits[keep],
                       pred=pred[keep],eligible=eligible[keep],score=zz[bm][keep[eligible]])
            print(rotation,key,train_stat["delta_tp"],record.get("matched",fixed)["delta_tp"],fixed["delta_fp"],flush=True)
    packed=dst/"specialists.joblib"
    joblib.dump(weights,packed,compress=3)
    write_json(dst/"results.json",dict(results=evidence,length_cutoff=dev_size,folds_source_sha256=digest,
               model_sha256=checksum(packed),code_sha256=checksum(Path(__file__))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    evaluation(p.parse_args().rotation)
