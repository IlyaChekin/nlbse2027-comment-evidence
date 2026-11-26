"""Replay a separately trained adviser only for explicitly rule-negative examples."""
import csv
from itertools import product
import sys

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from scope_eval import HERE,ROOT,R22,REFERENCE,OP,PARTS,baseline,blend,calibrate,f,checksum,json_at,write_json

SOURCE=ROOT/"round13/backbones/unix"
POLICIES=tuple(product(("closest","expected"),(.03,.08,.18),(.5,1.)))


def archived(y):
    p=np.full(len(y),np.nan)
    rows={}
    for name in ("probe","dev","test"):
        file=SOURCE/f"{name}.npz"
        receipt=json_at(file.with_suffix(".json"))
        sha=checksum(file)
        if sha!=receipt["sha256"]:
            raise ValueError("Adviser archive changed")
        with np.load(file) as data:
            ids=data["ids"]
            if not np.array_equal(data["y"][:,0],y[ids]) or np.isfinite(p[ids]).any():
                raise ValueError("Adviser rows or labels changed")
            p[ids]=data["probabilities"][:,0]
        rows[name]=sha
    return p,rows


def input_row(frame,base,side):
    return np.column_stack((logit(frame[REFERENCE].clip(1e-5,1-1e-5)),
                           logit(base["no"].clip(1e-5,1-1e-5)),side[frame["ids"]]))


def joint(frame,teacher):
    return np.column_stack((f.features(frame[REFERENCE],"negative"),logit(teacher.clip(1e-5,1-1e-5))))


def interested(model,frame,b,side,cost,cut,mode):
    if mode=="closest":
        priority=1/(.035+np.abs(b["p"]-cut))
    else:
        priority=model.predict(input_row(frame,b,side))
    return np.where(b["negative"],priority/np.sqrt(cost),-1e6)


def loss(p, y):
    return -np.log(np.where(y==1,p,1-p).clip(1e-5,1.))


def changed(b,teacher,mask,weight):
    p=b["p"].copy()
    p[mask]=(1-weight)*p[mask]+weight*teacher[mask]
    return p


def run(rotation,teacher,sha,side):
    dest=HERE/"adviser"/rotation
    dest.mkdir(parents=True,exist_ok=True)
    frames={}
    for key in ("dev","test"):
        with np.load(R22/rotation/f"{key}.npz") as z:
            frames[key]=dict(z)
    dev,test=frames["dev"],frames["test"]
    for x in (dev,test):
        x["adviser"]=teacher[x["ids"]]
        if not np.isfinite(x["adviser"]).all():
            raise ValueError("Adviser archive does not cover disjoint development/test rows")
    split=json_at(f.WEAK/rotation/"policy.json")
    fit=np.isin(dev["ids"],split["fit_rows"])
    cal=np.isin(dev["ids"],split["calibration_rows"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,cut,base_dev,_=calibrate(frames,REFERENCE,fit,cal,budget)
    old=baseline(dev,heads,gate)
    ref=baseline(test,heads,gate)
    fixed=ref["p"]>cut
    masks={k:f.valid(v["raw"])&~f.weak(v["raw"]) for k,v in frames.items()}
    pool=fit&masks["dev"]
    d,t=joint(dev,dev["adviser"]),joint(test,test["adviser"])
    calibrated=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(d[pool],dev["y"][pool])
    a,b=[calibrated.predict_proba(z)[:,1] for z in (d,t)]
    # Use a second file-disjoint split, so router targets are predictions on rows unseen by this calibration head.
    files=f.inputs()[3]
    file_ids=np.asarray([f"{pr}/{files[i]}" for pr,i in zip(dev["project"],dev["ids"],strict=True)])
    from zlib import crc32
    parity=np.fromiter((crc32(x.encode())%2 for x in file_ids),dtype=np.int8)
    oof=np.full(len(a),np.nan)
    for j in (0,1):
        training=pool&(parity!=j)
        fitted=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(d[training],dev["y"][training])
        oof[pool&(parity==j)]=fitted.predict_proba(d[pool&(parity==j)])[:,1]
    target=(loss(old["no"][pool],dev["y"][pool])-loss(oof[pool],dev["y"][pool])).clip(-2.5,2.5)
    router=HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=13,min_samples_leaf=80,
                                        l2_regularization=2.,random_state=42).fit(input_row(dev,old,side)[pool],target)
    # Extracted character counts are only a routing cost proxy, never measurements of the adviser.
    da=np.exp(side[dev["ids"],0]).clip(20,2048)
    ta=np.exp(side[test["ids"],0]).clip(20,2048)
    recipes={}
    for mode,fraction,weight in (*POLICIES,("all",1.,.5),("all",1.,1.)):
        if mode=="all":
            va,vb=old["negative"],ref["negative"]
            route=None
        else:
            pr=interested(router,dev,old,side,da,cut,mode)
            pt=interested(router,test,ref,side,ta,cut,mode)
            ranked=np.flatnonzero(cal&old["negative"])
            ranked=ranked[np.argsort(-pr[ranked],kind="stable")]
            fraction_cost=da[cal].sum()*fraction
            k=int(np.searchsorted(np.cumsum(da[ranked]),fraction_cost,side="right"))
            route=float(pr[ranked[k]]) if k<len(ranked) else -1e5
            va=old["negative"]&(pr>route)
            vb=ref["negative"]&(pt>route)
        z=changed(old,a,va,weight)
        dd,new_cut=f.strict_budget(z[cal],dev["y"][cal],budget)
        p=changed(ref,b,vb,weight)
        delta,per=f.per_project(test["y"],test["project"],ref["p"],p,fixed)
        key=f"{mode}_{fraction:g}_{weight:g}"
        recipes[key]=dict(dev=f.comparison(dev["y"][cal],base_dev,dd),project_tp=delta,per_project=per,
                          fixed=f.comparison(test["y"],fixed,p>new_cut),test_rows=int(vb.sum()),
                          test_cost_weighted_fraction=float(ta[vb].sum()/ta.sum()),
                          cal_rows=int((cal&va).sum()),cal_cost_weighted_fraction=float(da[cal&va].sum()/da[cal].sum()),
                          route_cutoff=route,decision_cutoff=new_cut)
        with (dest/f"{key}_test.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","gold","weak","reference","joint","route"))
            w.writerows(zip(test["ids"],test["project"],test["y"],f.weak(test["raw"]).astype(int),ref["p"],p,
                            vb.astype(int),strict=True))
        print(rotation,key,delta,recipes[key]["dev"]["delta_tp"],recipes[key]["test_rows"],flush=True)
    w=masks["test"]
    summary=dict(reference_ap=float(average_precision_score(test["y"][w],ref["no"][w])),
                 adviser_ap=float(average_precision_score(test["y"][w],test["adviser"][w])),
                 combined_ap=float(average_precision_score(test["y"][w],b[w])))
    write_json(dest/"results.json",dict(views=summary,recipes=recipes,archive=sha))
    print("AP",rotation,summary,flush=True)


if __name__=="__main__":
    from features import load
    y,side,_,_=load()
    adviser,sha=archived(y)
    for rotation in PARTS:
        run(rotation,adviser,sha,side)
