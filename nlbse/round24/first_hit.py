"""Assess the identity of the first firing rule without asking for any later rule."""
import json
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from conditioned import eligible,match
from veto import (HERE,R22,PARTS,REFERENCE,f,calibrate,checksum,json_at,write_json,exact_state)
from project_eval import raw_values
from train import fit_ids

OUT=HERE/"first_hit"


def order_for(rotation):
    plan=json_at(R22/"union_scanner.json")
    old=json_at(f.SOURCE/"policy.json")["rule_columns"]+["rule_195"]
    tags=[t.removeprefix("rule_") for t in old]
    if rotation=="official":
        order=[tags.index(t) for t in plan["order"]]
    else:
        view=raw_values(fit_ids(rotation))>0
        remain=np.ones(len(view),dtype=bool)
        costs=plan["estimated_costs"]
        active=set(range(len(tags)))
        order=[]
        while active:
            win=max(active,key=lambda i:(int((remain&view[:,i]).sum())/costs[tags[i]],-i))
            order.append(win)
            remain &= ~view[:,win]
            active.remove(win)
    return order,[tags[i] for i in order]


def first(raw,order):
    b=(raw[:,order]>0)
    return np.where(b.any(1),b.argmax(1)+1,0)


def signature(frame,order,mode):
    flag=first(frame["raw"],order)
    one=np.eye(len(order)+1,dtype=np.float32)[flag,1:]
    base=f.features(frame[REFERENCE],"positive")
    if mode=="identity":
        return np.column_stack((base,one))
    strength=np.clip(base[:,1],-4,4)
    return np.column_stack((base,one,one*strength[:,None]))


def project(rotation):
    out=OUT/rotation
    out.mkdir(parents=True,exist_ok=True)
    train=json_at(f.WEAK/rotation/"policy.json")
    frames={}
    for phase in ("dev","test"):
        with np.load(R22/rotation/f"{phase}.npz") as z:
            frames[phase]=dict(z)
    a,b=frames["dev"],frames["test"]
    fit=np.isin(a["ids"],train["fit_rows"])
    cal=np.isin(a["ids"],train["calibration_rows"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,cut,_,_=calibrate(frames,REFERENCE,fit,cal,budget)
    d,t=exact_state(a,heads,gate),exact_state(b,heads,gate)
    ref,dref=t["p"]>cut,d["p"]>cut
    order,tags=order_for(rotation)
    hit=fit&f.weak(a["raw"])
    u,v=eligible(d,"yes"),eligible(t,"yes")
    matchedb=int((dref&u&cal&(a["y"]==0)).sum())
    records={}
    weights={}
    for mode in ("identity","interaction"):
        sx=signature(a,order,mode)
        st=signature(b,order,mode)
        h=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(sx[hit],a["y"][hit])
        q=h.predict_proba(sx[u])[:,1]
        p=h.predict_proba(st[v])[:,1]
        weights[mode]=h
        for coefficient in (.3,.6,1.):
            q2=(1-coefficient)*d["p"][u]+coefficient*q
            p2=(1-coefficient)*t["p"][v]+coefficient*p
            bits,threshold=f.strict_budget(q2[cal[u]],a["y"][u][cal[u]],matchedb)
            dev=f.comparison(a["y"][u][cal[u]],dref[u][cal[u]],bits)
            exact,projs=match(b["y"],b["project"],ref,v,p2)
            frozen=ref.copy()
            frozen[v]=p2>threshold
            key=f"{mode}_{coefficient:g}"
            records[key]=dict(dev=dev,matched=f.comparison(b["y"],ref,exact),
                              fixed=f.comparison(b["y"],ref,frozen),project=projs,threshold=threshold)
            np.savez_compressed(out/f"{key}.npz",ids=b["ids"],selected=v,score=p2,baseline=ref,
                                matched=exact,frozen=frozen,y=b["y"])
            print(rotation,key,dev["delta_tp"],records[key]["matched"]["delta_tp"],flush=True)
    dst=out/"heads.joblib"
    joblib.dump(weights,dst,compress=3)
    best=max(records,key=lambda k:records[k]["dev"]["delta_tp"])
    write_json(out/"results.json",dict(order=tags,fit_rows=int(hit.sum()),test_rows=int(v.sum()),dev_fp=matchedb,
                 selection=best if records[best]["dev"]["delta_tp"]>0 else REFERENCE,options=records,weight_sha256=checksum(dst)))


if __name__=="__main__":
    for name in PARTS:
        project(name)
