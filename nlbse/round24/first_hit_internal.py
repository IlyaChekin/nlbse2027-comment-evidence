"""Verify transfer of first-hit identity with a frozen threshold on internal files."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from internal_sparse import recipe
from first_hit import HERE,OUT,PARTS,REFERENCE,f,order_for,signature,raw_values,exact_state,checksum,json_at,write_json
from conditioned import eligible


def tune():
    out=OUT/"official"
    out.mkdir(exist_ok=True)
    _,_,text,y,ids,_=f.dataset(full=True)
    with np.load(HERE.parent/"round8/frozen/holdout.npz") as z:
        if not np.array_equal(z["rows"],ids):
            raise ValueError("Official calibration rows changed")
        fold=z["folds"]
    support={}
    tables=[json_at(OUT/r/"results.json")["options"] for r in PARTS]
    for name in tables[0]:
        gains=[t[name]["matched"]["delta_tp"] for t in tables]
        if min(gains)>0:
            support[name]=gains
    ref,policy,heads=recipe()
    values,_,_=f.factor_data("official",text,f.small.tokenizer())
    frame={REFERENCE:values,"raw":raw_values(ids)}
    base=exact_state(frame,heads,ref["gate"],policy=policy)
    old=base["p"]>ref["threshold"]
    scope=eligible(base,"yes")
    train=f.weak(frame["raw"])&(fold<3)
    cal,rev=fold==3,fold==4
    truth=y[:,0]
    sec=int((scope&cal&old&(truth==0)).sum())
    order,tags=order_for("official")
    info={}
    setters={}
    for mode in ("identity","interaction"):
        x=signature(frame,order,mode)
        model=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(x[train],truth[train])
        q=model.predict_proba(x[scope])[:,1]
        setters[mode]=model
        for ratio in (.3,.6,1.):
            name=f"{mode}_{ratio:g}"
            if name not in support:
                continue
            scores=(1-ratio)*base["p"][scope]+ratio*q
            _,threshold=f.strict_budget(scores[cal[scope]],truth[cal&scope],sec)
            new=old.copy()
            new[scope]=scores>threshold
            dev=f.comparison(truth[cal],old[cal],new[cal])
            review=f.comparison(truth[rev],old[rev],new[rev])
            info[name]=dict(cal=dev,review=review,threshold=threshold,project_tp=support[name])
            np.savez_compressed(out/f"{name}.npz",ids=np.asarray(ids)[cal|rev],truth=truth[cal|rev],
                                baseline=old[cal|rev],frozen=new[cal|rev],eligible=scope[cal|rev])
            print(name,dev["delta_tp"],dev["delta_fp"],review["delta_tp"],review["delta_fp"],flush=True)
    dest=out/"heads.joblib"
    joblib.dump(setters,dest,compress=3)
    best=max(info,key=lambda name:info[name]["cal"]["f1_after"],default=REFERENCE)
    if best!=REFERENCE and info[best]["cal"]["delta_tp"]<=0:
        best=REFERENCE
    write_json(out/"policy.json",dict(selected=best,options=info,fit_rows=int(train.sum()),cal_yes_fp=sec,
                                     order=tags,hash=checksum(dest)))
    print("selected",best,flush=True)


if __name__=="__main__":
    tune()
