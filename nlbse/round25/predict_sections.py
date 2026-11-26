"""Predict with the verified Round 24 classifier and the frozen region correction."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round24"))
from predict_sparse import pf,PROGRAM,recover
from section_runtime import adjust


def decisions(texts,config,model,tokenizer,heads):
    settings=json.loads((HERE/"entry.json").read_text())
    pf.require_hash(PROGRAM,settings["scanner_sha256"])
    pf.require_hash(HERE/"section_head.joblib",settings["head_sha256"])
    base=HERE.parent/"round24"
    earlier=json.loads((base/"entry.json").read_text())
    pf.require_hash(base/"correction.joblib",earlier["weights_sha256"])
    package=joblib.load(base/"correction.joblib")
    head=joblib.load(HERE/"section_head.joblib")
    p,aux=pf.infer(model,tokenizer,texts,auxiliary=True)
    route=config["route"]
    second=(p[:,0]>route["lower"])&(p[:,0]<route["upper"])
    if second.any():
        canonical=pf.canonicalize([s for s,on in zip(texts,second,strict=True) if on])
        q,_=pf.infer(model.encoder,tokenizer,canonical)
        p[second,0]=.5*(p[second,0]+q[:,0])
    values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
    est=pf.estimate(heads,values)
    q,no,yes,asked=pf.choose(values,est,pf.Policy(**config["policy"]),config["gate"])
    known=np.zeros(len(texts),dtype=bool)
    flag=np.zeros_like(known)
    if asked.any():
        cols,raw=pf.scanner([s for s,on in zip(texts,asked,strict=True) if on],program=PROGRAM)
        if cols!=["rule_union"]:
            raise ValueError("The verified static scanner returned unexpected columns")
        known[asked]=raw[:,0]>=0
        flag[asked]=raw[:,0]==1
        q[known]=np.where(flag[known],yes[known],no[known])
    state=dict(p=q.astype(np.float32),w=est["w"],no=no,yes=yes,known=known,flag=flag)
    vr,_,_,_,_=adjust(texts,values,state,config["threshold"],earlier["variant"],package,
                         settings["option"],settings["length_cutoff"],head)
    return np.column_stack((vr,recover(texts,p[:,1]>.5))).astype(int)


if __name__=="__main__":
    pf.main(decisions)
