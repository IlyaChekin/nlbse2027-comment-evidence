"""Predict using the internally selected sparse correction and the proven static gate."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round22"))
from predict_union import predict_factor as pf, PROGRAM, recover

from fuse_sparse import correction


def decisions(texts,config,model,tokenizer,heads):
    record=json.loads((HERE/"entry.json").read_text())
    pf.require_hash(PROGRAM,record["scanner_sha256"])
    pf.require_hash(HERE/"correction.joblib",record["weights_sha256"])
    package=joblib.load(HERE/"correction.joblib")
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
            raise ValueError("The verified rule scanner returned unexpected columns")
        known[asked]=raw[:,0]>=0
        flag[asked]=raw[:,0]==1
        q[known]=np.where(flag[known],yes[known],no[known])
    state=dict(p=q.astype(np.float32),w=est["w"],no=no,yes=yes,known=known,flag=flag)
    vulnerability,_,_,_=correction(texts,values,state,config["threshold"],record["variant"],package)
    return np.column_stack((vulnerability,recover(texts,p[:,1]>.5))).astype(int)


if __name__=="__main__":
    pf.main(decisions)
