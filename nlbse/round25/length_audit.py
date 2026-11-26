"""Count errors by function length in the measured round 24 policy."""
import csv
import math
from pathlib import Path

import numpy as np

from length_eval import HERE,ROOT,R24,f,checksum,json_at,write_json
from pipeline import PUBLIC,load

BANDS=((0,128),(128,256),(256,512),(512,1024),(1024,float("inf")))


def read(path):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=list(range(len(rows))):
        raise ValueError(f"Row order changed in {path}")
    y=np.asarray([int(r["label_vulnerability"]) for r in rows],dtype=np.int8)
    v=np.asarray([int(r["pred_vulnerability"]) for r in rows],dtype=bool)
    return y,v


def main():
    texts,y,_=load(PUBLIC)
    length=np.fromiter((len(s) for s in texts),dtype=np.int32)
    pub=json_at(R24/"public/results.json")
    name=pub["selected"]
    outcome={}
    for key in ("reference",name):
        file=R24/"public"/f"{key.replace(':','_')}.csv"
        if pub["results"][key]["predictions_sha256"]!=checksum(file):
            raise ValueError("Round 24 archived decisions changed")
        gold,pred=read(file)
        if not np.array_equal(gold,y[:,0]):
            raise ValueError("Reference labels differ")
        outcome[key]=pred
    src=ROOT/"round22/union_public/scanner_predictions.npz"
    if checksum(src)!=json_at(ROOT/"round22/union_public/results.json")["scanner_predictions_sha256"]:
        raise ValueError("Frozen scanner flags changed")
    with np.load(src) as z:
        selected=z["gate"].copy()
        selected[selected]=z["flag"]==0
    base,new=outcome["reference"],outcome[name]
    if np.any(base[~selected]!=new[~selected]):
        raise ValueError("Sparse correction touched an unselected example")
    stat={}
    for low,high in BANDS:
        take=selected&(length>low)&(length<=high)
        key=f"{low+1}-{int(high)}" if math.isfinite(high) else f"{low+1}+"
        stat[key]=dict(rows=int(take.sum()),**f.comparison(y[take,0],base[take],new[take]))
    if sum(x["rows"] for x in stat.values())!=selected.sum():
        raise ValueError("Length bands do not cover routed functions")
    write_json(HERE/"public_length_audit.json",dict(rows=len(y),bands=stat,selected=int(selected.sum()),
                measured_public_sha256=checksum(PUBLIC),scanner_sha256=checksum(src),source_round=24))
    print(stat,flush=True)


if __name__=="__main__":
    main()
