"""Prepare the rule-negative and rule-positive regimes from official training rows."""
import argparse
from pathlib import Path
import sys

import numpy as np


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round20"))
from evaluate_rules import base,inputs,json_at,raw_rule_counts,write_json
from subsets import numeric_counts


def prepare():
    a=raw_rule_counts()
    b=numeric_counts()
    if len(a)!=len(b):
        raise ValueError("Both rule passes must cover the same official training rows")
    present=(a>0).any(1)|(b>0)
    known=present|((a>=0).all(1)&(b>=0))
    flags=np.where(known,present.astype(np.int8),-1).astype(np.int8)
    _,y,_,_,_=inputs()
    if len(y)!=len(flags):
        raise ValueError("Training source labels do not align with the rule outcomes")
    if not np.array_equal(y[:,0],y[:,0].astype(bool)):
        raise ValueError("Vulnerability labels must be binary")
    target=HERE/"cohorts.npz"
    tmp=target.with_suffix(".part.npz")
    np.savez_compressed(tmp,rows=np.arange(len(y),dtype=np.int32),weak=flags,y=y[:,0])
    tmp.replace(target)
    summary={}
    folder=ROOT/"round19"
    for name in base.ROTATIONS:
        path=folder/f"{name}.npz"
        partitions=json_at(folder/"partition.json")["rotations"]
        if base.checksum(path)!=partitions[name]["partition_sha256"]:
            raise ValueError("Project rotation changed")
        with np.load(path) as data:
            summary[name]={
                phase:{state:dict(rows=int((flags[ids]==c).sum()),
                                  positives=int(((flags[ids]==c)&(y[ids,0]==1)).sum()))
                       for state,c in (("weak_negative",0),("weak_positive",1),("unobserved",-1))}
                for phase,ids in ((field,data[field]) for field in ("fit","dev","test"))
            }
    previous=json_at(folder/"weak_satd/train_parallel.json")
    numeric=json_at(ROOT/"round20/extended_counts.json")
    write_json(HERE/"cohorts.json",dict(rows=len(y),sha256=base.checksum(target),
                   raw_rule_counts_sha256=previous["sha256"],numeric_counts_sha256=numeric["sha256"],
                   train_sha256=json_at(folder/"partition.json")["train_sha256"],rotations=summary))
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
