"""Quantify which errors are visible through disagreement on training-only scores."""
import argparse
from pathlib import Path

import numpy as np

from disagreement import HERE,ROOT,PARTS,checksum,json_at,write_json,estimate_training,LIMIT,encoded_context


def counts(y,p):
    return dict(tp=int((y&p).sum()),fp=int((~y&p).sum()),fn=int((y&~p).sum()),tn=int((~y&~p).sum()))


def inspect_out_of_file(rotation):
    path=HERE/"disagreement"/rotation
    expected=json_at(path/"results.json")
    if checksum(path/"training.npz")!=expected["oof_sha256"]:
        raise ValueError("Frozen training-only score source changed")
    with np.load(path/"training.npz") as z:
        data=dict(z)
    probability=estimate_training(data["truth"],data["compact"],data["sparse"],data["folds"])
    a,b,c=probability.T
    if not np.allclose(c,data["proxy"],rtol=0,atol=1e-12):
        raise ValueError("The recorded score proxy changed")
    y=data["truth"].astype(bool)
    yes=c>=LIMIT
    cross=(a>=LIMIT)!=(b>=LIMIT)
    choices={"all":np.ones(len(y),dtype=bool),"above_0.10_gap":np.abs(a-b)>=.10,
             "crosses_0.18":cross,"both_below_0.18":(a<LIMIT)&(b<LIMIT),
             "both_above_0.18":(a>=LIMIT)&(b>=LIMIT)}
    table={}
    for k,q in choices.items():
        table[k]=dict(rows=int(q.sum()),**counts(y[q],yes[q]))
    dst=HERE/"error_census"/rotation
    dst.mkdir(parents=True,exist_ok=True)
    # Independent proxy scores are reusable for a different training objective in this cohort.
    np.savez_compressed(dst/"evidence.npz",ids=data["rows"],fold=data["folds"],gold=data["truth"],
                        scores=probability,agree=~cross)
    write_json(dst/"census.json",dict(roles=table,source_sha256=checksum(path/"training.npz"),
                                     evidence_sha256=checksum(dst/"evidence.npz")))
    print(rotation,table,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    inspect_out_of_file(p.parse_args().rotation)
