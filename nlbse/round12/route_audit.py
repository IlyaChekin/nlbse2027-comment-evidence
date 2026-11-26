"""Locate the effect of the second view at the heldout policy's fixed threshold."""
import csv
import json

import numpy as np

from augmentation import DEST as AUG
from heldout_route import OUT as ROUTE
from project_heads import corpus
from project_split import HERE
from pipeline import write_json


def scores(path,ids):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if [int(r["row"]) for r in rows]!=ids.tolist():
        raise ValueError("Different heldout prediction IDs")
    return np.asarray([float(r["p_vulnerability"]) for r in rows])


def main():
    _,y,_,_,split=corpus()
    ids=split["test"]
    gold=y[ids,0].astype(bool)
    raw=scores(AUG/"aug_raw.csv",ids)
    norm=scores(AUG/"aug_norm.csv",ids)
    policy=json.loads((ROUTE/"policy.json").read_text())["chosen"]
    t=policy["threshold"]
    early=raw>t
    both=(raw+norm)/2>t
    results={}
    for name,mask in (("below_gate",raw<=policy["lower"]),("within_gate",(raw>policy["lower"])&(raw<policy["upper"])),
                      ("above_gate",raw>=policy["upper"])):
        results[name]=dict(rows=int(mask.sum()),ties_to_truth=int((mask&(early==gold)&(both==gold)).sum()),
                           fixes=int((mask&(early!=gold)&(both==gold)).sum()),
                           new_errors=int((mask&(early==gold)&(both!=gold)).sum()))
    write_json(HERE/"route_audit.json",results)
    print(results,flush=True)


if __name__=="__main__":
    main()
