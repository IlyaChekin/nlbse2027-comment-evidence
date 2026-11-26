"""Audit exactly which shared errors transfer features fix and where they cost errors."""
import csv
import json

import numpy as np
from sklearn.metrics import average_precision_score

from project_heads import OUT as HEADS, corpus
from project_split import HERE
from pipeline import write_json


def read(path, ids):
    with path.open(newline="") as stream:
        table=list(csv.DictReader(stream))
    if [int(row["row"]) for row in table]!=ids.tolist():
        raise ValueError("Different heldout IDs in saved predictions")
    p=np.asarray([float(r["p_vulnerability"]) for r in table])
    v=np.asarray([int(r["pred_vulnerability"]) for r in table],dtype=bool)
    return p,v


def main():
    _,y,project,_,roles=corpus()
    ids=roles["test"]
    gold=y[ids,0]
    groups=project[ids]
    config=json.loads((HEADS/"policy.json").read_text())
    old={name: read(HEADS/f"{name}_raw_test.csv",ids) for name in ("compact","unix")}
    common=(old["compact"][1]!=gold)&(old["unix"][1]!=gold)
    result={}
    for name in ("compact","unix"):
        base_p,base=old[name]
        files={**{mode:HEADS/f"{name}_{filename}_test.csv" for mode,filename in config["choices"][name]["representatives"].items()},
               "retrieval":HERE/f"retrieval/{name}_test_predictions.csv"}
        for label,path in files.items():
            if not path.is_file():
                continue
            p,v=read(path,ids)
            groups_to_report={"pooled":np.full(len(gold),True),**{key:groups==key for key in np.unique(groups)}}
            details={}
            for g,z in groups_to_report.items():
                details[g]=dict(average_precision_baseline=average_precision_score(gold[z],base_p[z]),
                           average_precision_candidate=average_precision_score(gold[z],p[z]),
                           repaired_false_negatives=int((z&(gold==1)&~base&v).sum()),
                           repaired_false_positives=int((z&(gold==0)&base&~v).sum()),
                           new_false_negatives=int((z&(gold==1)&base&~v).sum()),
                           new_false_positives=int((z&(gold==0)&~base&v).sum()),
                           repaired_joint_errors=int((z&common&(v==gold)).sum()))
            result[f"{name}/{label}"]=details
    rescue={}
    p_c,p_u=old["compact"][0],old["unix"][0]
    for threshold in (.05,.10,.20):
        rescue[str(threshold)]=dict(shared_wrong_low=int((common&(gold==1)&(p_c<threshold)&(p_u<threshold)).sum()),
                        shared_wrong_high=int((common&(gold==0)&(p_c>1-threshold)&(p_u>1-threshold)).sum()))
    write_json(HERE/"error_audit.json",dict(by_candidate=result,wrong_with_strong_agreement=rescue))
    print(result["compact/token_motifs"],rescue,flush=True)


if __name__=="__main__":
    main()
