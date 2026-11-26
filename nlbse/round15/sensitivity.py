"""Quantify how much precision changes can close the gap to the local target."""

import csv
import json
import math

import numpy as np

from crossfit import HERE, ROOT, checksum, write_json
from pipeline import PUBLIC, load, metric
from score_runtime import competition_formula


ARCHIVES = {
    "release": "round12/efficient_aug/selective/predictions.csv",
    "unix": "round10/unix_context/unix_lexical/predictions.csv",
    "t5": "round9/public/quality/predictions.csv",
    "retrieval_unix": "round11/neighbors/unixcoder9/predictions.csv",
    "small128": "round11/small/public/predictions.csv",
    "adaptive": "round11/cascade/adaptive/predictions.csv",
    "expanded_sparse": "portfolio/learned_sparse/runs/expansion/public/predictions.csv",
    "organizer": "runs/official_packed_public/predictions.csv",
    "comment_window": "round6/masking/public/full/predictions.csv",
    "bm25": "portfolio/simple/runs/bm25_comments_bigram/public/predictions.csv",
}


def archived(file, expected):
    bits, probs = [], []
    with file.open(newline="") as stream:
        for i,r in enumerate(csv.DictReader(stream)):
            if int(r["row"]) != i:
                raise ValueError(f"Row order changed in {file}")
            columns = ("pred_vulnerability","pred_MAT") if "pred_MAT" in r else ("vulnerability","MAT")
            bits.append(tuple(int(r[k]) for k in columns))
            probs.append(float(r["p_vulnerability"]))
    if len(bits) != expected:
        raise ValueError(f"Wrong split in {file}")
    return np.asarray(bits,dtype=bool),np.asarray(probs)


def main():
    manifest=json.loads((ROOT/"data/manifest.json").read_text())["files"]
    expected=next(r["sha256"] for r in manifest if r["split"]=="test")
    if checksum(PUBLIC)!=expected:
        raise ValueError("The public evaluation data changed")
    _, y, _=load(PUBLIC)
    predictions={key:archived(ROOT/path,len(y)) for key,path in ARCHIVES.items()}
    base, probability=predictions["release"]
    old = json.loads((ROOT/"round12/efficient_aug/analysis.json").read_text())["selective"]
    cost=old["cost"]
    truth=y.astype(bool)
    negative=~truth[:,0]
    fp=base[:,0]&negative
    shared=fp&(probability>.9)&(predictions["unix"][1]>.9)
    pos=int(truth[:,0].sum())
    vulnerability_tp=int((base[:,0]&~negative).sum())
    mat_f1=old["metrics"]["MAT"]["f1"]
    target=2*(.97-competition_formula(0,cost["forward_seconds"],cost["analytic_gflops"]))/0.6-mat_f1

    def quality(bits):
        m=metric(y,bits)
        return dict(metrics=m,cost_assumption="unchanged reference cost",
                    score=competition_formula(m["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                    zero_cost_score=.4+.6*m["mean"]["f1"])

    port=np.stack([b for b,_ in predictions.values()])
    observed_correct=(port==truth).any(axis=0)
    best=truth^~observed_correct
    b=base.copy()
    b[shared,0]=0
    c=base.copy()
    c[fp,0]=0
    def wrong_by_label(j):
        return {"always_false_negative":int((~observed_correct[:,j]&truth[:,j]).sum()),
                "always_false_positive":int((~observed_correct[:,j]&~truth[:,j]).sum())}
    result=dict(archive_paths=ARCHIVES,public_sha256=expected,reference=quality(base),fp_count=int(fp.sum()),
                shared_high_fp_count=int(shared.sum()),all_shared_high_fp_removed=quality(b),all_fp_removed=quality(c),
                extra_vulnerability_tp_needed_after_all_fp_removed=max(0,math.ceil(pos*target/(2-target))-vulnerability_tp),
                binary_vulnerability_oracle_fixed_MAT=quality(np.column_stack((best[:,0],base[:,1]))),
                binary_two_label_oracle=quality(best),vulnerability=wrong_by_label(0),mat=wrong_by_label(1))
    write_json(HERE/"sensitivity.json",result)
    print({"reference":result["reference"]["score"],"shared_fp":result["shared_high_fp_count"],
           "all_fp_score":result["all_fp_removed"]["score"],"tp_needed_after_removing_all_fp":result["extra_vulnerability_tp_needed_after_all_fp_removed"],
           "oracles":[result[k]["score"] for k in ("binary_vulnerability_oracle_fixed_MAT","binary_two_label_oracle")]},flush=True)


if __name__=="__main__":
    main()
