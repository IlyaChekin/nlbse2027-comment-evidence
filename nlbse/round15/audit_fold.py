"""Count shared out-of-fold errors before weighting any training example."""

import argparse
import csv
import json

import numpy as np

from crossfit import HERE, N_FOLDS, FOLDS, PLAN, TRAIN, checksum, load, write_json


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--fold",type=int,choices=range(N_FOLDS),required=True)
    fold = args.parse_args().fold
    manifest = json.loads(PLAN.read_text())
    if checksum(FOLDS)!=manifest["partition_sha256"] or checksum(TRAIN)!=manifest["input_sha256"]:
        raise ValueError("Cross-fit data changed")
    with np.load(FOLDS) as f:
        ids=f[f"held_{fold}"]
    _,labels,columns=load(TRAIN,audit=True)
    y=labels[ids]
    root=HERE/f"fold_{fold}"
    probabilities={}
    for family in manifest["families"]:
        file=root/family/"test.npz"
        expected=json.loads(file.with_suffix(".json").read_text())["sha256"]
        if checksum(file)!=expected:
            raise ValueError(f"Untracked probabilities for {family}")
        with np.load(file) as f:
            if not np.array_equal(f["ids"],ids) or not np.array_equal(f["y"],y):
                raise ValueError(f"Mismatched rows for {family}")
            probabilities[family]=f["probabilities"][:,0]
    ordered=np.stack(list(probabilities.values()))
    paths=np.asarray(columns["Filepath"])[ids]
    policy={}
    for cutoff in (.8,.9,.95):
        high=ordered.min(axis=0)>cutoff
        fp=high&(y[:,0]==0)
        policy[str(cutoff)]=dict(joint_high_positive=int((high&(y[:,0]==1)).sum()),joint_false_positive=int(fp.sum()),
                                 files_with_joint_false_positive=len(set(paths[fp])))
    low=(ordered.max(axis=0)<.05)&(y[:,0]==1)
    result=dict(fold=fold,rows=len(ids),vulnerability_positive=int(y[:,0].sum()),
                jointly_high=policy,joint_confident_false_negatives=int(low.sum()))
    with (root/"out_of_fold.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("row","label_vulnerability",*(f"p_{k}" for k in probabilities)))
        writer.writerows(zip(ids,y[:,0],*probabilities.values(),strict=True))
    write_json(root/"audit.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    main()
