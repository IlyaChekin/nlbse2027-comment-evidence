"""Prepare file-disjoint training folds for the next NLBSE precision experiment."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/"round13"), str(ROOT/"round12"), str(ROOT)]
from rotate import read_partition, load_backbones
from project_split import checksum, family, fingerprint
from pipeline import TRAIN, load, write_json


N_FOLDS = 5
PLAN = HERE/"folds.json"
FOLDS = HERE/"folds.npz"


def file_fold(project, path):
    value = hashlib.sha256(f"NLBSE-R15-42\0{project}\0{path}".encode()).digest()
    return int.from_bytes(value[:8], "big") % N_FOLDS


def prepare():
    texts, labels, fields = load(TRAIN, audit=True)
    parent = read_partition()
    eligible = parent["fit"]
    paths = [(family(fields["Projectname"][i]), fields["Filepath"][i]) for i in eligible]
    folds = np.fromiter((file_fold(*key) for key in paths), dtype=np.int8, count=len(paths))
    normalized = [fingerprint(texts[i]) for i in eligible]
    membership = defaultdict(int)
    for fp, fold in zip(normalized, folds, strict=True):
        membership[fp] |= 1 << int(fold)
    archive, report = {}, {}
    for fold in range(N_FOLDS):
        held = folds == fold
        text_overlap = np.fromiter((bool(membership[fp] & (1 << fold)) for fp in normalized), dtype=bool, count=len(folds))
        fit = ~held & ~text_overlap
        a,b = eligible[fit],eligible[held]
        left = {key for key, keep in zip(paths,fit,strict=True) if keep}
        right = {key for key, keep in zip(paths,held,strict=True) if keep}
        if left & right or set(a) & set(b):
            raise ValueError(f"File-disjoint fold {fold} overlaps")
        archive[f"fit_{fold}"], archive[f"held_{fold}"] = a,b
        report[str(fold)]=dict(fit_rows=len(a),held_rows=len(b),fit_files=len(left),held_files=len(right),
                              copies_removed=int((~held & text_overlap).sum()),held_positives=labels[b].sum(0).tolist())
    if not np.array_equal(np.sort(np.concatenate([archive[f"held_{i}"] for i in range(N_FOLDS)])), eligible):
        raise ValueError("Held folds do not cover the source fit exactly once")
    temp=FOLDS.with_suffix(".part.npz")
    np.savez_compressed(temp,**archive)
    temp.replace(FOLDS)
    parent_split = ROOT/"round13/partition.npz"
    write_json(PLAN,dict(parent_split_sha256=checksum(parent_split),input_sha256=checksum(TRAIN),
                         partition_sha256=checksum(FOLDS),counts=report,families=("unix","compact"),
                         epochs={"unix":1,"compact":2},seed=42))
    print(report,flush=True)


def pilot(fold):
    cfg=json.loads(PLAN.read_text())
    if checksum(TRAIN)!=cfg["input_sha256"] or checksum(FOLDS)!=cfg["partition_sha256"]:
        raise ValueError("Cross-fit input changed")
    with np.load(FOLDS) as stored:
        fit, held = stored[f"fit_{fold}"], stored[f"held_{fold}"]
    base=load_backbones()
    base.MODELS=HERE/f"fold_{fold}"
    base.PLAN=PLAN
    base.read_partition=lambda: {"fit":fit,"test":held}
    for name in ("unix","compact"):
        base.train(name,cfg["epochs"][name])
        base.export_scores(name,["fit","test"] if name=="unix" else ["test"])
    print(f"Completed held-file predictions for fold {fold}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","pilot"))
    parser.add_argument("--fold",choices=range(N_FOLDS),type=int,default=0)
    args=parser.parse_args()
    prepare() if args.stage=="prepare" else pilot(args.fold)
