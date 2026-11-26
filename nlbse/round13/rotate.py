"""Second project rotation for short-function transfer, with disjoint file probes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0]=[str(ROOT/"round12"),str(ROOT)]
from project_split import checksum, family, fingerprint
from pipeline import TRAIN, load, write_json

SPLIT=HERE/"partition.npz"
PLAN=HERE/"partition.json"
TEST={"mozilla firefox","android","php","imagemagick","radare2","krb5","file"}
DEV={"chromium","qemu","ffmpeg","tcpdump"}


def read_partition():
    cfg=json.loads(PLAN.read_text())
    if checksum(SPLIT)!=cfg["partition_sha256"] or checksum(TRAIN)!=cfg["input_sha256"]:
        raise ValueError("Rotated partition or dataset changed")
    with np.load(SPLIT) as f:
        return {k:f[k] for k in ("fit","probe","dev","test")}


def partition():
    inputs,y,m=load(TRAIN,audit=True)
    projects=[family(name) for name in m["Projectname"]]
    groups=list(zip(projects,m["Filepath"],strict=True))
    expected=json.loads((ROOT/"data/manifest.json").read_text())["files"][0]["sha256"]
    if checksum(TRAIN)!=expected:
        raise ValueError("Training dataset changed")
    prior_dir=ROOT/"round12"
    with np.load(prior_dir/"partition.npz") as old:
        previous=old["test"]
    # The short cutoff comes from the previous, explicitly exploratory rotation.
    limit=int(np.quantile([len(inputs[i]) for i in previous],.25))
    original=[]
    by_role={k:[] for k in ("test","dev","probe","fit")}
    for i,(project,file) in enumerate(groups):
        if project in TEST:
            role="test"
        elif project in DEV:
            role="dev"
        else:
            value=hashlib.sha256(f"NLBSE-R13-42\0{project}\0{file}".encode()).digest()
            role="probe" if int.from_bytes(value[:4],"big")%100<12 else "fit"
        by_role[role].append(i)
        original.append(fingerprint(inputs[i]))
    blocked=set()
    keep={}
    excluded={}
    for role,ids in by_role.items():
        keep[role]=np.asarray([i for i in ids if original[i] not in blocked],dtype=np.int32)
        excluded[role]=len(ids)-len(keep[role])
        blocked.update(original[i] for i in keep[role])
    if set(groups[i] for i in keep["fit"]) & set(groups[i] for i in keep["probe"]):
        raise ValueError("Repeated fit and probe file")
    if set(projects[i] for i in keep["fit"]) & (TEST|DEV):
        raise ValueError("Held-out project in training")
    temp=SPLIT.with_suffix(".part.npz")
    np.savez_compressed(temp,**keep)
    temp.replace(SPLIT)
    record=dict(input_sha256=expected,partition_sha256=checksum(SPLIT),short_chars_max=limit,
        jointly_low_cutoff=.05,excluded_cross_role_token_copies=excluded,roles={role:dict(rows=len(ids),
        projects=dict(Counter(projects[i] for i in ids)),labels=y[ids].sum(0).tolist(),
        short_rows=sum(len(inputs[i])<=limit for i in ids)) for role,ids in keep.items()})
    write_json(PLAN,record)
    print(json.dumps(record,indent=2),flush=True)


def load_backbones():
    import backbones
    backbones.MODELS=HERE/"backbones"
    backbones.PLAN=PLAN
    backbones.read_partition=read_partition
    return backbones


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("partition","fit","score"))
    parser.add_argument("--family",choices=("unix","compact"))
    parser.add_argument("--parts",nargs="+",choices=("fit","probe","dev","test"))
    a=parser.parse_args()
    if a.phase=="partition":
        partition()
    else:
        if a.family is None:
            parser.error("--family is required for model operations")
        model=load_backbones()
        if a.phase=="fit":
            model.train(a.family,2 if a.family=="unix" else 5)
        elif a.parts:
            model.export_scores(a.family,a.parts)
        else:
            parser.error("--parts is required for score")
