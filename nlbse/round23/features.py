"""Build the syntax feature store on the fixed official training set."""
import argparse
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import load_npz,save_npz

from branches import HERE,DENSE,collect

ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round22"))
from project_eval import f,checksum,json_at,write_json

OUT=HERE/"features"
FILES=("rows.npz","flat.npz","linked.npz")


def build(workers):
    texts,y,_,_,_=f.inputs()
    source=f.ROOT/"data/raw/train-00000-of-00001.parquet"
    sha=json_at(f.R19/"partition.json")["train_sha256"]
    if checksum(source)!=sha:
        raise ValueError("Training data differs from the fixed rotations")
    OUT.mkdir(exist_ok=True)
    t=time.perf_counter()
    dense,flat,linked=collect(texts,workers)
    np.savez_compressed(OUT/"rows.npz",ids=np.arange(len(texts),dtype=np.int32),y=y[:,0],dense=dense)
    save_npz(OUT/"flat.npz",flat)
    save_npz(OUT/"linked.npz",linked)
    rec=dict(rows=len(texts),wall_seconds=time.perf_counter()-t,source_sha256=sha,
           source_code_sha256=checksum(HERE/"branches.py"),workers=workers,dense_columns=DENSE,
           tree_sitter=importlib.metadata.version("tree-sitter"),cpp_grammar=importlib.metadata.version("tree-sitter-cpp"),
           parse_errors=int((dense[:,DENSE.index("parse_error")]>0).sum()),nnz=[flat.nnz,linked.nnz],
           files={k:checksum(OUT/k) for k in FILES})
    write_json(OUT/"manifest.json",rec)
    print(rec,flush=True)


def load():
    receipt=json_at(OUT/"manifest.json")
    if any(checksum(OUT/k)!=h for k,h in receipt["files"].items()):
        raise ValueError("A syntax feature file has changed")
    with np.load(OUT/"rows.npz") as a:
        if not np.array_equal(a["ids"],np.arange(receipt["rows"])):
            raise ValueError("Unexpected feature row alignment")
        y,dense=a["y"],a["dense"]
    return y,dense,load_npz(OUT/"flat.npz"),load_npz(OUT/"linked.npz")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=5)
    build(p.parse_args().workers)
