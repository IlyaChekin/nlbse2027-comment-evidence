"""Train per-regime sparse rankers with predictions outside the training files."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from sparse_model import HERE, MODES, WORDS, features, predict

ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round22"))
from project_eval import f, checksum, json_at, write_json
from train import cohorts,fit_ids
from project_split import fingerprint

CACHE=HERE/"features"
FOLDS=4


def cache(workers):
    text,y,_,_,_=f.inputs()
    source=f.ROOT/"data/raw/train-00000-of-00001.parquet"
    sha=json_at(f.R19/"partition.json")["train_sha256"]
    if checksum(source)!=sha:
        raise ValueError("Training data changed")
    CACHE.mkdir(exist_ok=True)
    start=time.perf_counter()
    x=features(text,workers)
    save_npz(CACHE/"matrix.npz",x)
    cfg=dict(rows=len(text),features=x.shape[1],nnz=x.nnz,wall_seconds=time.perf_counter()-start,
             data_sha256=sha,matrix_sha256=checksum(CACHE/"matrix.npz"),
             code_sha256=checksum(HERE/"sparse_model.py"))
    write_json(CACHE/"manifest.json",cfg)
    print(cfg,flush=True)


def group_ids(ids,text,project,paths):
    folds=np.empty(len(ids),dtype=np.int8)
    keys=[]
    presence={}
    for j,i in enumerate(ids):
        key=f"NLBSE-R24-42\0{project[i]}\0{paths[i]}".encode()
        fold=hashlib.sha256(key).digest()[0]%FOLDS
        folds[j]=fold
        fp=fingerprint(text[i])
        keys.append(fp)
        presence[fp]=presence.get(fp,0)|(1<<int(fold))
    metadata=np.asarray([presence[k] for k in keys],dtype=np.int8)
    return folds,metadata


def arms(x,y,group,idx):
    models=[]
    for mode in (0,1):
        rows=idx[group[idx]==mode]
        model=LogisticRegression(C=2.,solver="liblinear",max_iter=150,random_state=42)
        model.fit(x[rows],y[rows])
        models.append(model)
    return models


def run(rotation, requested=MODES):
    text,label,project,paths,_=f.inputs()
    manifest=json_at(CACHE/"manifest.json")
    file=CACHE/"matrix.npz"
    if checksum(file)!=manifest["matrix_sha256"]:
        raise ValueError("Feature cache changed")
    both=load_npz(file)
    truth=label[:,0]
    weak,y,cohort=cohorts()
    if not np.array_equal(y,truth):
        raise ValueError("The exact cohorts changed row order")
    idx=np.asarray(fit_ids(rotation))
    if rotation=="official":
        _,_,_,_,held,_=f.dataset(full=True)
        targets={"dev":np.asarray(held)}
    else:
        plan=json_at(f.R19/"partition.json")["rotations"][rotation]
        partition=f.R19/plan["partition_file"]
        if checksum(partition)!=plan["partition_sha256"]:
            raise ValueError("The fixed rotation changed")
        with np.load(partition) as z:
            targets={name:z[name] for name in ("dev","test")}
        if set(project[idx])&set(project[targets["test"]]):
            raise ValueError("Held-out projects appear in sparse fit")
    base=HERE/rotation
    base.mkdir(exist_ok=True)
    start=time.perf_counter()
    fold,dup=group_ids(idx,text,project,paths)
    times={}
    for name in requested:
        x=both if name=="both" else both[:,:WORDS.n_features].tocsr()
        all_p=np.empty((len(idx),2),dtype=np.float32)
        receipts=[]
        for i in range(FOLDS):
            held=fold==i
            keep=(dup&(1<<i))==0
            if np.any(held&keep):
                raise ValueError("Fold leaks text fingerprints")
            estimator=arms(x,truth,weak,idx[keep])
            all_p[held]=predict(estimator,x[idx[held]])
            receipts.append(dict(fold=i,fit=int(keep.sum()),held=int(held.sum()),
                                 copies_removed=int((~keep&~held).sum())))
            print(rotation,name,i,round(time.perf_counter()-start,2),flush=True)
        estimator=arms(x,truth,weak,idx)
        artifact=base/f"{name}.joblib"
        joblib.dump(estimator,artifact,compress=3)
        path=base/f"{name}_oof.npz"
        np.savez_compressed(path,ids=idx,fold=fold,dup=dup,y=truth[idx],weak=weak[idx],logits=all_p)
        times[name]=dict(weights_sha256=checksum(artifact),predictions_sha256=checksum(path),folds=receipts,
                        oof_ap={str(q):float(average_precision_score(
                            truth[idx][weak[idx]==q],all_p[weak[idx]==q,q])) for q in (0,1)})
        for phase,selected in targets.items():
            out=base/f"{name}_{phase}.npz"
            np.savez_compressed(out,ids=selected,logits=predict(estimator,x[selected]))
            times[name][phase]=checksum(out)
    write_json(base/"fit.json",dict(rows=len(idx),manifest_sha256=checksum(CACHE/"manifest.json"),
                                   exact_cohorts_sha256=cohort,models=times))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("cache","test_a","test_b","official"))
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--modes",nargs="+",choices=MODES,default=MODES)
    a=p.parse_args()
    cache(a.workers) if a.stage=="cache" else run(a.stage,a.modes)
