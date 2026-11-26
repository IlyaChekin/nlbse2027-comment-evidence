"""Prepare token and AST teachers over the full function and predict out of fold."""

import argparse
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz, vstack
from scipy.special import expit, logit
from sklearn.linear_model import SGDClassifier


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round17"))
from view_train import checksum,inputs,json_at,write_json
from structure import assemble


OUT=HERE/"teacher"
CACHE=ROOT/"round12/structure"
FOLDS=ROOT/"round15/folds.npz"
VIEWS=("token","ast")


def positions(sorted_ids, selected):
    at=np.searchsorted(sorted_ids,selected)
    if (at>=len(sorted_ids)).any() or not np.array_equal(sorted_ids[at],selected):
        raise ValueError("Structural feature index misses required row IDs")
    return at


def prepare(workers):
    OUT.mkdir(exist_ok=True)
    texts,labels,_,_,roles=inputs()
    wanted=np.sort(np.concatenate(list(roles.values())))
    ids=[]
    channels={v:[] for v in VIEWS}
    hashes={}
    for role in ("fit","probe","dev","test"):
        path=CACHE/f"{role}.npz"
        with np.load(path) as f:
            row=f["ids"]
            if not np.array_equal(f["y"],labels[row]):
                raise ValueError("A round 12 cache has different labels")
        eligible=np.isin(row,wanted)
        ids.append(row[eligible])
        hashes[path.name]=checksum(path)
        for v in VIEWS:
            source=CACHE/f"{role}_{v}.npz"
            hashes[source.name]=checksum(source)
            mat=load_npz(source)
            if mat.shape[0]!=len(row):
                raise ValueError("Feature cache length differs from row IDs")
            channels[v].append(mat[eligible])
    present=np.concatenate(ids)
    if len(np.unique(present))!=len(present):
        raise ValueError("Feature cache contains duplicate row IDs")
    missing=np.setdiff1d(wanted,present)
    if len(missing):
        _,token,ast=assemble([texts[i] for i in missing],workers=workers)
        channels["token"].append(token)
        channels["ast"].append(ast)
        present=np.concatenate((present,missing))
    order=np.argsort(present)
    if not np.array_equal(wanted,present[order]):
        raise ValueError("New feature store differs from the fixed split")
    np.savez_compressed(OUT/"rows.npz",ids=wanted,y=labels[wanted])
    result={"rows.npz":checksum(OUT/"rows.npz")}
    for key,parts in channels.items():
        target=OUT/f"{key}.npz"
        save_npz(target,vstack(parts,format="csr")[order])
        result[target.name]=checksum(target)
    write_json(OUT/"features.json",dict(rows=len(wanted),cached=len(wanted)-len(missing),generated=len(missing),
                source_sha256=hashes,output_sha256=result,split_sha256=checksum(ROOT/"round13/partition.npz")))
    print("features",len(wanted),"new",len(missing),flush=True)


def open_features():
    cfg=json_at(OUT/"features.json")
    if any(checksum(OUT/name)!=digest for name,digest in cfg["output_sha256"].items()):
        raise ValueError("Structural feature store changed")
    with np.load(OUT/"rows.npz") as f:
        ids,y=f["ids"],f["y"]
    return ids,y,{v:load_npz(OUT/f"{v}.npz") for v in VIEWS}


def train(x,y):
    return SGDClassifier(loss="log_loss",alpha=3e-6,max_iter=45,random_state=42).fit(x,y)


def merged(token,ast):
    return expit(.6*logit(token.clip(1e-6,1-1e-6))+.4*logit(ast.clip(1e-6,1-1e-6)))


def fit():
    ids,labels,x=open_features()
    _,y,_,_,parts=inputs()
    expected=json_at(FOLDS.with_suffix(".json"))["partition_sha256"]
    if checksum(FOLDS)!=expected:
        raise ValueError("Out-of-fold training groups changed")
    wanted=parts["fit"]
    predictions={k:np.empty(len(wanted),dtype=np.float32) for k in VIEWS}
    coverage=np.zeros(len(wanted),dtype=np.int8)
    with np.load(FOLDS) as folds:
        for i in range(5):
            tr,held=folds[f"fit_{i}"],folds[f"held_{i}"]
            if np.intersect1d(tr,held).size or not np.all(np.isin(tr,wanted)):
                raise ValueError("Structural OOF fit intersects its holdout")
            a,b,at=positions(ids,tr),positions(ids,held),positions(wanted,held)
            coverage[at]+=1
            started=time.perf_counter()
            for k in VIEWS:
                clf=train(x[k][a],labels[a,0])
                predictions[k][at]=clf.predict_proba(x[k][b])[:,1]
            print("fold",i,"fit",len(tr),"held",len(held),"elapsed",round(time.perf_counter()-started,2),flush=True)
    if not (coverage==1).all() or not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Out-of-fold coverage or probabilities are invalid")
    p=merged(*[predictions[k] for k in VIEWS])
    oof=OUT/"oof.npz"
    np.savez_compressed(oof,ids=wanted,y=y[wanted],combined=p,**predictions)
    tr=positions(ids,wanted)
    models={k:train(x[k][tr],labels[tr,0]) for k in VIEWS}
    artifact=OUT/"teachers.joblib"
    joblib.dump(models,artifact,compress=3)
    digests={}
    for name in ("probe","dev","test"):
        sl=positions(ids,parts[name])
        pred={k:models[k].predict_proba(x[k][sl])[:,1] for k in VIEWS}
        output=OUT/f"{name}.npz"
        np.savez_compressed(output,ids=ids[sl],y=labels[sl],combined=merged(*[pred[k] for k in VIEWS]),**pred)
        digests[name]=checksum(output)
    write_json(OUT/"fit.json",dict(rows=len(wanted),oof_sha256=checksum(oof),fold_sha256=expected,
                feature_manifest_sha256=checksum(OUT/"features.json"),models_sha256=checksum(artifact),holdout_sha256=digests))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=3)
    args=p.parse_args()
    prepare(args.workers)
    fit()
