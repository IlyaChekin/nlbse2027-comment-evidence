"""Use only weak-negative, file-disjoint fit neighbors to correct weak-negative decisions."""
import argparse
from itertools import product
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from length_eval import (HERE,ROOT,PARTS,f,match,REFERENCE,source_contexts,checksum,json_at,write_json,log_odds)
from train import cohorts,fit_ids
from project_split import fingerprint
sys.path.insert(0,str(ROOT/"round11"))
from build_index import export
from neighbor_eval import neighbors,features

OUT=HERE/"retrieval"
CODE=ROOT/"round11/neighbors.cpp"
MODES=tuple(product(("all","short","long"),(.5,1.)))


def extract(rotation,d,b,dest):
    text,y,_,_,_=f.inputs()
    weak,truth,cohort_hash=cohorts()
    rows=np.asarray(fit_ids(rotation))
    rows=rows[weak[rows]==0]
    evaluation=np.unique(np.r_[d.frame["ids"],b.frame["ids"]])
    if np.intersect1d(evaluation,rows).size:
        raise ValueError("Retrieval includes development or held-out rows")
    reserved={fingerprint(text[i]) for i in evaluation}
    before=len(rows)
    rows=np.asarray([i for i in rows if fingerprint(text[i]) not in reserved],dtype=np.int32)
    train=dest/"fit.bin"
    cmd=[str(OUT/"score"),str(train)]
    t=time.perf_counter()
    export(train,text,rows,y)
    paths=[]
    for role,ctx in (("dev",d),("test",b)):
        i=ctx.frame["ids"][ctx.active]
        source=dest/f"{role}.bin"
        result=dest/f"{role}.f32"
        export(source,text,i)
        cmd.extend((str(source),str(result)))
        paths.append(result)
    result=subprocess.run(cmd,check=True,text=True,capture_output=True,timeout=300)
    (dest/"timing.log").write_text(result.stdout+result.stderr)
    write_json(dest/"index.json",dict(training_rows=len(rows),copies_excluded=before-len(rows),
                     fit_sha256=checksum(train),implementation_sha256=checksum(CODE),
                     executable_sha256=checksum(OUT/"score"),cohort_sha256=cohort_hash,
                     benchmark_wall_seconds=time.perf_counter()-t))
    return (neighbors(paths[0],int(d.active.sum())),neighbors(paths[1],int(b.active.sum())))


def matrix(d,n,mode):
    xd=features(n,d.current[d.active])
    x=d.inputs[d.active]
    if mode=="control":
        return x
    if mode=="neighbors":
        return np.column_stack((x,xd))
    chars=np.log1p(d.lengths[d.active])/5
    return np.column_stack((x,xd,chars,xd[:,1:13]*chars[:,None]))


def evaluate(rotation):
    OUT.mkdir(exist_ok=True)
    if not (OUT/"score").exists():
        subprocess.run(["c++","-std=c++17","-O3",str(CODE),"-o",str(OUT/"score")],check=True)
    dest=OUT/rotation
    dest.mkdir(exist_ok=True)
    d,b,fit,cal,review=source_contexts(rotation)
    if b is None:
        b=d
    a,c=extract(rotation,d,b,dest)
    size=int(np.median(d.lengths[fit&d.active]))
    info={}
    models={}
    c2=(c[:,4:6].max(axis=1)>.6)
    coverage=dict(dev_at_06=int((a[:,4:6].max(axis=1)>.6).sum()),eval_at_06=int(c2.sum()),
                  dev_rows=len(a),eval_rows=len(c))
    for mode in ("control","neighbors","neighbors_length"):
        data=matrix(d,a,mode)
        survey=matrix(b,c,mode)
        clf=LogisticRegression(C=.2,solver="liblinear",max_iter=350,random_state=42)
        clf.fit(data[fit[d.active]],d.frame["y"][fit&d.active])
        p=clf.predict_proba(data)[:,1]
        q=clf.predict_proba(survey)[:,1]
        models[mode]=clf
        for scope,ratio in MODES:
            short_dev=d.lengths[d.active]<=size
            short_test=b.lengths[b.active]<=size
            mask=(short_dev if scope=="short" else ~short_dev if scope=="long" else np.ones(len(p),dtype=bool))
            choose=(short_test if scope=="short" else ~short_test if scope=="long" else np.ones(len(q),dtype=bool))
            ca=cal[d.active]&mask
            dd=(1-ratio)*d.current[d.active]+ratio*p
            qq=(1-ratio)*b.current[b.active]+ratio*q
            z=d.frame["y"][d.active][ca]
            initial=d.bits[d.active][ca]
            nfp=int((initial&(z==0)).sum())
            chosen,cut=f.strict_budget(dd[ca],z,nfp)
            trained=f.comparison(z,initial,chosen)
            active=b.active.copy()
            active[b.active]=choose
            w=b.bits.copy()
            w[active]=qq[choose]>cut
            held=np.ones(len(w),dtype=bool) if review is None else review
            outcome=f.comparison(b.frame["y"][held],b.bits[held],w[held])
            key=f"{mode}_{scope}_{ratio:g}"
            rec=dict(calibration=trained,review=outcome,threshold=cut,scope=scope,ratio=ratio)
            if "project" in b.frame:
                strict,per=match(b.frame["y"],b.frame["project"],b.bits,active,qq[choose])
                rec["same_fp"]=f.comparison(b.frame["y"],b.bits,strict)
                rec["per_project"]=per
            info[key]=rec
            np.savez_compressed(dest/f"{key}.npz",ids=b.frame["ids"][held],baseline=b.bits[held],
                               frozen=w[held],eligible=active[held],score=qq[choose][held[active]])
            print(rotation,key,trained["delta_tp"],rec.get("same_fp",outcome)["delta_tp"],outcome["delta_fp"],flush=True)
    joblib.dump(models,dest/"weights.joblib",compress=3)
    write_json(dest/"results.json",dict(length_cutoff=size,results=info,coverage=coverage,
                                      weights_sha256=checksum(dest/"weights.joblib")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    evaluate(p.parse_args().rotation)
