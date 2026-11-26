"""Test whether syntax summaries improve out-of-file error ranking in the negative cohort."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

from disagreement import (HERE,ROOT,PARTS,f,checksum,json_at,write_json,load_sources,estimate_training,
                          encoded_context,match,clip_logit,trained_logistic,LIMIT)

sys.path.insert(0,str(ROOT/"round23"))
from branches import batch

OUT=HERE/"shape_margin"
SHAPE=ROOT/"round23/features"
LEADS=("control","shape")


def pack(chance,dense,view):
    p=clip_logit(chance)
    core=np.column_stack((p,np.abs(p[:,0]-p[:,1])))
    if view=="control":
        return core
    return np.column_stack((core,dense))


def data_for(rotation):
    receipt=json_at(SHAPE/"manifest.json")
    path=SHAPE/"rows.npz"
    if checksum(path)!=receipt["files"][path.name]:
        raise ValueError("Syntax summary changed")
    with np.load(path) as z:
        dense=z["dense"]
        y=z["y"]
        if not np.array_equal(z["ids"],np.arange(len(dense))):
            raise ValueError("Syntax row order changed")
    ids,target,pc,ps,fold,hashes=load_sources(rotation)
    if not np.array_equal(y[ids],target):
        raise ValueError("Out-of-file targets do not align")
    chance=estimate_training(target,pc,ps,fold)
    return dense,ids,target,chance,hashes


def base_features(ctx):
    return np.column_stack((ctx.state["no"],expit(ctx.inputs[:,2]),ctx.current))


def run(rotation):
    d,b,fit,cal,review,old=encoded_context(rotation)
    if b is None:
        b=d
    baseline,source=old[0]
    control,target=old[-1]
    keep=np.ones(len(control),dtype=bool) if review is None else review
    dense,ids,y,chance,hashes=data_for(rotation)
    root=OUT/rotation
    root.mkdir(parents=True,exist_ok=True)
    results={}
    saved={}
    for model in LEADS:
        xx=pack(chance,dense[ids],model)
        samples=[pack(base_features(c),dense[c.frame["ids"]],model) for c in (d,b)]
        decision=chance[:,2]>=LIMIT
        raw=[]
        for above in (False,True):
            focus=decision==above
            clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,
                  learning_rate=.075,l2_regularization=2.5,random_state=42).fit(
                      xx[focus],1-y[focus] if above else y[focus])
            a,bx=[clf.predict_proba(m)[:,1] for m in samples]
            read=fit&d.active&(baseline==above)
            balance=1-d.frame["y"] if above else d.frame["y"]
            hx=np.column_stack((source,clip_logit(a)))
            adjust=trained_logistic(hx[read],balance[read],c=.20)
            p=adjust.predict_proba(hx)[:,1]
            q=adjust.predict_proba(np.column_stack((target,clip_logit(bx))))[:,1]
            select=read&(np.abs(source)<.4)
            anchor=float(np.median(clip_logit(p[select])))
            raw.append((p,q,anchor,clf,adjust))
        saved[model]=[n[2:] for n in raw]
        for lane,w,width in product(("promote","veto","both"),(.35,.70),(.16,.40,1.)):
            task=[]
            for k,(c,pre,margin) in enumerate(((d,baseline,source),(b,control,target))):
                active=c.active&(np.abs(margin)<width)
                score=margin.copy()
                for above,features in zip((False,True),raw,strict=True):
                    on=active&(pre==above)
                    if (lane=="veto" and not above) or (lane=="promote" and above):
                        continue
                    q=clip_logit(features[k])-features[2]
                    score[on]+=(1 if not above else -1)*w*np.maximum(q[on],0.)
                task.append((active,score))
            m,train=task[0]
            im,values=task[1]
            label=d.frame["y"][m&cal]
            base=baseline[m&cal]
            fp=int((base&(label==0)).sum())
            bits,cut=f.strict_budget(train[m&cal],label,fp)
            changed=baseline.copy()
            changed[m]=train[m]>cut
            dev=f.comparison(d.frame["y"][cal],baseline[cal],changed[cal])
            predicted=control.copy()
            predicted[im]=values[im]>cut
            fixed=f.comparison(b.frame["y"][keep],control[keep],predicted[keep])
            key=f"{model}_{lane}_{w:g}_{width:g}"
            row=dict(cal=dev,frozen=fixed,threshold=cut,weight=w,band=width,lane=lane,view=model)
            if review is None:
                match_fp,prj=match(b.frame["y"],b.frame["project"],control,im,values[im])
                row["matched"]=f.comparison(b.frame["y"],control,match_fp)
                row["project"]=prj
            else:
                # Parsing time alone is a lower bound on additional processing.
                text=[s for s,k in zip(d.texts,m&cal,strict=True) if k]
                measured=[]
                for _ in range(3):
                    tick=time.perf_counter()
                    if model=="shape" and text:
                        batch(text)
                    measured.append(time.perf_counter()-tick)
                gain=.3*(dev["f1_after"]-dev["f1_before"])
                row["proxy"]=float(gain-.2*f.PUBLIC_N/cal.sum()*median(measured)/1000)
                row["extra_shape_seconds"]=median(measured)
            results[key]=row
            np.savez_compressed(root/f"{key}.npz",ids=b.frame["ids"][keep],baseline=control[keep],
                frozen=predicted[keep],active=im[keep],scores=values[keep])
    pack_file=root/"selected.joblib"
    joblib.dump(saved,pack_file,compress=3)
    write_json(root/"results.json",dict(options=results,source_hashes=hashes,
                    fit_rows=len(ids),weights_sha256=checksum(pack_file)))
    p=sorted([(v.get("matched",v["frozen"])["delta_tp"],k,v["cal"]["delta_tp"],
               v["frozen"]["delta_fp"]) for k,v in results.items()],reverse=True)
    print(rotation,p[:10],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    run(p.parse_args().rotation)
