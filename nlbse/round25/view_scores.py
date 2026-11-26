"""Compare sparse scores of code sections inside the scanner's negative cohort."""
import argparse
from itertools import product
from pathlib import Path
import re
from statistics import median
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from length_eval import HERE,ROOT,R24,PARTS,f,source_contexts,match,checksum,json_at,write_json
from sparse_model import WORDS,TOKEN
from comment_spans import LEXEME

OUT=HERE/"views"
SYMBOL=re.compile(TOKEN)
BRACE=re.compile(r"\{")
MIN_BODY=16
LOGITS=("regions","segments")


def opening(text):
    lo=0
    for span in LEXEME.finditer(text):
        mark=BRACE.search(text,lo,span.start())
        if mark:
            return mark.start()
        lo=span.end()
    tail=BRACE.search(text,lo)
    return tail.start() if tail else 0


def views(text,kind):
    if kind=="regions":
        pos=opening(text)
        return text[:pos],text[pos:]
    spans=list(SYMBOL.finditer(text))
    n=len(spans)
    if n<MIN_BODY:
        return text,text
    return text[:spans[(3*n)//5-1].end()],text[spans[(2*n)//5].start():]


def compute(text,kind,sparse):
    left,right=zip(*(views(x,kind) for x in text),strict=True)
    x0=WORDS.transform(left)
    x1=WORDS.transform(right)
    a=sparse.decision_function(x0)
    b=sparse.decision_function(x1)
    return np.column_stack((a,b,np.maximum(a,b),np.abs(a-b))),(x0.nnz+x1.nnz)


def advance(context,mask,extra):
    return np.column_stack((context.inputs[mask],extra))


def stage(rotation):
    out=OUT/rotation
    out.mkdir(parents=True,exist_ok=True)
    d,b,fit,cal,review=source_contexts(rotation)
    if b is None:
        b=d
    a_mask,b_mask=(z.active for z in (d,b))
    a_text=[s for s,k in zip(d.texts,a_mask,strict=True) if k]
    b_text=[s for s,k in zip(b.texts,b_mask,strict=True) if k]
    sparse_path=R24/rotation/"token.joblib"
    record=json_at(R24/rotation/"fit.json")["models"]["token"]
    if checksum(sparse_path)!=record["weights_sha256"]:
        raise ValueError("Round 24 token expert changed")
    sparse=joblib.load(sparse_path)[0]
    trained=fit[a_mask]
    truth=d.frame["y"][a_mask]
    sizes=d.lengths[a_mask]
    cut_size=int(np.median(sizes[trained]))
    variants={}
    models={}
    for name in LOGITS:
        t=time.perf_counter()
        ax,_=compute(a_text,name,sparse)
        bx=ax if b is d else compute(b_text,name,sparse)[0]
        xa=advance(d,a_mask,ax)
        xb=advance(b,b_mask,bx)
        h=LogisticRegression(C=.3,max_iter=350,solver="liblinear",random_state=42).fit(xa[trained],truth[trained])
        models[name]=h
        p=h.predict_proba(xa)[:,1]
        q=h.predict_proba(xb)[:,1]
        elapsed=time.perf_counter()-t
        print(rotation,name,"features",len(ax),len(bx),round(elapsed,3),flush=True)
        for branch,damping,distance in product(("all","short","long"),(.4,.8),(None,.16,.075,.035)):
            da=np.ones(len(p),dtype=bool)
            db=np.ones(len(q),dtype=bool)
            if branch!="all":
                da &= (sizes<=cut_size)==(branch=="short")
                db &= (b.lengths[b_mask]<=cut_size)==(branch=="short")
            if distance is not None:
                da &= (abs(d.current[a_mask]-d.cutoff)<distance)
                db &= (abs(b.current[b_mask]-b.cutoff)<distance)
            know=cal[a_mask]&da
            dscore=(1-damping)*d.current[a_mask]+damping*p
            bscore=(1-damping)*b.current[b_mask]+damping*q
            assigned=d.bits[a_mask][know]
            y=truth[know]
            fp=int((assigned&(y==0)).sum())
            at,threshold=f.strict_budget(dscore[know],y,fp)
            dev=f.comparison(y,assigned,at)
            all_dev=d.bits.copy()
            within=a_mask.copy()
            within[a_mask]=da
            all_dev[within]=dscore[da]>threshold
            full=f.comparison(d.frame["y"][cal],d.bits[cal],all_dev[cal])
            checked=b_mask.copy()
            checked[b_mask]=db
            pred=b.bits.copy()
            pred[checked]=bscore[db]>threshold
            evalrows=np.ones(len(pred),dtype=bool) if review is None else review
            fixed=f.comparison(b.frame["y"][evalrows],b.bits[evalrows],pred[evalrows])
            key=f"{name}_{branch}_{damping:g}_{'all' if distance is None else f'margin{distance:g}'}"
            rec=dict(cal=dev,cal_full=full,frozen=fixed,threshold=threshold,rows=int((evalrows&checked).sum()),cal_rows=int(know.sum()),
                     region=name,branch=branch,damping=damping,distance=distance)
            if b is not d:
                paired,per=match(b.frame["y"],b.frame["project"],b.bits,checked,bscore[db])
                rec["matched"]=f.comparison(b.frame["y"],b.bits,paired)
                rec["per_project"]=per
            else:
                # Measure only this fixed subset, so unused views do not contribute to the cost.
                text=[s for s,k in zip(a_text,know,strict=True) if k]
                times=[]
                nnz=0
                if text:
                    compute(text,name,sparse)
                    for _ in range(3):
                        begin=time.perf_counter()
                        sampled,nnz=compute(text,name,sparse)
                        h.predict_proba(np.column_stack((d.inputs[d.active][know],sampled)))
                        times.append(time.perf_counter()-begin)
                if not times:
                    times=[0.]
                impact=.3*(full["f1_after"]-full["f1_before"])
                rec["cost"]=dict(sec=median(times),trials=times,nnz=nnz)
                rec["proxy"]=float(impact-.2*f.PUBLIC_N/cal.sum()*(median(times)/1000+4*nnz/3e15))
            variants[key]=rec
            np.savez_compressed(out/f"{key}.npz",ids=b.frame["ids"][evalrows],baseline=b.bits[evalrows],fixed=pred[evalrows],
                                 checked=checked[evalrows],score=bscore[db][evalrows[checked]])
            print(rotation,key,dev["delta_tp"],rec.get("matched",fixed)["delta_tp"],fixed["delta_fp"],
                   rec.get("proxy",0.),flush=True)
    artifact=out/"calibrator.joblib"
    joblib.dump(models,artifact,compress=3)
    write_json(out/"results.json",dict(results=variants,length_cutoff=cut_size,
               baseline_cutoff=b.cutoff,cal_rows=int(cal.sum()),source_expert_sha256=checksum(sparse_path),
               code_sha256=checksum(Path(__file__)),model_sha256=checksum(artifact)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    stage(p.parse_args().rotation)
