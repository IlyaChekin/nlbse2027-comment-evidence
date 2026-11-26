"""Train separate sparse experts for likely missed positives and false alarms."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np
from scipy.sparse import load_npz,vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from compact_oof import HERE,ROOT,R24,PARTS,f,checksum,json_at,write_json,signed_source
from length_eval import source_contexts,match
from view_scores import compute
from fuse_sparse import clip_logit

OUT=HERE/"disagreement"
R25=ROOT/"round25"
LIMIT=.18
STRENGTHS=(.35,.70)
MARGINS=(.16,.40,1.0)


def load_sources(rotation):
    sparse,signature=signed_source(rotation)
    if rotation=="official":
        folder=HERE/"compact/official"
        path=folder/"scores.npz"
        frozen=json_at(folder/"manifest.json")
        if frozen["sparse_oof_sha256"]!=signature or checksum(path)!=frozen["sha256"]:
            raise ValueError("Independent compact scores changed")
        with np.load(path) as z:
            if not np.array_equal(z["ids"],sparse["ids"]) or not np.array_equal(z["y"],sparse["y"]):
                raise ValueError("Compact and sparse predictions disagree on row order")
            selected=sparse["weak"]==0
            rows=z["ids"][selected]
            small=z["values"][selected,2]
            neural_fold=z["fold"][selected]
            position=np.flatnonzero(selected)
    else:
        folder=ROOT/"round15/mining"
        path=folder/"scores.npz"
        if checksum(path)!=json_at(folder/"manifest.json")["sha256"]:
            raise ValueError("Independent compact scores changed")
        with np.load(path) as z:
            rows,i,j=np.intersect1d(sparse["ids"],z["ids"],return_indices=True)
            keep=sparse["weak"][i]==0
            rows,position,i0=rows[keep],i[keep],j[keep]
            small=z["compact"][i0,0]
            neural_fold=z["fold"][i0]
            if not np.array_equal(z["y"][i0,0],sparse["y"][position]):
                raise ValueError("Matched independent rows disagree on labels")
    return rows,sparse["y"][position],small.astype(np.float64),sparse["logits"][position,0],neural_fold,{
        "token_oof":signature,"compact_oof":checksum(path)}


def trained_logistic(x,y,weight=None,c=.3):
    return LogisticRegression(C=c,solver="liblinear",max_iter=200,random_state=42).fit(x,y,sample_weight=weight)


def estimate_training(y,small,sparse,fold):
    x=np.column_stack((clip_logit(small),sparse))
    pred=np.empty((len(y),3))
    for k in np.unique(fold):
        keep=fold!=k
        held=~keep
        for j,shape in enumerate((x[:,0,None],x[:,1,None],x)):
            p=trained_logistic(shape[keep],y[keep],c=.8)
            pred[held,j]=p.predict_proba(shape[held])[:,1]
    return pred


def paired_classifier(x,rows,errors,proxy,length,difficult):
    # Match code size and score before comparing a mistake with a correct decision.
    bucket=np.clip(np.floor(np.log2(length)/1.).astype(int),0,16)
    l=[]
    r=[]
    weights=[]
    for level in np.unique(bucket):
        wins=np.flatnonzero((bucket==level)&(errors==1))
        loss=np.flatnonzero((bucket==level)&(errors==0))
        if not len(wins) or not len(loss):
            continue
        loss=loss[np.argsort(proxy[loss],kind="stable")]
        rank=np.searchsorted(proxy[loss],proxy[wins])
        for off in (-3,-1,0,2):
            near=np.clip(rank+off,0,len(loss)-1)
            l.extend(rows[wins])
            r.extend(rows[loss[near]])
            weights.extend(.25+2*difficult[wins])
    if not l:
        raise ValueError("Could not form any matched outcome pairs")
    a=np.asarray(l)
    b=np.asarray(r)
    d=x[a]-x[b]
    train=vstack((d,-d),format="csr")
    labels=np.r_[np.ones(len(l),dtype=np.int8),np.zeros(len(l),dtype=np.int8)]
    return LogisticRegression(C=.45,solver="liblinear",fit_intercept=False,max_iter=150,
                  random_state=42).fit(train,labels,sample_weight=np.tile(weights,2))


def train_experts(x,rotation,styles):
    rows,y,small,sparse,fold,hashes=load_sources(rotation)
    a,b,proxy=estimate_training(y,small,sparse,fold).T
    disagreement=np.abs(a-b)
    upper=proxy>=LIMIT
    difficult=np.exp(-np.abs(clip_logit(proxy)-clip_logit(np.asarray(LIMIT))))
    models={}
    corpus,_,_,_,_=f.inputs()
    length=np.array([max(len(corpus[i]),1) for i in rows])
    for name in styles:
        heads=[]
        for above in (False,True):
            choose=np.ones(len(y),dtype=bool) if name=="pooled" else upper==above
            weight=np.ones(len(y)) if name!="focused" else .30+3*difficult+4*disagreement
            yy=y if not above else 1-y
            if name=="pairwise":
                heads.append(paired_classifier(x,rows[choose],yy[choose],proxy[choose],length[choose],difficult[choose]))
            else:
                heads.append(trained_logistic(x[rows[choose]],yy[choose],weight[choose],c=1.0))
        models[name]=heads
    summary=dict(rows=len(rows),positive=int(y.sum()),positive_proxy=int(upper.sum()),
                 false_alarm=int((upper&(y==0)).sum()),miss=int((~upper&(y==1)).sum()),
                 strong_disagreements=int((disagreement>=.10).sum()),threshold=LIMIT,
                 source_hashes=hashes,proxy_ap=float(average_precision_score(y,proxy)))
    return models,summary,(rows,y,small,sparse,proxy,disagreement,upper,fold)


def encoded_context(rotation):
    d,b,fit,cal,review=source_contexts(rotation)
    actual=R25/"views/public/policy.json"
    winner=json_at(actual)["selected"]
    base=R25/"views"/rotation
    v=json_at(base/"results.json")
    h=joblib.load(base/"calibrator.joblib")
    if checksum(base/"calibrator.joblib")!=v["model_sha256"]:
        raise ValueError("Section calibrator changed")
    variant=v["results"][winner]
    path=R24/rotation/"token.joblib"
    if checksum(path)!=json_at(R24/rotation/"fit.json")["models"]["token"]["weights_sha256"]:
        raise ValueError("Token expert changed")
    sparse=joblib.load(path)[0]
    def update(ctx):
        active=ctx.active&(np.abs(ctx.current-ctx.cutoff)<variant["distance"])
        scores=ctx.current.copy()
        base=(scores>ctx.cutoff)
        margin=clip_logit(scores.astype(np.float64))-clip_logit(np.asarray(np.float32(ctx.cutoff),dtype=np.float64))
        if active.any():
            # Match the original calibration's matrix shape and tie behavior at its strict cutoff.
            r,_=compute([s for s,k in zip(ctx.texts,ctx.active,strict=True) if k],variant["region"],sparse)
            t=h[variant["region"]].predict_proba(np.column_stack((ctx.inputs[ctx.active],r)))[:,1]
            w=variant["damping"]
            blended=((1-w)*ctx.current[ctx.active]+w*t)[active[ctx.active]]
            base[active]=blended>variant["threshold"]
            margin[active]=clip_logit(blended)-clip_logit(np.asarray(variant["threshold"]))
        result=ctx.bits.copy()
        result[active]=base[active]
        if not np.array_equal(margin[ctx.active]>0,result[ctx.active]):
            raise ValueError("Centered margins do not exactly represent the comparator decisions")
        return result,margin
    old=[update(d)]
    if b is not None:
        old.append(update(b))
    checked=base/f"{winner}.npz"
    with np.load(checked) as z:
        baseline=old[-1][0] if b is not None else old[0][0][review]
        ids=b.frame["ids"] if b is not None else d.frame["ids"][review]
        if not np.array_equal(z["fixed"],baseline) or not np.array_equal(z["ids"],ids):
            raise ValueError("The measured section baseline changed")
    if b is None:
        trial=f.comparison(d.frame["y"][cal],d.bits[cal],old[0][0][cal])
        if trial!=variant["cal_full"]:
            raise ValueError("The measured section calibration changed")
    return d,b,fit,cal,review,old


def feature_scope(c,margins,models,matrix):
    ids=c.frame["ids"][c.active]
    small=margins[c.active]
    scores=[h.decision_function(matrix[ids]) for h in models]
    base=c.inputs[c.active]
    return [np.column_stack((small,base[:,0],base[:,2],score)) for score in scores]


def learn_readout(d,dec,dinputs,fit):
    local=d.frame["y"][d.active]
    phase=dec[d.active]
    ready=fit[d.active]
    heads=[]
    for above in (False,True):
        group=ready&(phase==above)
        errors=(1-local) if above else local
        heads.append(trained_logistic(dinputs[above][group],errors[group]))
    return heads


def choose_adjust(c,base,margin,x,heads,anchor,scope,weight,band):
    local=c.active.copy()
    local &= abs(margin)<band
    active=local[c.active]
    positive=base[c.active]
    mid=margin[c.active]
    change=np.zeros(len(mid))
    for above,h in zip((False,True),heads,strict=True):
        chosen=active&(positive==above)
        if scope=="veto" and not above or scope=="promote" and above or not chosen.any():
            continue
        q=h.predict_proba(x[above][chosen])[:,1]
        correction=clip_logit(q)-anchor[above]
        change[chosen]=np.maximum(correction,0.)*(-1 if above else 1)
    return local,(mid+weight*change)[active]


def evaluate(rotation,pairwise=False):
    begin=time.perf_counter()
    cache=R24/"features"
    if checksum(cache/"matrix.npz")!=json_at(cache/"manifest.json")["matrix_sha256"]:
        raise ValueError("Sparse training features changed")
    matrix=load_npz(cache/"matrix.npz")[:,:2**17].tocsr()
    fitted,train,data=train_experts(matrix,rotation,
            ("pairwise",) if pairwise else ("pooled","direction","focused"))
    d,b,fit,cal,review,old=encoded_context(rotation)
    if b is None:
        b=d
    base,margin=old[0]
    reference,oldmargin=old[-1]
    results={}
    recipes={}
    check={}
    for width in MARGINS:
        scope=cal&d.active&(abs(margin)<width)
        y=d.frame["y"][scope]
        bits=base[scope]
        budget=int((bits&(y==0)).sum())
        control,_=f.strict_budget(margin[scope],y,budget)
        if not np.array_equal(bits,control):
            raise ValueError("Uncorrected calibration does not reproduce the original budget")
        check[str(width)]=dict(rows=int(len(y)),fp=budget,tp=int((bits&(y==1)).sum()))
    out=(HERE/"ranking" if pairwise else OUT)/rotation
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"training.npz",**dict(zip(
        ("rows","truth","compact","sparse","proxy","disagreement","positive","folds"),data,strict=True)))
    for name,models in fitted.items():
        xx=feature_scope(d,margin,models,matrix)
        yy=xx if b is d else feature_scope(b,oldmargin,models,matrix)
        h=learn_readout(d,base,xx,fit)
        select=fit[d.active]
        values=[]
        for above,head in zip((False,True),h,strict=True):
            use=select&(base[d.active]==above)&(abs(margin[d.active])<.4)
            probs=head.predict_proba(xx[above][use])[:,1]
            values.append(float(np.median(clip_logit(probs))))
        recipes[name]=dict(experts=models,readout=h,anchors=values)
        for scope,weight,width in product(("veto","promote","both"),STRENGTHS,MARGINS):
            cand,p=choose_adjust(d,base,margin,xx,h,values,scope,weight,width)
            target=d.frame["y"][cal&cand]
            previous=base[cal&cand]
            n=int((previous&(target==0)).sum())
            _,cut=f.strict_budget(p[cal[cand]],target,n)
            after=base.copy()
            after[cand]=p>cut
            cal_metric=f.comparison(d.frame["y"][cal],base[cal],after[cal])
            marked,q=choose_adjust(b,reference,oldmargin,yy,h,values,scope,weight,width)
            pred=reference.copy()
            pred[marked]=q>cut
            evalmask=np.ones(len(pred),dtype=bool) if review is None else review
            frozen=f.comparison(b.frame["y"][evalmask],reference[evalmask],pred[evalmask])
            key=f"{name}_{scope}_{weight:g}_{width:g}"
            entry=dict(cal=cal_metric,frozen=frozen,cutoff=cut,scope=scope,weight=weight,
                       width=width,mode=name,cal_rows=int((cal&cand).sum()),eval_rows=int((evalmask&marked).sum()))
            if review is None:
                same,each=match(b.frame["y"],b.frame["project"],reference,marked,q)
                entry["matched"]=f.comparison(b.frame["y"],reference,same)
                entry["per_project"]=each
            else:
                # The existing sparse correction already paid for the token matrix.
                # Count a second dot product and the readout only on this policy's candidates.
                passes=[]
                dots=0
                for _ in range(4):
                    start=time.perf_counter()
                    dots=0
                    for above,expert,readout in zip((False,True),models,h,strict=True):
                        active=cal&cand&(base==above)
                        if (scope=="veto" and not above) or (scope=="promote" and above) or not active.any():
                            continue
                        token=matrix[d.frame["ids"][active]]
                        ql=expert.decision_function(token)
                        readout.predict_proba(np.column_stack((margin[active],d.inputs[active,0],d.inputs[active,2],ql)))
                        dots+=token.nnz
                    passes.append(time.perf_counter()-start)
                cost=median(passes)
                grade=.3*(cal_metric["f1_after"]-cal_metric["f1_before"])
                extra=.2*f.PUBLIC_N/cal.sum()*(cost/1000+2*dots/3e15)
                entry.update(proxy=float(grade-extra),times=passes,sparse_nnz=dots,extra_seconds=cost)
            results[key]=entry
            np.savez_compressed(out/f"{key}.npz",ids=b.frame["ids"][evalmask],baseline=reference[evalmask],
                frozen=pred[evalmask],mask=marked[evalmask],score=q[evalmask[marked]])
    packed=out/"models.joblib"
    joblib.dump(recipes,packed,compress=3)
    write_json(out/"results.json",dict(options=results,training=train,replay_control=check,weights_sha256=checksum(packed),
                    oof_sha256=checksum(out/"training.npz"),elapsed_seconds=time.perf_counter()-begin))
    print(rotation,train,flush=True)
    print(sorted([(r.get("matched",r["frozen"])["delta_tp"],n,r["cal"]["delta_tp"]) for n,r in results.items()],
                 reverse=True)[:10],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    p.add_argument("--pairwise",action="store_true")
    a=p.parse_args()
    evaluate(a.rotation,a.pairwise)
