"""Sparse correction math used by both evaluation and the inference entry point."""
import numpy as np
from scipy.special import logit

from conditional import features
from sparse_model import aligned

REFERENCE="reference"


def clip_logit(x):
    return logit(x.clip(1e-5,1-1e-5))



def calibrate_sparse(oof,features):
    return np.column_stack([head.predict_proba(features[:,k,None])[:,1] for k,head in enumerate(oof)])



def combination_inputs(values,sparse):
    return [np.column_stack((features(values,regime),clip_logit(sparse[:,k])))
            for k,regime in enumerate(("negative","positive"))]



def combined_inputs(state,sparse):
    q=(1-state["w"])*sparse[:,0]+state["w"]*sparse[:,1]
    sure=state["known"]
    q[sure]=np.where(state["flag"][sure],sparse[sure,1],sparse[sure,0])
    odds=clip_logit(state["p"])
    return np.column_stack((odds,clip_logit(q),state["known"],state["known"]&state["flag"],state["w"]))



def proposals(frame,old,sparse,models,cut):
    learned,direction=models
    x=combination_inputs(frame[REFERENCE],sparse)
    no,yes=[p.predict_proba(a)[:,1] for p,a in zip(learned,x,strict=True)]
    ungated=old["p"]+(1-old["w"])*(no-old["no"])+old["w"]*(yes-old["yes"])
    known=old["known"]
    ungated[known]=np.where(old["flag"][known],yes[known],no[known])
    above=old["p"]>cut
    z=combined_inputs(old,sparse)
    directed=np.empty(len(above),dtype=np.float64)
    for mask,model in ((~above,direction[0]),(above,direction[1])):
        if mask.any():
            directed[mask]=model.predict_proba(z[mask])[:,1]
    return ungated,directed



def restrict(old,cut,scope):
    p=old["p"]
    if scope=="exact":
        return old["known"]
    if scope=="near":
        return (abs(p-cut)<.105)&(p>.025)&(p<.98)
    return (abs(p-cut)<.23)&(p>.015)&(p<.985)



def revise(old,suggestions,cut,name,scope,mix):
    original=old["p"]
    ungated,directed=suggestions
    proposed=directed if name=="two_gate" else ungated
    active=restrict(old,cut,scope)
    direction=original>cut
    if name=="veto":
        change=direction&(proposed<original)
    elif name=="promote":
        change=~direction&(proposed>original)
    elif name=="two_gate":
        change=(direction&(proposed<original))|(~direction&(proposed>original))
    else:
        change=np.ones(len(original),dtype=bool)
    value=np.clip(np.where(active&change,(1-mix)*original+mix*proposed,original),0,1)
    return value.astype(np.float32),active



def eligible(state,scope):
    if scope=="no":
        return state["known"]&~state["flag"]
    if scope=="yes":
        return state["known"]&state["flag"]
    return ~state["known"]&(state["w"]<.5)



def apply(state,primary,cut,direction,scope,mix):
    inside=eligible(state,scope)
    p=state["p"][inside]
    q=primary[inside].astype(np.float32)
    if direction=="veto":
        q=np.where(p>cut,np.minimum(q,p),p)
    elif direction=="promote":
        q=np.where(p<=cut,np.maximum(q,p),p)
    return inside,((1-mix)*p+mix*q).astype(np.float32)



def correction(texts,values,state,threshold,variant,package):
    score=state["p"].copy()
    bits=score>threshold
    cohort=variant["cohort_only"]
    selected=(eligible(state,variant["scope"]) if cohort
              else restrict(state,threshold,variant["scope"]))
    nnz=0
    if selected.any():
        names=[s for s,on in zip(texts,selected,strict=True) if on]
        logits,nnz=aligned(package["expert"],names,variant["mode"])
        sparse=calibrate_sparse(package["oof"],logits)
        part={k:x[selected] for k,x in state.items()}
        opts=proposals({REFERENCE:values[selected]},part,sparse,package["combiner"],threshold)
        if cohort:
            inside,q=apply(part,opts[0],threshold,variant["kind"],variant["scope"],variant["mix"])
            if not inside.all():
                raise ValueError("Conditional evaluation queried the wrong rows")
            bits[selected]=q>variant["threshold"]
            score[selected]=q
        else:
            score[selected],_=revise(part,opts,threshold,variant["kind"],variant["scope"],variant["mix"])
    if not cohort:
        bits=score>variant["threshold"]
    return bits,score,selected,nnz
