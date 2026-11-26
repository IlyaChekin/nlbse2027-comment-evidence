"""Score the two code regions for the selected correction."""
import re
import time
import numpy as np

from conditional import features
from fuse_sparse import aligned, calibrate_sparse, clip_logit, proposals, apply as sparse_apply
from sparse_model import WORDS, TOKEN
from comment_spans import LEXEME

REFERENCE="reference"
SYMBOL=re.compile(TOKEN)
BRACE=re.compile(r"\{")
MIN_BODY=16


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


def adjust(texts,values,state,threshold,base_variant,package,option=None,length_cut=None,head=None):
    active=state["known"]&~state["flag"]
    p=state["p"].copy()
    original=p>threshold
    touched=np.zeros(len(p),dtype=bool)
    nnz=0
    if not active.any():
        return original,p,touched,nnz,0.
    local=[s for s,on in zip(texts,active,strict=True) if on]
    compact={k:v[active] for k,v in state.items()}
    raw,nnz=aligned(package["expert"],local,"token")
    sparse=calibrate_sparse(package["oof"],raw)
    recalib=proposals({REFERENCE:values[active]},compact,sparse,package["combiner"],threshold)[0]
    _,score=sparse_apply(compact,recalib,threshold,"blend","no",.5)
    cutoff=base_variant["threshold"]
    new=score>cutoff
    p[active]=score
    original[active]=new
    start=time.perf_counter()
    if option:
        chose=np.ones(len(local),dtype=bool)
        branch=option["branch"]
        if branch!="all":
            length=np.fromiter((len(s) for s in local),dtype=np.int32)
            chose &= (length<=length_cut)==(branch=="short")
        if option["distance"] is not None:
            chose &= abs(score-cutoff)<option["distance"]
        if chose.any():
            extra,total=compute([s for s,on in zip(local,chose,strict=True) if on],
                                 option["region"],package["expert"][0])
            x=np.column_stack((features(values[active][chose],"negative"),clip_logit(sparse[chose,0]),
                                clip_logit(score[chose]),extra))
            w=option["damping"]
            new_score=(1.-w)*score[chose]+w*head.predict_proba(x)[:,1]
            original[np.flatnonzero(active)[chose]]=new_score>option["threshold"]
            p[np.flatnonzero(active)[chose]]=new_score
            nnz+=total
        touched[active]=chose
    return original,p,touched,nnz,time.perf_counter()-start
