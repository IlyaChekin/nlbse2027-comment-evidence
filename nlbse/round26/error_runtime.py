"""Apply the signed sparse correction using the token matrix already computed."""
import time

import numpy as np

from conditional import features
from fuse_sparse import clip_logit,calibrate_sparse,proposals,apply as sparse_apply
from sparse_model import vectorize,predict
from section_runtime import compute

REFERENCE="reference"


def correct(texts,values,state,threshold,base_variant,package,section,section_head,option=None,extra=None):
    eligible=state["known"]&~state["flag"]
    active=np.flatnonzero(eligible)
    p=state["p"]
    predicts=p>threshold
    margin=clip_logit(p.astype(np.float64))-clip_logit(np.asarray(threshold))
    affected=np.zeros(len(p),dtype=bool)
    cost=dict(scoped_features=0,additional_features=0,section_calls=0,error_calls=0,error_seconds=0.)
    if not len(active):
        return predicts,margin,affected,cost
    local=[texts[i] for i in active]
    compact={k:v[eligible] for k,v in state.items()}
    x=vectorize(local,"token")
    sparse=calibrate_sparse(package["oof"],predict(package["expert"],x))
    recalib=proposals({REFERENCE:values[eligible]},compact,sparse,package["combiner"],threshold)[0]
    _,after=sparse_apply(compact,recalib,threshold,"blend","no",.5)
    cutoff=base_variant["threshold"]
    bits=after>cutoff
    side=features(values[eligible],"negative")
    old=clip_logit(sparse[:,0])
    section_rows=np.abs(after-cutoff)<section["distance"]
    ctime=0
    if section_rows.any():
        extra_values,cnt=compute([s for s,b in zip(local,section_rows,strict=True) if b],
                                 section["region"],package["expert"][0])
        inp=np.column_stack((side[section_rows],old[section_rows],clip_logit(after[section_rows]),extra_values))
        w=section["damping"]
        corrected=(1.-w)*after[section_rows]+w*section_head.predict_proba(inp)[:,1]
        bits[section_rows]=corrected>section["threshold"]
        ctime=cnt
    m=clip_logit(after.astype(np.float64))-clip_logit(np.asarray(np.float32(cutoff),dtype=np.float64))
    if section_rows.any():
        m[section_rows]=clip_logit(corrected)-clip_logit(np.asarray(section["threshold"]))
    margin[eligible]=m
    predicts[eligible]=bits
    cost.update(scoped_features=4*(x.nnz+ctime),section_calls=int(section_rows.sum()))
    if option is None:
        return predicts,margin,affected,cost
    start=time.perf_counter()
    local_mask=np.abs(m)<option["width"]
    diffs=np.zeros(len(m))
    hits=0
    for above,model,head in zip((False,True),extra["experts"],extra["readout"],strict=True):
        rows=local_mask&(bits==above)
        if ((option["scope"]=="veto" and not above) or (option["scope"]=="promote" and above) or not rows.any()):
            continue
        sample=x[rows]
        logits=model.decision_function(sample)
        probabilities=head.predict_proba(np.column_stack((m[rows],side[rows,0],old[rows],logits)))[:,1]
        difference=clip_logit(probabilities)-extra["anchors"][above]
        diffs[rows]=np.maximum(difference,0.)*(-1 if above else 1)
        cost["additional_features"]+=2*sample.nnz
        hits+=sample.shape[0]
    amount=m[local_mask]+option["weight"]*diffs[local_mask]
    predicts[active[local_mask]]=amount>option["cutoff"]
    margin[active[local_mask]]=amount
    affected[active[local_mask]]=True
    cost.update(error_calls=hits,error_seconds=time.perf_counter()-start)
    return predicts,margin,affected,cost
