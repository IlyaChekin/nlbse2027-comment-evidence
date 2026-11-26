"""Reuse the existing exact-negative token vectors for the recalibrated readout."""
import time

import numpy as np

from conditional import features
from fuse_sparse import clip_logit, calibrate_sparse, proposals, apply as sparse_apply
from section_runtime import compute
from sparse_model import vectorize, predict


def rerank(texts, values, state, threshold, base_variant, package, section, head, option=None, models=None, margin_out=None):
    p=state['p']
    bits=p>threshold
    mask=state['known'] & ~state['flag']
    touched=np.zeros(len(bits),dtype=bool)
    pending=np.empty(0)
    if not mask.any():
        return bits,touched,pending,dict(base_ops=0,extra_ops=0,added=0.)
    chosen=np.flatnonzero(mask)
    local=[texts[i] for i in chosen]
    old={k:v[mask] for k,v in state.items()}
    x=vectorize(local,'token')
    sparse=calibrate_sparse(package['oof'],predict(package['expert'],x))
    mix=proposals({'reference':values[mask]},old,sparse,package['combiner'],threshold)[0]
    _, after=sparse_apply(old,mix,threshold,'blend','no',.5)
    cutoff=base_variant['threshold']
    bits[chosen]=after>cutoff
    scoped=abs(after-cutoff)<section['distance']
    work=x.nnz
    if scoped.any():
        candidate=chosen[scoped]
        extra, count=compute([texts[i] for i in candidate],section['region'],package['expert'][0])
        work+=count
        a=np.column_stack((features(values[candidate],'negative'),clip_logit(sparse[scoped,0]),
                           clip_logit(after[scoped]),extra))
        strength=section['damping']
        q=(1.-strength)*after[scoped]+strength*head.predict_proba(a)[:,1]
        bits[candidate]=q>section['threshold']
    if option is None:
        return bits,touched,pending,dict(base_ops=4*work,extra_ops=0,added=0.)
    start=time.perf_counter()
    margin=clip_logit(after.astype(np.float64))-clip_logit(np.asarray(np.float32(cutoff),dtype=np.float64))
    if scoped.any():
        margin[scoped]=clip_logit(q)-clip_logit(np.asarray(section['threshold']))
    if not np.array_equal(margin>0,bits[chosen]):
        raise ValueError('The calibrated margin does not reproduce the frozen decision')
    branch=np.ones(len(margin),dtype=bool) if option['width'] is None else abs(margin)<option['width']
    target=chosen[branch]
    touched[target]=True
    operations=0
    if option['kind']=='threshold':
        pending=margin[branch]
        bits[target]=pending>option['cut']
        recover=option.get('recover')
        if recover is not None:
            rec=margin-option['cut']
            selected=abs(rec)<recover['width']
            rows=chosen[selected]
            touched[:]=False
            touched[rows]=True
            env=np.column_stack((margin[selected],rec[selected],features(values[rows],'negative'),
                                 clip_logit(sparse[selected,0])))
            estimate=models.decision_function(env) if len(rows) else np.empty(0)
            w=recover['mix']
            pending=(1.-w)*rec[selected]+w*estimate
            if recover['direction']=='recover':
                pending=np.where(bits[rows],rec[selected],np.maximum(rec[selected],pending))
            bits[rows]=pending>recover['cut']
            if margin_out is not None:
                margin_out[chosen]=rec
                margin_out[rows]=pending-recover['cut']
            operations=2*env.size
        return bits,touched,pending,dict(base_ops=4*work,extra_ops=operations,added=time.perf_counter()-start)
    if len(target):
        if option['kind']=='canonical':
            from normalize_code import batch as canonicalize
            matrix=vectorize(canonicalize([texts[i] for i in target]),'token')
        else:
            matrix=x[branch]
        models_r=models['rankers']
        a=np.column_stack((margin[branch],*[m.decision_function(matrix) for m in models_r]))
        extension=models['readout'].decision_function(a)
        w=option['mix']
        pending=(1.-w)*margin[branch]+w*extension
        bits[target]=pending>option['cut']
        operations=2*matrix.nnz*len(models_r)+2*a.size
    return bits,touched,pending,dict(base_ops=4*work,extra_ops=operations,added=time.perf_counter()-start)

def student_bits(bits, margin, logits, aux, model, option):
    active=np.isfinite(margin)
    if option['width'] is not None:
        active&=abs(margin)<option['width']
    current=margin[active]
    new=logits[active].astype(np.float64)
    delta=new-clip_logit(aux[active,20].astype(np.float64))
    proposed=current+delta if model is None else model.decision_function(np.column_stack((current,new,delta)))
    w=option['mix']
    rank=(1.-w)*current+w*proposed
    changed=bits.copy()
    changed[active]=rank>option['cut']
    return changed,active,rank
