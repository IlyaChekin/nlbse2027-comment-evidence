"""Use the preserved compact representation and the frozen pooled projection."""
import numpy as np
import torch

from conditional import query_value
from predict_factor import full_context, lexical_features, ordered_batches, batch_plan, batch, small
from rule_model import pooling


@torch.inference_mode()
def encode(model,tokenizer,texts,projection):
    raw=np.empty((len(texts),2),np.float32)
    aux=np.empty((len(texts),22),np.float32)
    result=np.empty((len(texts),2),np.float32)
    def prepare(offset):
        local=texts[offset:offset+2048]
        return full_context(local,tokenizer),lexical_features(local)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset,(seq,side) in ordered_batches(range(0,len(texts),2048),prepare,True):
            raw[offset:offset+len(seq),1]=side[:,:4].any(axis=1)
            lexical=torch.as_tensor(side,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                ids,mask=batch(group,seq,tokenizer.pad_token_id)
                states=model.encoder.token_states(ids,mask)
                first=model.encoder.readout(states,mask,lexical[group])
                pooled=model.rule_head[0](pooling(states,mask))
                proof=model.rule_head[1](pooled)
                original=torch.cat((first,proof),1).float().sigmoid()
                with torch.autocast('cuda',enabled=False):
                    projected=projection(pooled.half().float())
                values=torch.cat((original,projected),1).cpu().numpy()
                rows=offset+np.asarray(group)
                raw[rows,0],aux[rows],result[rows]=values[:,0],values[:,2:24],values[:,24:]
    return raw,aux,result


def regional_adjust(texts,original,margin,values,estimate,no,yes,prior,known,flag,bits,config,policy,score,learned):
    scope=known & ~flag & (((original>0.) & (margin<=0.)) | ((margin>0.) & (margin<SPANS)))
    if learned['width'] is not None:
        scope &= abs(margin)<learned['width']
    if not scope.any():
        return bits
    t=config['threshold']
    utility=query_value(policy,prior,no,yes,estimate['w'],config['gate']['base'])
    pre=np.column_stack((values,estimate['w'],no,yes,prior,utility))
    rows=dict(pre=pre[scope],unqueried=original[scope],queried=margin[scope],
              length=np.fromiter((len(s) for s,k in zip(texts,scope,strict=True) if k),np.int32))
    x=design(rows,t,score[scope])
    rank=learned['calibrator'].decision_function(x)
    weight=learned['strength']
    result=bits.copy()
    result[scope]=(1.-weight)*margin[scope]+weight*rank>learned['cut']
    return result
