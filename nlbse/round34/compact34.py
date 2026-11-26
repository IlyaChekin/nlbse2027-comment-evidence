"""Read local evidence without a second compact encoder pass."""
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn.functional import layer_norm

from predict_factor import full_context, lexical_features, ordered_batches, batch_plan, batch, small
from rule_model import pooling

ARMS={'global':(2,1),'local':(5,1),'teacher_head':(5,2),'teacher_regions':(5,5)}


def head(width,mode):
    supplied,count=ARMS[mode]
    return nn.Sequential(nn.Linear(supplied*width,96),nn.GELU(),nn.Dropout(.10),nn.Linear(96,count))


def pool(hidden,mask,normalizer,need_local=True):
    pooled=normalizer(pooling(hidden,mask))
    if not need_local:
        return pooled,pooled
    length=mask.sum(1,keepdim=True).clamp_min(1)
    idx=torch.arange(mask.shape[1],device=mask.device)[None,:]
    part=(3*idx)//length
    divided=(part[:,None,:]==torch.arange(3,device=mask.device)[None,:,None]) & mask[:,None,:].bool()
    owned=divided.to(hidden.dtype)
    regions=(owned@hidden)/owned.sum(2,keepdim=True).clamp_min(1.)
    regions=layer_norm(regions,(hidden.shape[-1],))
    return pooled,torch.cat((pooled,regions.flatten(1)),1)


@torch.inference_mode()
def cache(model,tok,texts):
    target=torch.empty(size=(len(texts),5*model.width),dtype=torch.float16,device='cuda')
    def prepare(offset):
        return full_context(texts[offset:offset+2048],tok)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset,seq in ordered_batches(range(0,len(texts),2048),prepare,True):
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                x,m=batch(group,seq,tok.pad_token_id)
                hidden=model.encoder.token_states(x,m)
                _,features=pool(hidden,m,model.rule_head[0])
                target[offset+np.asarray(group)]=features.half()
    return target


@torch.inference_mode()
def predict(model,tok,texts,extra=None,mode=None):
    p=np.empty((len(texts),2),np.float32)
    aux=np.empty((len(texts),22),np.float32)
    count=0 if extra is None else ARMS[mode][1]
    result=np.empty((len(texts),count),np.float32)
    compute,gflops,debit=0.,0.,0.
    start=perf_counter()
    use_local=extra is not None and ARMS[mode][0]==5
    def prepare(offset):
        rows=texts[offset:offset+2048]
        return full_context(rows,tok),lexical_features(rows)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset,(seq,side) in ordered_batches(range(0,len(texts),2048),prepare,True):
            p[offset:offset+len(seq),1]=side[:,:4].any(axis=1)
            known=torch.as_tensor(side,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                ids,mask=batch(group,seq,tok.pad_token_id)
                before=perf_counter()
                hidden=model.encoder.token_states(ids,mask)
                standard,regional=pool(hidden,mask,model.rule_head[0],use_local)
                original=model.encoder.readout(hidden,mask,known[group])
                auxiliary=model.rule_head[1](standard)
                merged=torch.cat((original,auxiliary),1).float().sigmoid()
                location=offset+np.asarray(group)
                if extra is not None:
                    adjusted=extra(regional).float()
                    merged=torch.cat((merged,adjusted),1)
                    if use_local:
                        debit+=len(group)*6*ids.shape[1]*model.width
                    debit+=len(group)*2*(ARMS[mode][0]*model.width*96+96*count)
                values=merged.cpu().numpy()
                p[location,0],aux[location],result[location]=values[:,0],values[:,2:24],values[:,24:]
                compute+=perf_counter()-before
                gflops+=small.operations(len(group),ids.shape[1],model.width)
                gflops+=len(group)*(4*model.width*22+6*model.width)/1e9
    return p,aux,result,dict(forward_seconds=compute,wall_seconds=perf_counter()-start,
                             analytic_gflops=gflops+debit/1e9,extra_gflops=debit/1e9)
