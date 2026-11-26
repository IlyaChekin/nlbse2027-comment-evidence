"""Read an extra score from the compact model's existing token states."""
import time

import numpy as np
import torch
from torch import nn
from torch.nn.functional import layer_norm

from predict_factor import full_context, lexical_features, ordered_batches, batch_plan, batch, small
from rule_model import pooling

FEATURES=4


def head(width):
    layer=nn.Sequential(nn.Linear(FEATURES*width,96),nn.GELU(),nn.Dropout(.12),nn.Linear(96,1))
    nn.init.zeros_(layer[-1].weight)
    nn.init.zeros_(layer[-1].bias)
    return layer


def views(hidden,mask,normalizer):
    n=hidden.shape[1]
    lengths=mask.sum(1,keepdim=True)
    ix=torch.arange(n,device=hidden.device).unsqueeze(0)
    preferred=(ix*3<lengths)&mask.bool()
    tail=(ix*3>=2*lengths)&mask.bool()
    mix=torch.stack((preferred,tail),dim=1).to(hidden.dtype)
    regions=(mix@hidden)/mix.sum(-1,keepdim=True).clamp_min(1.)
    regions=layer_norm(regions,(hidden.shape[-1],))
    pooled=normalizer(pooling(hidden,mask))
    return pooled,torch.cat((pooled,regions.flatten(1)),dim=1)


def stack(model,tok,text):
    base=torch.empty(len(text),dtype=torch.float32,device='cuda')
    vec=torch.empty((len(text),model.width*FEATURES),dtype=torch.float16,device='cuda')
    def prepare(offset):
        return full_context(text[offset:offset+2048],tok)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i,seq in ordered_batches(range(0,len(text),2048),prepare,True):
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                x,mask=batch(group,seq,tok.pad_token_id)
                z=model.encoder.token_states(x,mask)
                a,b=views(z,mask,model.rule_head[0])
                sel=i+np.asarray(group)
                vec[sel]=b.half()
                base[sel]=model.rule_head[1](a)[:,20].float()
    return vec,base


@torch.inference_mode()
def predict(model,tok,code,head_list):
    p=np.empty((len(code),2),dtype=np.float32)
    ar=np.empty((len(code),22),dtype=np.float32)
    extra=np.empty((len(code),len(head_list)),dtype=np.float32)
    start=time.perf_counter()
    spent=0.
    overhead=0
    gflops=0.
    def prepare(offset):
        segment=code[offset:offset+2048]
        return full_context(segment,tok),lexical_features(segment)
    with torch.autocast('cuda',dtype=torch.float16):
        for u,(seq,lex) in ordered_batches(range(0,len(code),2048),prepare,True):
            p[u:u+len(seq),1]=lex[:,:4].any(axis=1)
            side=torch.as_tensor(lex,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                x,m=batch(group,seq,tok.pad_token_id)
                now=time.perf_counter()
                hidden=model.encoder.token_states(x,m)
                if head_list:
                    a,b=views(hidden,m,model.rule_head[0])
                else:
                    a=model.rule_head[0](pooling(hidden,m))
                aux=model.rule_head[1](a)
                main=model.encoder.readout(hidden,m,side[group])
                exposed=[aux[:,20].float()+h(b)[:,0].float() for h in head_list]
                raw=torch.column_stack((main.float().sigmoid(),aux.float().sigmoid(),*exposed)).cpu().numpy()
                sel=u+np.asarray(group)
                p[sel,0],ar[sel],extra[sel]=raw[:,0],raw[:,2:24],raw[:,24:]
                spent+=time.perf_counter()-now
                gflops+=small.operations(len(group),x.shape[1],model.width)
                gflops+=len(group)*(4*model.width*22+6*model.width)/1e9
                if head_list:
                    overhead+=(4*x.shape[1]*model.width + 12*model.width+
                               len(head_list)*2*(FEATURES*model.width*96+96))*len(group)
    return p,ar,extra,dict(forward_seconds=spent,wall_seconds=time.perf_counter()-start,extra_gflops=overhead/1e9,
                                     analytic_gflops=gflops+overhead/1e9)
