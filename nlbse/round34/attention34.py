"""Train token attention for within-file ranking without repeating the compact pass."""
import argparse
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sections34 import HERE, PARTS, f, checksum, json_at, write_json
from eval34 import loaded, signatures
from predict_factor import full_context, ordered_batches, batch_plan, batch, small

DEST=HERE/'attention'
SAMPLE=160
HIDDEN=48
N_HEADS=4
EPOCHS=11
BATCH=128
SEED=2042
TRAINING=('point','paired','distilled_pairs')


class TokenHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.compress=nn.Linear(width,HIDDEN)
        self.attention=nn.Linear(HIDDEN,N_HEADS)
        self.rank=nn.Sequential(nn.Linear((N_HEADS+1)*HIDDEN,96),nn.GELU(),nn.Dropout(.10),nn.Linear(96,2))

    def forward(self,x,mask):
        state=F.gelu(self.compress(x))
        score=self.attention(state).transpose(1,2).masked_fill(~mask[:,None,:],-10000.)
        weights=score.float().softmax(-1).to(state.dtype)
        pooled=(weights@state).flatten(1)
        mean=(state*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.rank(torch.cat((mean,pooled),1))


def select_tokens(state,mask):
    length=mask.sum(1,keepdim=True).long().clamp_min(1)
    xy=torch.arange(SAMPLE,device=state.device)[None,:]
    selected=torch.where(length<=SAMPLE,xy,(xy*(length-1))//(SAMPLE-1)).clamp_max(state.shape[1]-1)
    valid=(xy<length).expand(-1,SAMPLE)
    data=state.gather(1,selected[:,:,None].expand(-1,-1,state.shape[-1]))
    return data,valid


@torch.inference_mode()
def cache_sequence(model,tok,texts):
    value=torch.empty((len(texts),SAMPLE,model.width),dtype=torch.float16,device='cuda')
    valid=torch.empty((len(texts),SAMPLE),dtype=torch.bool,device='cuda')
    def prepare(offset):
        return full_context(texts[offset:offset+1024],tok)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset,seq in ordered_batches(range(0,len(texts),1024),prepare,True):
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                ids,mask=batch(group,seq,tok.pad_token_id)
                state=model.encoder.token_states(ids,mask)
                data,keep=select_tokens(state,mask)
                rows=offset+np.asarray(group)
                value[rows],valid[rows]=data.half(),keep
    return value,valid


def train(x,mask,y,q,teacher,paired,kind,width):
    torch.manual_seed(SEED)
    head=TokenHead(width).cuda().train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0014,weight_decay=.025,fused=True)
    scale=(len(y)/y.sum()-1.).sqrt().clamp(1.,4.)
    a,b,pair_weights=paired
    n=len(a)
    records=[]
    start=perf_counter()
    for epoch in range(EPOCHS):
        loss_stats=torch.zeros(3,device='cuda')
        order=torch.randperm(len(x),device='cuda')
        for row in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logit=head(x[row],mask[row]).float()
                point_weights=.28+1.45*(-.7*q[row].abs()).exp()
                hard=(F.binary_cross_entropy_with_logits(logit[:,0],y[row],pos_weight=scale,reduction='none')*point_weights).mean()
                ranking=logit.new_zeros(())
                teachers=logit.new_zeros(())
                if kind!='point':
                    choose=torch.multinomial(pair_weights,48,replacement=True)
                    u,v=a[choose],b[choose]
                    lhs,rhs=head(x[u],mask[u]).float(),head(x[v],mask[v]).float()
                    ranking=F.softplus(-(lhs[:,0]-rhs[:,0])).mean()
                    if kind=='distilled_pairs':
                        teachers=(F.smooth_l1_loss(logit[:,1],teacher[row])+
                             .5*F.smooth_l1_loss(lhs[:,1]-rhs[:,1],teacher[u]-teacher[v]))
                loss=hard+.40*ranking+.22*teachers
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),.9)
            optimizer.step()
            loss_stats+=torch.stack((hard.detach(),ranking.detach(),teachers.detach()))
        records.append((loss_stats/len(order.split(BATCH))).tolist())
    return head.eval(),dict(training_seconds=perf_counter()-start,epoch_loss=records,pairs=n if kind!='point' else 0)


def main(rotation):
    phases,reference,receipt=loaded(rotation)
    all_text=f.inputs()[0]
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    chosen={}
    for phase,src in phases.items():
        ids=src['ids'][src['support']]
        chosen[phase]=(ids,*cache_sequence(model,tok,[all_text[i] for i in ids]))
        print(rotation,phase,'states',len(ids),flush=True)
    fit=phases['oof']
    mask=fit['support']
    zp,zn=signatures(rotation,fit)
    used=np.flatnonzero(mask)
    plus,minus=np.searchsorted(used,zp),np.searchsorted(used,zn)
    assert (used[plus]==zp).all() and (used[minus]==zn).all()
    _,labels,amount=np.unique(plus,return_inverse=True,return_counts=True)
    prepared=(torch.as_tensor(plus,device='cuda'),torch.as_tensor(minus,device='cuda'),
              torch.as_tensor(1./amount[labels],device='cuda',dtype=torch.float32))
    y=torch.as_tensor(fit['y'][mask],device='cuda',dtype=torch.float32)
    reference_rank=torch.as_tensor(fit['queried'][mask],device='cuda',dtype=torch.float32)
    teacher=torch.as_tensor(fit['teacher'][mask,0],device='cuda',dtype=torch.float32).clamp(-8.,8.)
    dest=DEST/rotation
    dest.mkdir(parents=True,exist_ok=True)
    results={}
    forecasts={}
    for kind in TRAINING:
        head,meta=train(*chosen['oof'][1:],y,reference_rank,teacher,prepared,kind,model.width)
        path=dest/f'{kind}.pt'
        torch.save(head.cpu().state_dict(),path)
        results[kind]=meta | {'sha256':checksum(path)}
        head.cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for phase,(ids,x,mask) in chosen.items():
                forecast=np.concatenate([head(a,b).float().cpu().numpy()
                                         for a,b in zip(x.split(BATCH),mask.split(BATCH),strict=True)])
                forecasts[phase+'_ids']=ids
                forecasts[phase+'_'+kind]=forecast
        print(rotation,kind,meta['training_seconds'],flush=True)
    path=dest/'scores.npz'
    np.savez_compressed(path,**forecasts)
    write_json(dest/'receipt.json',dict(points=reference['phases'],teacher_sha256=receipt['predictions_sha256'],
                   predictions_sha256=checksum(path),heads=results,sample=SAMPLE,epochs=EPOCHS,width=model.width,
                   code_sha256=checksum(HERE/'attention34.py'),seed=SEED,rows=len(y)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
