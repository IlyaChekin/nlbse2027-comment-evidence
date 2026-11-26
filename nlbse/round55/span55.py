"""Train short token readouts with an optional exact archival span objective."""
import argparse
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'round53'),str(ROOT/'round51')]
from train51 import (PARTS, batch, batch_plan, checksum, f, full_context, json_at, lexical_features,
                     load, release_start, small, write_json)
from rule_model import pooling
from project_split import fingerprint

FIT=HERE/'training'
SOURCE=ROOT/'round54/spans'
LABELS=ROOT/'round49/components/labels.npz'
ARMS=('pooled','local','sham','span')
SEED=5537
EPOCHS=4
WIDTH=32
BATCH=128
BETA=.22


class Head(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode=mode
        self.project=nn.Linear(width,WIDTH)
        self.logits=nn.Linear(WIDTH,2)
        self.skip=nn.Linear(2,2,bias=False)
        with torch.no_grad():
            self.skip.weight.copy_(torch.eye(2))

    def forward(self, states, mask, native):
        valid=mask.bool()
        skip=self.skip(native).float()
        if self.mode=='pooled':
            joined=(mask.unsqueeze(-1).to(states.dtype)*states).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            return self.logits(F.gelu(self.project(joined))).float()+skip,None
        local=self.logits(F.gelu(self.project(states))).float()
        local=local.masked_fill(~valid[:,:,None],-1.e4)
        # Smooth pooling makes short evidence compete within each function.
        score=.5*(torch.logsumexp(local*2.,dim=1)-valid.sum(1).clamp_min(1).log()[:,None])
        return score+skip,local[:,:,1]


def roster(rotation):
    path=SOURCE/'targets.npz'
    check=json_at(SOURCE/'receipt.json')
    assert check['file_sha256']==checksum(path)
    labels_proof=json_at(LABELS.with_name('receipt.json'))
    assert labels_proof['predictions_sha256']==checksum(LABELS)
    with np.load(path) as z:
        selected=z[rotation+'_fit']
        labeled=z['ids'][selected]
        regions=z['spans'][selected,:,2:4]
        measured=z['visible_scanner'][selected,0]
        archived=z['archival_w'][selected]
    with np.load(LABELS) as z:
        archive=z['selected']
        bits=z['component']
        gold=z['truth']
    texts, truth, projects, files, _=f.inputs()
    assert np.array_equal(truth[:,0],gold) and archive[labeled].all()
    assert np.array_equal((bits[labeled]&1)>0,archived)
    state,_,stamp=load(rotation)
    oof=state['oof']
    hold=np.unique(np.r_[state['dev']['ids'],state.get('test',state['dev'])['ids']])
    blacklist={(projects[i],files[i]) for i in hold if files[i]}
    duplicate={fingerprint(texts[i]) for i in hold}
    assert not np.intersect1d(labeled,hold).size and np.isin(labeled,oof['ids']).all()
    assert all(files[i] and (projects[i],files[i]) not in blacklist and fingerprint(texts[i]) not in duplicate for i in labeled)
    absent=(archive[oof['ids']] & ~np.isin(oof['ids'],labeled))
    also=np.zeros(len(absent),bool)
    rng=np.random.default_rng(SEED)
    # Add assessment strata at the observed decision boundary and background strata.
    for scope,n in ((oof['old'],1800),(~oof['old'],2600)):
        for comp in (1,2,0):
            code=bits[oof['ids']]
            field=((code&1)>0) if comp==1 else ((code>0)&((code&1)==0)) if comp==2 else (code==0)
            candidates=np.flatnonzero(absent & scope & field)
            order=rng.permutation(candidates)
            keep=[]
            for k in order:
                i=int(oof['ids'][k])
                if files[i] and (projects[i],files[i]) not in blacklist and fingerprint(texts[i]) not in duplicate:
                    keep.append(k)
                    if len(keep)==n:
                        break
            also[np.asarray(keep,dtype=np.int32)]=True
    ids=np.r_[labeled,oof['ids'][also]]
    assert len(ids)==len(np.unique(ids))
    tagged=np.full((len(ids),2,2),-1,dtype=np.int32)
    tagged[:len(labeled)]=regions
    y=np.column_stack((gold[ids],(bits[ids]&1)>0)).astype(np.float32)
    tagged[~y[:,1].astype(bool)]=-1
    return ids,[texts[i] for i in ids],y,tagged,dict(archival_span_sha256=checksum(path),
             annotation_sha256=checksum(LABELS),matrix_sha256=stamp,aligned_rows=len(labeled),
             aligned_missed=int(((measured==0)&archived).sum()),held_rows=len(hold))


def tokenize(texts,tok,regions=None):
    seq, from_, control=[],[],[]
    real=np.zeros(len(texts),bool)
    for first in range(0,len(texts),1024):
        local=texts[first:first+1024]
        q=tok(local,truncation=False,verbose=False,return_offsets_mapping=regions is not None)
        seq.extend(q['input_ids'])
        if regions is None:
            continue
        for k, offsets in enumerate(q['offset_mapping'], first):
            o=np.asarray(offsets)
            content=o[:,1]>o[:,0]
            mark=np.zeros(len(o),bool)
            for a,b in regions[k]:
                if a>=0 and b>a:
                    mark|=content & (o[:,0]<b) & (o[:,1]>a)
            shuffled=np.zeros(len(o),bool)
            where=np.flatnonzero(content)
            if len(where) and mark.any():
                roll=np.roll(mark[where],max(1,len(where)//2))
                shuffled[where]=roll
                real[k]=True
                assert shuffled.sum()==mark.sum()
            from_.append(mark)
            control.append(shuffled)
    assert seq[:64]==full_context(texts[:64],tok)
    return seq,from_,control,real


def masks(rows, seq, width, device):
    a=np.zeros((len(rows),width),bool)
    for j,i in enumerate(rows):
        a[j,:len(seq[i])]=seq[i]
    return torch.from_numpy(a).to(device,non_blocking=True)


def region_loss(local, selected, usable):
    active=selected.any(1)
    total=torch.logsumexp(local.masked_fill(~usable.bool(),-1.e4),1)
    on=torch.logsumexp(local.masked_fill(~selected,-1.e4),1)
    return ((total-on)*active).sum()/active.sum().clamp_min(1)


def train(rotation):
    ids,text,y,targets,source=roster(rotation)
    tok=f.small.tokenizer()
    seq,pos,decoy,live=tokenize(text,tok,targets)
    device='cuda'
    original,anchor=release_start(rotation,tok)
    original.eval().requires_grad_(False)
    side=torch.as_tensor(lexical_features(text),device=device)
    wanted=torch.as_tensor(y,device=device)
    class_weights=torch.as_tensor(np.sqrt((len(y)-y.sum(0))/y.sum(0)).clip(1.,2.5),device=device,dtype=torch.float32)
    models={}
    for name in ARMS:
        torch.manual_seed(SEED)
        models[name]=Head(original.width,name).cuda()
    opt={n:torch.optim.AdamW(m.parameters(),lr=.0024,weight_decay=.018,fused=True) for n,m in models.items()}
    rows=list(range(len(seq)))
    total=EPOCHS*len(list(batch_plan(rows,seq,BATCH)))
    sched={n:get_linear_schedule_with_warmup(o,int(total*.08),total) for n,o in opt.items()}
    rng=random.Random(SEED)
    trace={n:[] for n in ARMS}
    start=perf_counter()
    step=0
    for epoch in range(EPOCHS):
        rng.shuffle(rows)
        plan=list(batch_plan(rows,seq,BATCH))
        rng.shuffle(plan)
        meter={n:torch.zeros(3,device=device) for n in ARMS}
        for group in plan:
            step+=1
            bt,mask=batch(group,seq,tok.pad_token_id)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                hidden=original.encoder.token_states(bt,mask)
                q=original.encoder.readout(hidden,mask,side[group])[:,0]
                w=original.rule_head(pooling(hidden,mask))[:,0]
                native=torch.stack((q,w),dim=1)
            region={name:masks(group,s,hidden.shape[1],device) for name,s in (('span',pos),('sham',decoy))}
            for name,m in models.items():
                opt[name].zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    p,local=m(hidden,mask,native)
                    main=F.binary_cross_entropy_with_logits(p,wanted[group],pos_weight=class_weights,reduction='none').mean(0)
                    loc=region_loss(local,region[name],mask) if name in region else main.new_zeros(())
                    loss=.52*main[0]+main[1]+BETA*loc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt[name].step()
                sched[name].step()
                meter[name]+=torch.cat((main.detach(),loc.detach()[None]))
        for name in ARMS:
            trace[name].append((meter[name]/len(plan)).tolist())
        print(rotation,epoch+1,step,round(perf_counter()-start,2),{n:t[-1] for n,t in trace.items()},flush=True)
    assert total==step
    root=FIT/rotation
    root.mkdir(exist_ok=True,parents=True)
    data=root/'rows.npz'
    np.savez_compressed(data,ids=ids,y=y,targets=targets,span_access=live)
    receipt={}
    for name,m in models.items():
        path=root/(name+'.pt')
        temp=path.with_suffix('.part')
        torch.save(m.cpu().state_dict(),temp)
        temp.replace(path)
        receipt[name]=dict(sha256=checksum(path),loss=trace[name],steps=step,
                           parameters=sum(x.numel() for x in m.parameters()))
    write_json(root/'fit.json',dict(rows=len(y),positive=y.sum(0).astype(int).tolist(),available_spans=int(live.sum()),
               aligned=source,checkpoint_sha256=anchor,fit_sha256=checksum(data),train_sha256=checksum(Path(__file__)),
               arms=receipt,config=dict(epochs=EPOCHS,seed=SEED,projection=WIDTH,batch=BATCH,span_weight=BETA,
                                       backbone='frozen',class_weights=class_weights.cpu().tolist()),
               elapsed=perf_counter()-start))


def readout(root,name,model,receipt):
    weight=root/(name+'.pt')
    assert checksum(weight)==receipt['arms'][name]['sha256']
    head=Head(model.width,name).cuda().eval()
    head.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
    return head


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    train(p.parse_args().rotation)
