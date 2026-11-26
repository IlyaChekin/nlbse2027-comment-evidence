"""Fit supervised teacher heads and upper layers strictly outside each scored file fold."""
import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from embed import HERE, MODEL, f, checksum, json_at, write_json, tokens
from teacher_probe import PARTS, signed_inputs, earlier_scores, census
from disagreement import clip_logit
from students import load_model, lexical_features, batch_plan, batch

TARGET=HERE/'tuned_teacher'
SEED=42
BATCH=96


@torch.inference_mode()
def evaluate(model,tok,seq,sides,indices,weight):
    result=np.empty(len(indices),dtype=np.float32)
    selected=[seq[i] for i in indices]
    side=sides[indices]
    with torch.autocast('cuda',dtype=torch.float16):
        for group in batch_plan(list(range(len(indices))),selected,BATCH):
            a,b=batch(group,selected,tok.pad_token_id)
            z=model(a,b,side[group])[:,0].float().cpu().numpy()
            result[group]=z-np.log(weight)
    return result


def run(rotation):
    rows,p,sources=signed_inputs(rotation)
    tasks=rows['ids']
    text,truth,_,_,_=f.inputs()
    assert np.array_equal(truth[tasks,0],rows['y'])
    _,tok=load_model('unixcoder9')
    torch.cuda.empty_cache()
    x=[text[i] for i in tasks]
    sequence=[]
    for i in range(0,len(x),2048):
        sequence.extend(tokens(x[i:i+2048],tok))
    side=torch.as_tensor(lexical_features(x),device='cuda')
    y=torch.as_tensor(rows['y'],dtype=torch.float32,device='cuda')
    score=np.empty(len(tasks),dtype=np.float32)
    out=TARGET/rotation
    out.mkdir(parents=True,exist_ok=True)
    records={}
    for k in np.unique(rows['fold']):
        held=rows['fold']==k
        keep=(rows['dup']&(1<<int(k)))==0
        if np.any(held&keep):
            raise ValueError('Teacher fit includes a target file or known copy')
        torch.manual_seed(SEED)
        rng=random.Random(SEED)
        net,_=load_model('unixcoder9')
        net.encoder.embeddings.requires_grad_(False)
        for layer in net.encoder.encoder.layer[:3]:
            layer.requires_grad_(False)
        upper=[p for p in net.encoder.parameters() if p.requires_grad]
        opt=torch.optim.AdamW([dict(params=upper,lr=.000045),
                               dict(params=[*net.head.parameters(),*net.side.parameters()],lr=.00020)],
                              weight_decay=.015,fused=True)
        order=np.flatnonzero(keep).tolist()
        rng.shuffle(order)
        plans=list(batch_plan(order,sequence,BATCH))
        rng.shuffle(plans)
        schedule=get_linear_schedule_with_warmup(opt,int(.04*len(plans)),len(plans))
        balance=min(4.,float(((len(order)-y[order].sum())/y[order].sum()).sqrt()))
        start=time.perf_counter()
        net.train()
        net.encoder.embeddings.eval()
        for layer in net.encoder.encoder.layer[:3]:
            layer.eval()
        for step, group in enumerate(plans,1):
            tokens_,mask=batch(group,sequence,tok.pad_token_id)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=net(tokens_,mask,side[group])[:,0]
                loss=F.binary_cross_entropy_with_logits(z,y[group],pos_weight=y.new_tensor(balance))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),.9)
            opt.step()
            schedule.step()
            if step%350==0:
                print(rotation,int(k),step,len(plans),round(time.perf_counter()-start,2),flush=True)
        net.eval()
        score[held]=evaluate(net,tok,sequence,side,np.flatnonzero(held),balance)
        checkpoint=out/f'upper_{k}.pt'
        torch.save({name:p.detach().cpu() for name,p in net.named_parameters() if p.requires_grad},checkpoint)
        records[int(k)]=dict(fit=int(keep.sum()),held=int(held.sum()),balance=balance,
                             seconds=time.perf_counter()-start,sha256=checksum(checkpoint))
        print(rotation,int(k),records[int(k)],flush=True)
        del net,opt,schedule
    raw=np.column_stack((clip_logit(p),rows['logits'][:,0],score))
    earlier=earlier_scores(tasks,rows['y'])
    values,report=census(rows['y'].astype(bool),raw,rows['fold'],earlier)
    path=out/'oof.npz'
    np.savez_compressed(path,ids=tasks,y=rows['y'],fold=rows['fold'],raw=raw,calibrated=values)
    write_json(out/'fit.json',dict(scope=report,sources=sources,folds=records,
               predictions_sha256=checksum(path),code_sha256=checksum(Path(__file__)),
               upstream_sha256=checksum(MODEL/'pytorch_model.bin'),rows=len(tasks),seed=SEED))
    print(rotation,report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    run(p.parse_args().part)
