"""Test external-label heads with and without unannotated negatives."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from eval51 import HERE, FIT, PARTS, f, load, json_at, checksum, configured, write_json
from eval51 import full_context, batch_plan, batch, compare
from support51 import DEST as AVAIL
from train51 import pooling
from source41 import parts

DEST=HERE/'external_probe'
ENCODERS=('original','components')
SUPERVISION=('all_matched','measured')
SEED=5192
EPOCHS=3
HEAD_BATCH=1024


@torch.inference_mode()
def vectors(model,tok,seq):
    z=torch.empty((len(seq),2*model.width),device='cuda',dtype=torch.float16)
    with torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(seq),4096):
            s=seq[start:start+4096]
            for idx in batch_plan(list(range(len(s))),s,128):
                x,mask=batch(idx,s,tok.pad_token_id)
                state=model.encoder.token_states(x,mask)
                value=model.rule_head[0](pooling(state,mask))
                z[start+np.asarray(idx)]=value.half()
    return z


def fit_head(pool,target,mask,seed):
    width=pool.shape[1]
    linear=nn.Linear(width,1).cuda()
    nn.init.zeros_(linear.weight)
    nn.init.constant_(linear.bias,-2.5)
    opt=torch.optim.AdamW(linear.parameters(),lr=.0022,weight_decay=.02)
    rng=torch.Generator(device='cuda').manual_seed(seed)
    prior=target[mask].mean()
    balance=((1.-prior)/prior.clamp_min(1e-5)).sqrt().clamp(1.,12.)
    raw_loss=[]
    for _ in range(EPOCHS):
        schedule=torch.randperm(len(pool),generator=rng,device='cuda')
        total=target.new_zeros(())
        for r in schedule.split(HEAD_BATCH):
            x,y=pool[r].float(),target[r]
            active=mask[r]
            opt.zero_grad(set_to_none=True)
            output=linear(x)[:,0]
            loss=(F.binary_cross_entropy_with_logits(output,y,pos_weight=balance,reduction='none')*active).sum()
            loss=loss/active.sum().clamp_min(1.)
            loss.backward()
            opt.step()
            total+=loss.detach()
        raw_loss.append(float(total))
    return linear.eval(),dict(loss=raw_loss,balance=float(balance),training_rows=int(mask.sum()),
                              examples=len(mask),steps=EPOCHS*((len(mask)+HEAD_BATCH-1)//HEAD_BATCH))


def main(rotation):
    begin=perf_counter()
    raw,_,stamp=load(rotation)
    frames=parts(raw)
    source=FIT/rotation/'fit.json'
    proof=json_at(source)
    assert proof['config']['sources']['matrix_sha256']==stamp
    roster=source.with_name('rows.npz')
    assert checksum(roster)==proof['config']['roster_sha256']
    with np.load(roster) as r:
        train=r['ids']
        positive=r['targets'][:,2]
        verified=r['component_available']
    mask=AVAIL/'mask.npz'
    assert json_at(mask.with_name('receipt.json'))['mask_sha256']==checksum(mask)
    with np.load(mask) as q:
        exercise=q['external_available']
        labels=(q['component']&6)>0
        append=q['available']
        training_coverage=exercise[train]
        assert np.array_equal(labels[train][verified],positive[verified])
        assert np.array_equal(verified,append[train])
    assert not np.isin(train,frames[0]['ids']).any() and not np.isin(train,frames[1]['ids']).any()
    corpus=f.inputs()[0]
    tok=f.small.tokenizer()
    seq=[full_context([corpus[i] for i in ids],tok) for ids in (train,*(m['ids'] for m in frames))]
    held_array={}
    forecasters={}
    trained={}
    performance={}
    for variant in ENCODERS:
        backbone=None if variant=='original' else variant
        model,_=configured(rotation,backbone,tok,proof)
        fit,*verify=[vectors(model,tok,s) for s in seq]
        for coverage in SUPERVISION:
            mask=verified if coverage=='all_matched' else training_coverage
            y=torch.as_tensor(positive,device='cuda',dtype=torch.float32)
            valid=torch.as_tensor(mask,device='cuda')
            head,notes=fit_head(fit,y,valid,SEED)
            measure={}
            output=[]
            with torch.inference_mode():
                for n,(field,frame) in enumerate(zip(verify,frames,strict=True)):
                    val=head(field.float())[:,0].cpu().numpy()
                    output.append(val.astype(np.float64))
                    ids=frame['ids']
                    known=append[ids]
                    in_scope=exercise[ids]
                    measure[str(n)]=dict(ap_all=float(average_precision_score(labels[ids][known],val[known])),
                         ap_measured=float(average_precision_score(labels[ids][in_scope],val[in_scope])),
                         scope=int(in_scope.sum()),positives=int((labels[ids]&in_scope).sum()))
                    held_array[f'{n}_{variant}_{coverage}']=val
            key=(variant,coverage)
            trained['/'.join(key)]=head.cpu().state_dict()
            performance['/'.join(key)]=dict(fit=notes,held=measure)
            forecasters[key]=[np.maximum(x['signed'],v) for x,v in zip(frames,output,strict=True)]
            print(rotation,variant,coverage,measure,round(perf_counter()-begin,2),flush=True)
        del fit,verify,model
    table,bits=compare(rotation,frames,forecasters)
    folder=DEST/rotation
    folder.mkdir(exist_ok=True,parents=True)
    p=folder/'predictions.npz'
    np.savez_compressed(p,**bits,**held_array)
    weights=folder/'weights.pt'
    torch.save(trained,weights)
    write_json(folder/'results.json',dict(matrix_sha256=stamp,training_sha256=checksum(source),
               support_sha256=checksum(AVAIL/'receipt.json'),code_sha256=checksum(Path(__file__)),
               predictions_sha256=checksum(p),weights_sha256=checksum(weights),
               results=table,probes=performance,seconds=perf_counter()-begin))
    print(rotation,len(table),round(perf_counter()-begin,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
