"""Compare pointwise and matched-file contrastive teacher updates."""
import argparse
from bisect import bisect_left
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'round34'))
from sections34 import PARTS, R30, MODEL, CAP, f, checksum, json_at, write_json, checkpoint, opening
from eval34 import loaded, signatures
from students import load_model, lexical_features, batch, batch_plan

DEST=HERE/'teachers'
VIEWS=('prefix','body')
ARMS=('point','pair')
SEED=5035
EPOCHS=2
PAIR_BATCH=40
BACKGROUND=24
BODY_TOKENS=192
HEADER_TOKENS=36
LAST_LAYERS=2
PAIR_COEFFICIENT=.52


def windows(texts,tok):
    role=tok.convert_tokens_to_ids('<encoder-only>')
    if role==tok.unk_token_id:
        raise ValueError('Encoder-only token is missing')
    header=[tok.bos_token_id,role,tok.eos_token_id]
    sides=torch.as_tensor(lexical_features(texts),device='cuda')
    result={k:[] for k in VIEWS}
    for start in range(0,len(texts),1024):
        chunk=texts[start:start+1024]
        found=tok(chunk,truncation=False,add_special_tokens=False,return_offsets_mapping=True,verbose=False)
        for text,ids,spans in zip(chunk,found['input_ids'],found['offset_mapping'],strict=True):
            cut=bisect_left([a for a,_ in spans],opening(text))
            full=ids[:CAP-4]
            if len(ids)<=BODY_TOKENS:
                shortened=ids
            else:
                lead=min(cut,HEADER_TOKENS)
                shortened=ids[:lead]+ids[cut:cut+BODY_TOKENS-lead]
            for kind,selected in zip(VIEWS,(full,shortened),strict=True):
                result[kind].append([*header,*selected,tok.eos_token_id])
    return result,sides


@torch.inference_mode()
def evaluate(model,tok,seq,sides,indices,center):
    selected=[seq[i] for i in indices]
    loc=sides[indices]
    output=np.empty(len(indices),np.float32)
    with torch.autocast('cuda',dtype=torch.float16):
        for group in batch_plan(list(range(len(indices))),selected,96):
            x,mask=batch(group,selected,tok.pad_token_id)
            output[group]=model(x,mask,loc[group])[:,0].float().cpu().numpy()-center
    return output


def trainable(net):
    net.requires_grad_(False)
    net.head.requires_grad_(True)
    net.side.requires_grad_(True)
    encoder=net.encoder.encoder.layer[-LAST_LAYERS:]
    for layer in encoder:
        layer.requires_grad_(True)
    net.eval()
    net.head.train()
    for layer in encoder:
        layer.train()
    return [p for layer in encoder for p in layer.parameters()]


def plans(plus,minus,weight,background,seed):
    rng=np.random.default_rng(seed)
    all_plans=[]
    for _ in range(EPOCHS):
        pair_order=rng.permutation(len(plus))
        for i in range(0,len(pair_order),PAIR_BATCH):
            group=pair_order[i:i+PAIR_BATCH]
            back=rng.choice(background,size=min(BACKGROUND,len(background)),replace=False)
            # Preserve the positive/control correspondence in both updates.
            chosen=np.r_[plus[group],minus[group],back]
            all_plans.append((chosen,weight[group],len(group)))
    return all_plans


def fit(net,tok,scheme,sequence,side,y,updates,balance):
    torch.manual_seed(SEED)
    learned=trainable(net)
    train=list(learned)+[*net.head.parameters(),*net.side.parameters()]
    optimizer=torch.optim.AdamW([dict(params=learned,lr=.000024),
                                 dict(params=[*net.head.parameters(),*net.side.parameters()],lr=.00012)],
                                weight_decay=.028,fused=True)
    schedule=get_linear_schedule_with_warmup(optimizer,max(1,len(updates)//10),len(updates))
    losses=torch.zeros(3,device='cuda')
    started=perf_counter()
    for step,(ix,importance,n) in enumerate(updates,1):
        tokens,mask=batch(ix.tolist(),sequence,tok.pad_token_id)
        w=torch.as_tensor(importance,device='cuda',dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=net(tokens,mask,side[ix])[:,0].float()
            gold=y[ix]
            loss=F.binary_cross_entropy_with_logits(z,gold,pos_weight=z.new_tensor(balance),reduction='none')
            weight=torch.cat((w,w,loss.new_full((len(ix)-2*n,),.65)))
            point=(loss*weight).sum()/weight.sum()
            delta=z[:n]-z[n:2*n]
            contrast=(F.softplus(-delta)*w).sum()/w.sum()
            cost=point+PAIR_COEFFICIENT*contrast if scheme=='pair' else point
        cost.backward()
        torch.nn.utils.clip_grad_norm_(train,.9)
        optimizer.step()
        schedule.step()
        losses+=torch.stack((cost.detach(),point.detach(),contrast.detach()))
    net.eval()
    return dict(steps=len(updates),seconds=perf_counter()-started,loss=(losses/len(updates)).tolist())


def main(rotation):
    phases,ref,windows_proof=loaded(rotation)
    train=phases['oof']
    index=np.flatnonzero(train['support'])
    a,b=signatures(rotation,train)
    ia,ib=np.searchsorted(index,a),np.searchsorted(index,b)
    assert np.array_equal(index[ia],a) and np.array_equal(index[ib],b)
    cohort={k:v[index] for k,v in train.items()}
    src=R30/'tuned_teacher'/rotation
    meta=json_at(src/'fit.json')
    assert checksum(MODEL/'pytorch_model.bin')==meta['upstream_sha256']
    assert checksum(src/'oof.npz')==meta['predictions_sha256']
    corpus=f.inputs()[0]
    net,tok=load_model('unixcoder9')
    seq,side=windows([corpus[i] for i in cohort['ids']],tok)
    truth=torch.as_tensor(cohort['y'],device='cuda',dtype=torch.float32)
    tests={}
    output={'oof_ids':cohort['ids'],'fold':cohort['fold']}
    for kind in VIEWS:
        for model in ('initial',*ARMS):
            output[f'oof_{kind}_{model}']=np.empty(len(index),np.float32)
    for part,data in phases.items():
        if part=='oof':
            continue
        ids=data['ids'][data['support']]
        scores,aux=windows([corpus[i] for i in ids],tok)
        output[part+'_ids']=ids
        tests[part]=(scores,aux)
    out=DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    records={}
    for fold in np.unique(cohort['fold']):
        keep=cohort['duplicates']&(1<<int(fold))==0
        check=cohort['fold']==fold
        assert not (keep&check).any()
        matching=keep[ia]&keep[ib]
        pairs=ia[matching],ib[matching]
        _,assignment,freq=np.unique(pairs[0],return_inverse=True,return_counts=True)
        cost=1./freq[assignment]
        # Equal total weight per missed positive, independent of its number of controls.
        cost/=cost.mean()
        background=np.flatnonzero(keep & (np.abs(cohort['queried'])<1.))
        assert len(background) and len(pairs[0])
        records[int(fold)]={}
        for variant,encoded in seq.items():
            start_balance=checkpoint(net,src,meta,int(fold))
            output[f'oof_{variant}_initial'][check]=evaluate(net,tok,encoded,side,np.flatnonzero(check),np.log(start_balance))
            if fold==0:
                for part,(examples,features) in tests.items():
                    output[f'{part}_{variant}_initial']=evaluate(net,tok,examples[variant],features,
                                                                  np.arange(len(features)),np.log(start_balance))
            updates=plans(*pairs,cost,background,SEED+int(fold))
            # The two arms use identical rows, order, steps, targets and optimizer schedules.
            balance=min(2.5,float(((int(keep.sum())-truth[keep].sum())/truth[keep].sum()).sqrt()))
            fixed_center=np.log(balance)
            for scheme in ARMS:
                checkpoint(net,src,meta,int(fold))
                row=fit(net,tok,scheme,encoded,side,truth,updates,balance)
                score=evaluate(net,tok,encoded,side,np.flatnonzero(check),fixed_center)
                output[f'oof_{variant}_{scheme}'][check]=score
                if fold==0:
                    for part,(items,features) in tests.items():
                        output[f'{part}_{variant}_{scheme}']=evaluate(net,tok,items[variant],features,
                                                                       np.arange(len(features)),fixed_center)
                checkpoint_path=out/f'{variant}_{scheme}_{int(fold)}.pt'
                torch.save({k:v.detach().cpu() for k,v in net.named_parameters() if v.requires_grad},checkpoint_path)
                row.update(sha256=checksum(checkpoint_path),balance=balance,pairs=int(matching.sum()),
                           held=int(check.sum()),fit_pool=int(keep.sum()))
                records[int(fold)][variant+'_'+scheme]=row
                print(rotation,int(fold),variant,scheme,row,flush=True)
    dest=out/'scores.npz'
    np.savez_compressed(dest,**output)
    write_json(out/'receipt.json',dict(rotation=rotation,teacher_scores_sha256=meta['predictions_sha256'],
               prior_views_sha256=windows_proof['predictions_sha256'],points=ref['phases'],heads=records,
               seed=SEED,epochs=EPOCHS,background=BACKGROUND,weights_last_layers=LAST_LAYERS,body_tokens=BODY_TOKENS,
               pair_batch=PAIR_BATCH,coefficient=PAIR_COEFFICIENT,predictions_sha256=checksum(dest),
               code_sha256=checksum(HERE/'teacher35.py')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
