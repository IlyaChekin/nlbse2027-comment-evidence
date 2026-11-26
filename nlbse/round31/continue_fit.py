"""Train the same residual readout with either frozen or trainable late encoder blocks."""
import argparse
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from common31 import HERE,R30,OUT,PARTS,DEST,f,checksum,json_at,write_json,load_target,signed_source
from student_runtime import head,views
from predict_factor import full_context
from students import batch,batch_plan,lexical_features,ordered_batches

SEED=201
BATCH=128
EPOCHS=2
MODES=('head','late','late_selected')


def original_weights(where):
    root=f.HERE/where/str(f.SEED)
    receipt=json_at(root/'fit.json')['modes']['factor']
    assert checksum(root/receipt['file'])==receipt['sha256']
    return receipt['sha256']


def extract(net,tok,text,sequence,side,index):
    out=torch.empty((len(index),24),dtype=torch.float16,device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for base in range(0,len(index),4096):
            ix=index[base:base+4096]
            seq=[sequence[j] for j in ix]
            for group in batch_plan(list(range(len(seq))),seq,BATCH):
                x,mask=batch(group,seq,tok.pad_token_id)
                score=net(x,mask,side[ix[group]])
                out[base+np.asarray(group)]=score.float().clamp(-16.,16.).half()
    return out


def fit(rotation,which):
    teacher,receipt=load_target(R30/'tuned_teacher',rotation)
    earlier,old_receipt=load_target(DEST,rotation)
    if not all(np.array_equal(teacher[k],earlier[k]) for k in ('ids','y','fold')):
        raise ValueError('The aligned file-disjoint teachers changed')
    all_rows,source=signed_source(rotation)
    mask=all_rows['weak']==0
    if not np.array_equal(all_rows['ids'][mask],teacher['ids']):
        raise ValueError('Rule-negative fit was reordered')
    corpus,labels,_,_,_=f.inputs()
    fit_rows=all_rows['ids']
    if not np.array_equal(labels[fit_rows,0],all_rows['y']):
        raise ValueError('Training labels do not align')
    # Keep every rule-negative example and an independently sampled reference for other routes.
    rnd=np.random.default_rng(SEED)
    remainder=rnd.permutation(np.flatnonzero(~mask))[:32768]
    selected=np.sort(np.r_[np.flatnonzero(mask),remainder])
    ids=fit_rows[selected]
    is_negative=mask[selected]
    target=np.zeros((len(ids),2),np.float32)
    target[is_negative]=np.column_stack((earlier['calibrated'][:,3],teacher['calibrated'][:,4]))
    tok=f.small.tokenizer()
    phrases=[corpus[i] for i in ids]
    sequence=full_context(phrases,tok)
    side=torch.as_tensor(lexical_features(phrases),device='cuda')
    trainer=f.pipe(rotation,tok).encoder
    for block in trainer.encoder.local:
        block.use_shifted=False
    anchor=extract(trainer,tok,phrases,sequence,side,np.arange(len(phrases)))
    y=torch.as_tensor(labels[ids,0],dtype=torch.float32,device='cuda')
    active=torch.as_tensor(is_negative,dtype=torch.bool,device='cuda')
    q=torch.as_tensor(target,device='cuda').clamp(.0001,.9999)
    temp=1.7
    aimed=(q[:,1].logit()/temp).sigmoid()
    reference_likelihood=torch.where(y[:,None]==1,q,1.-q).log()
    useful=((reference_likelihood[:,1]-reference_likelihood[:,0])*3.).clamp(0.,1.)
    balance=min(3.5,float(((active.sum()-y[active].sum())/y[active].sum()).sqrt()))
    out=OUT/rotation
    out.mkdir(parents=True,exist_ok=True)
    artifact=out/'fit.json'
    settings=dict(rows=len(ids),negative=int(is_negative.sum()),other=int((~is_negative).sum()),
                  teacher_sha256=receipt['predictions_sha256'],earlier_sha256=old_receipt['predictions_sha256'],
                  source_sha256=source,baseline_sha256=original_weights(rotation),seed=SEED,
                  batch=BATCH,epochs=EPOCHS,width=trainer.width,code_sha256=checksum(HERE/'continue_fit.py'))
    checked=json_at(artifact) if artifact.exists() else dict(config=settings,arms={})
    if checked['config']!=settings:
        raise ValueError('Previous continuation run used different settings')
    del trainer
    for mode in which:
        if mode in checked['arms'] and checksum(out/f'{mode}.pt')==checked['arms'][mode]['sha256']:
            continue
        torch.manual_seed(SEED)
        rng=random.Random(SEED)
        model=f.pipe(rotation,tok).encoder
        model.eval()
        model.requires_grad_(False)
        for block in model.encoder.local:
            block.use_shifted=False
        if mode!='head':
            for block in model.encoder.local[-2:]:
                block.requires_grad_(True)
            for block in (model.encoder.norm,model.encoder.query,model.encoder.kv,model.encoder.global_out):
                block.requires_grad_(True)
        addon=head(model.width).cuda()
        opt=torch.optim.AdamW([dict(params=addon.parameters(),lr=.0010),
                               dict(params=[p for p in model.parameters() if p.requires_grad],lr=.00015)],
                               weight_decay=.022,fused=True)
        order=list(range(len(ids)))
        groups=list(batch_plan(order,sequence,BATCH))
        total=EPOCHS*len(groups)
        lr=get_linear_schedule_with_warmup(opt,int(.04*total),total)
        stats=[]
        t0=time.perf_counter()
        for epoch in range(EPOCHS):
            rng.shuffle(order)
            groups=list(batch_plan(order,sequence,BATCH))
            rng.shuffle(groups)
            sums=torch.zeros(3,device='cuda')
            addon.train()
            for step,ix in enumerate(groups,1):
                tokens,mask_=batch(ix,sequence,tok.pad_token_id)
                truth=y[ix]
                use=active[ix]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model.encoder.token_states(tokens,mask_)
                    pooled,representation=views(hidden,mask_,model.rule_head[0])
                    aux=model.rule_head[1](pooled)
                    raw=model.encoder.readout(hidden,mask_,side[ix])
                    base=torch.cat((raw,aux),1).float()
                    offset=addon(representation)[:,0].float()
                    z=aux[:,20].float()+offset
                    elem=F.binary_cross_entropy_with_logits(z,truth,pos_weight=truth.new_tensor(balance),
                                                            reduction='none')
                    hard=(elem*use).sum()/use.sum().clamp_min(1.)
                    per=F.binary_cross_entropy_with_logits(z/temp,aimed[ix],reduction='none')
                    weight=use.float() if mode!='late_selected' else use*useful[ix]
                    soft=(per*weight).sum()/weight.sum().clamp_min(1.)
                    delta=F.smooth_l1_loss(base.clamp(-16,16),anchor[ix].float(),reduction='none',beta=.5)
                    preserve=delta[:,[0,2,22,23]].mean()+.18*delta.mean()
                    loss=hard+.42*temp**2*soft+.026*offset.square().mean()+3.2*preserve
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*addon.parameters(),*[p for p in model.parameters() if p.requires_grad]],.85)
                opt.step()
                lr.step()
                sums+=torch.stack((hard.detach(),soft.detach(),preserve.detach()))
                if step%600==0:
                    print(rotation,mode,epoch,step,len(groups),round(time.perf_counter()-t0,2),flush=True)
            stats.append((sums/len(groups)).tolist())
        path=out/f'{mode}.pt'
        record=dict(backbone={k:v.detach().cpu() for k,v in model.named_parameters() if v.requires_grad},
                    head=addon.cpu().state_dict())
        stage=path.with_suffix('.part')
        torch.save(record,stage)
        stage.replace(path)
        checked['arms'][mode]=dict(sha256=checksum(path),seconds=time.perf_counter()-t0,
                                  loss=stats,trainable_backbone=len(record['backbone']))
        write_json(artifact,checked)
        print(rotation,mode,checked['arms'][mode],flush=True)
        del addon,model,opt,lr,record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    a=p.parse_args()
    fit(a.part,a.modes)
