"""Compare matched compact continuations with and without archive components."""
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
sys.path[:0]=[str(ROOT/'round50'),str(ROOT/'round31')]
from study50 import PARTS, f, checksum, json_at, load, write_json
from common31 import signed_source
from components49 import DEST as COMPONENTS
from project_split import fingerprint
from predict_factor import full_context, lexical_features, batch, batch_plan, ordered_batches, small
from rule_model import pooling

DEST=HERE/'training'
SEED=5116
EPOCHS=2
BATCH=128
ARMS=('frozen_union','frozen_components','union','components')
COMPONENT_WEIGHTS=(.55,.45)
KEEP=2.5
HEAD_WIDTH=64


def extra_head(width):
    model=nn.Sequential(nn.Linear(2*width,HEAD_WIDTH),nn.GELU(),nn.Dropout(.1),nn.Linear(HEAD_WIDTH,3))
    nn.init.zeros_(model[-1].weight)
    with torch.no_grad():
        model[-1].bias[:] = torch.tensor([0.,0.,-4.])
    return model


def encode(model,ids,mask,side):
    hidden=model.encoder.token_states(ids,mask)
    state=model.rule_head[0](pooling(hidden,mask))
    original=model.encoder.readout(hidden,mask,side)
    rules=model.rule_head[1](state)
    return torch.cat((original,rules),1).float(),state


def added(addon, logits, pooled):
    core=addon(pooled).float()
    return core+F.pad(logits[:,[0,2]],(0,1))


def release_start(rotation, tok):
    p=f.HERE/rotation/str(f.SEED)/'fit.json'
    report=json_at(p)['modes']['factor']
    assert checksum(p.parent/report['file'])==report['sha256']
    return f.pipe(rotation,tok).encoder,report['sha256']


def prepare(rotation):
    files,_,stamp=load(rotation)
    raw=files['oof']
    initial,proof=signed_source(rotation)
    for mark in ('ids','y','fold'):
        assert np.array_equal(raw[mark],initial[mark])
    texts,labels,project,paths,_=f.inputs()
    y=labels[:,0].astype(bool)
    assert np.array_equal(y[raw['ids']],raw['y'])
    d,h=files['dev'],files.get('test',files['dev'])
    reject=np.unique(np.r_[d['ids'],h['ids']])
    assert not np.intersect1d(reject,raw['ids']).size
    fit_files={(project[i],paths[i]) for i in raw['ids'] if paths[i]}
    dev_in_train=sum(bool(paths[i]) and (project[i],paths[i]) in fit_files for i in d['ids'])
    # Keep the original official row split for comparability. Its clean-file subset
    # is reported independently. Both project rotations prohibit shared files.
    if rotation!='official':
        assert dev_in_train==0 and not set(project[h['ids']])&set(project[raw['ids']])
    copies={fingerprint(texts[i]) for i in reject}
    ids=np.asarray([i for i in raw['ids'] if fingerprint(texts[i]) not in copies],np.int64)
    signature=COMPONENTS/'receipt.json'
    src=signature.with_name('labels.npz')
    ticket=json_at(signature)
    assert checksum(src)==ticket['predictions_sha256']
    assert ticket['training_sha256']==json_at(ROOT/'data/manifest.json')['files'][0]['sha256']
    with np.load(src) as z:
        assert np.array_equal(z['truth'][ids],y[ids])
        measured=z['selected'][ids]
        component=z['component'][ids]
    w,e=(component&1)>0,(component&6)>0
    assert np.array_equal(y[ids][measured],(w|e)[measured])
    target=np.column_stack((y[ids],w&measured,e&measured)).astype(np.float32)
    return ids,[texts[i] for i in ids],target,measured,dict(matrix_sha256=stamp,signed_folds_sha256=proof,
                      annotations_sha256=checksum(signature),removed_copies=len(raw['ids'])-len(ids),
                      dev_rows_in_fit_files=dev_in_train)


@torch.inference_mode()
def cache_base(model,tok,seq,side):
    original=torch.empty((len(seq),24),dtype=torch.float16,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(seq),4096):
            block=seq[start:start+4096]
            for rows in batch_plan(list(range(len(block))),block,BATCH):
                token,mask=batch(rows,block,tok.pad_token_id)
                ids=start+np.asarray(rows)
                value,_=encode(model,token,mask,side[ids])
                original[ids]=value.clamp(-16.,16.).half()
    return original


def late(model,mode):
    model.eval().requires_grad_(False)
    for layer in model.encoder.local:
        layer.use_shifted=False
    if not mode.startswith('frozen'):
        for layer in model.encoder.local[-2:]:
            layer.requires_grad_(True)
        for layer in (model.encoder.norm,model.encoder.query,model.encoder.kv,model.encoder.global_out):
            layer.requires_grad_(True)


def optimize(rotation,mode,tok,seq,side,targets,valid,weight,anchor):
    torch.manual_seed(SEED)
    rng=random.Random(SEED)
    net=f.pipe(rotation,tok).encoder
    late(net,mode)
    readout=extra_head(net.width).cuda()
    updated=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([dict(params=readout.parameters(),lr=.0011),dict(params=updated,lr=.00018)],
                                 fused=True,weight_decay=.02)
    ids=list(range(len(seq)))
    total=EPOCHS*len(list(batch_plan(ids,seq,BATCH)))
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.04*total),total)
    started=perf_counter()
    history=[]
    step=0
    aux=targets.new_tensor(COMPONENT_WEIGHTS if mode.endswith('components') else (0.,0.))
    for epoch in range(EPOCHS):
        rng.shuffle(ids)
        groups=list(batch_plan(ids,seq,BATCH))
        rng.shuffle(groups)
        totals=torch.zeros(4,device='cuda')
        for active in groups:
            step+=1
            token,mask=batch(active,seq,tok.pad_token_id)
            y=targets[active]
            teach=valid[active][:,None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                legacy,view=encode(net,token,mask,side[active])
                guess=added(readout,legacy,view)
                fit=F.binary_cross_entropy_with_logits(guess,y,pos_weight=weight,reduction='none')
                usual=fit[:,0].mean()
                source=(fit[:,1:]*teach).sum(0)/teach.sum().clamp_min(1.)
                drift=F.smooth_l1_loss(legacy.clamp(-16.,16.),anchor[active].float(),beta=.5,reduction='none')
                # Preserve the main output and every logit used by the existing routing policy.
                preserve=drift[:,[0,2,22,23]].mean()+.18*drift.mean()
                loss=usual+source@aux+KEEP*preserve
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*readout.parameters(),*updated],.85)
            optimizer.step()
            scheduler.step()
            totals+=torch.cat((usual.detach()[None],source.detach(),preserve.detach()[None]))
            if step%600==0:
                print(rotation,mode,epoch,step,total,round(perf_counter()-started,2),flush=True)
        history.append((totals/len(groups)).tolist())
    assert step==total
    state=dict(backbone={k:v.detach().cpu() for k,v in net.named_parameters() if v.requires_grad},
               head=readout.cpu().state_dict())
    return state,dict(seconds=perf_counter()-started,steps=step,loss=history,backbone_tensors=len(state['backbone']))


def main(rotation):
    ids,texts,target,usable,source=prepare(rotation)
    tok=f.small.tokenizer()
    token=full_context(texts,tok)
    side=torch.as_tensor(lexical_features(texts),device='cuda')
    model,checkpoint=release_start(rotation,tok)
    late(model,'frozen_union')
    starting=cache_base(model,tok,token,side)
    del model
    y=torch.as_tensor(target,device='cuda')
    valid=torch.as_tensor(usable,device='cuda')
    rates=np.array([len(ids),int(usable.sum()),int(usable.sum())])
    probability=(rates-target.sum(0))/np.maximum(1.,target.sum(0))
    weight=torch.as_tensor(np.sqrt(probability).clip(1.,[2.,3.,12.]),device='cuda',dtype=torch.float32)
    folder=DEST/rotation
    folder.mkdir(exist_ok=True,parents=True)
    roster=folder/'rows.npz'
    np.savez_compressed(roster,ids=ids,component_available=usable,targets=target.astype(bool))
    manifest=folder/'fit.json'
    receipt=dict(config=dict(seed=SEED,epochs=EPOCHS,batch_size=BATCH,hidden_width=HEAD_WIDTH,
                             regularizer=KEEP,component_weights=COMPONENT_WEIGHTS,
                             samples=len(y),observed=int(usable.sum()),positive=target.sum(0).astype(int).tolist(),
                             weights=weight.tolist(),base_checkpoint_sha256=checkpoint,
                             roster_sha256=checksum(roster),code_sha256=checksum(Path(__file__)),sources=source),
                 arms={})
    if manifest.exists():
        previous=json_at(manifest)
        assert json_at_dict(receipt['config'])==previous['config']
        receipt=previous
    for name in ARMS:
        artifact=folder/(name+'.pt')
        if name in receipt['arms'] and checksum(artifact)==receipt['arms'][name]['sha256']:
            continue
        state,elapsed=optimize(rotation,name,tok,token,side,y,valid,weight,starting)
        pending=artifact.with_suffix('.partial')
        torch.save(state,pending)
        pending.replace(artifact)
        receipt['arms'][name]=elapsed|dict(sha256=checksum(artifact))
        write_json(manifest,receipt)
        print(rotation,name,receipt['arms'][name],flush=True)


def json_at_dict(value):
    import json
    return json.loads(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
