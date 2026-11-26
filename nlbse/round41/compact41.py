"""Read existing pooled compact states and align file-fold checkpoints without labels."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from source41 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, load, supported
sys.path.insert(0,str(ROOT/'round37'))
from section37 import oof_state
from predict_factor import full_context, ordered_batches, batch, batch_plan, small
from rule_model import pooling

DEST=HERE/'compact'
BATCHING=2048
ANCHORS=2600
RIDGE=3.5


@torch.inference_mode()
def encode(text,tok,model):
    linears=np.empty((len(text),2*model.width),np.float32)
    def prepare(start):
        return full_context(text[start:start+BATCHING],tok)
    with torch.autocast('cuda',dtype=torch.float16):
        for start,data in ordered_batches(range(0,len(text),BATCHING),prepare,True):
            for take in batch_plan(list(range(len(data))),data,small.BATCH):
                ids,mask=batch(take,data,tok.pad_token_id)
                u=model.encoder.token_states(ids,mask)
                v=model.rule_head[0](pooling(u,mask)).half().float()
                linears[start+np.asarray(take)]=v.cpu().numpy()
    return linears


def align(x,y):
    a,b=[torch.as_tensor(z,device='cuda',dtype=torch.float32) for z in (x,y)]
    means=[z.mean(0) for z in (a,b)]
    core=a-means[0]
    shape=b-means[1]
    I=torch.eye(core.shape[1],device='cuda')
    beta=torch.linalg.solve(core.T@core+RIDGE*I,core.T@shape+RIDGE*I)
    bias=means[1]-means[0]@beta
    return beta.cpu().numpy(),bias.cpu().numpy()


def main(rotation):
    frames,_,proof=load(rotation)
    ids=frames['dev']['ids'][frames['dev']['fit']]
    if len(ids)<128:
        raise ValueError('Insufficient unsupervised alignment anchors')
    rng=np.random.default_rng(4101)
    anchor=rng.permutation(ids)[:ANCHORS]
    groups=np.split(anchor,[round(len(anchor)*.8)])
    initial=frames['oof']
    assert not np.isin(anchor,initial['ids']).any()
    assert not (frames['dev']['cal']&frames['dev']['fit']).any()
    assert not (frames['dev']['held']&frames['dev']['fit']).any() or rotation!='official'
    names=f.inputs()[0]
    token=small.tokenizer()
    orig=f.pipe(rotation,token).encoder
    samples=[names[i] for i in anchor]
    sample_out=encode(samples,token,orig)
    cut=len(groups[0])
    src=ROOT/'round26/compact'/rotation
    capsule=json_at(src/'manifest.json')
    reference=src/'scores.npz'
    assert checksum(reference)==capsule['sha256']
    with np.load(reference) as z:
        assert np.array_equal(z['ids'],initial['ids'])
        assert np.array_equal(z['fold'],initial['fold'])
        assert np.array_equal(z['y'],initial['y'])
    proj={}
    history={}
    bank={}
    use=supported(initial)
    ids=initial['ids'][use]
    subsets=initial['fold'][use]
    output=np.empty((len(ids),sample_out.shape[1]),np.float16)
    started=perf_counter()
    for fold in np.unique(subsets):
        model=oof_state(src,token,capsule,int(fold))
        raw=encode(samples,token,model)
        beta,bias=align(raw[:cut],sample_out[:cut])
        before=np.square(raw[cut:]-sample_out[cut:]).mean()
        after=np.square(raw[cut:]@beta+bias-sample_out[cut:]).mean()
        keep=subsets==fold
        family=encode([names[i] for i in ids[keep]],token,model)
        output[keep]=(family@beta+bias).astype(np.float16)
        key=f'fold_{int(fold)}'
        proj[key+'_weight']=beta
        proj[key+'_bias']=bias
        history[int(fold)]=dict(rows=int(keep.sum()),mse_before=float(before),mse_after=float(after),
                               encoder_sha256=capsule['folds'][str(int(fold))]['sha256'])
        print(rotation,key,history[int(fold)],round(perf_counter()-started,2),flush=True)
        del model
    bank.update(oof_ids=ids,oof=output,fold=subsets)
    for name,rows in frames.items():
        if name!='oof':
            ids=rows['ids'][supported(rows)]
            bank[name+'_ids']=ids
            bank[name]=encode([names[i] for i in ids],token,orig).astype(np.float16)
    out=DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    data=out/'states.npz'
    maps=out/'alignment.npz'
    np.savez_compressed(data,**bank)
    np.savez_compressed(maps,anchor_fit=groups[0],anchor_check=groups[1],**proj)
    ref=dict(rotation=rotation,folds=history,matrix_sha256=proof,
             state_sha256=checksum(data),alignment_sha256=checksum(maps),
             code_sha256=checksum(Path(__file__)),model_receipt_sha256=checksum(src/'manifest.json'),
             seconds=perf_counter()-started,regularization=RIDGE)
    write_json(out/'receipt.json',ref)
    print(rotation,'done',ref['seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    main(parser.parse_args().rotation)
