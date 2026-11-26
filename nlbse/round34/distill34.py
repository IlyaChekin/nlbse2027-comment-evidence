"""Train small regional readouts using the frozen compact encoder and OOF teacher scores."""
import argparse
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from sections34 import HERE, PARTS, f, checksum, json_at, write_json
from eval34 import loaded
from compact34 import ARMS, head, cache

DEST=HERE/'distilled'
EPOCHS=12
BATCH=192
SEED=2042


def fit(x,y,target,margin,mode,width):
    torch.manual_seed(SEED)
    size,count=ARMS[mode]
    net=head(width,mode).cuda().train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.025,fused=True)
    weight=(len(y)/y.sum()-1.).sqrt().clamp(1.,4.)
    start=perf_counter()
    best=[]
    for epoch in range(EPOCHS):
        totals=torch.zeros(3,device='cuda')
        order=torch.randperm(len(y),device='cuda')
        for group in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw=net(x[group,:size*width]).float()
            w=.25+1.50*(-.75*margin[group].abs()).exp()
            hard=(F.binary_cross_entropy_with_logits(raw[:,0],y[group],pos_weight=weight,reduction='none')*w).mean()
            soft=raw.new_zeros(())
            if count>1:
                expected=target[group,:count-1]
                soft=(F.smooth_l1_loss(raw[:,1:],expected,reduction='none').mean(1)*w).mean()
            loss=hard+.25*soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),.9)
            optimizer.step()
            totals+=torch.stack((loss.detach(),hard.detach(),soft.detach()))
        best.append((totals/len(order.split(BATCH))).tolist())
    return net.eval(),dict(seconds=perf_counter()-start,epoch_losses=best,outputs=count)


def main(rotation):
    data,proof,region=loaded(rotation)
    samples=f.inputs()[0]
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    point={}
    start=perf_counter()
    for role,rows in data.items():
        select=rows['support']
        ix=rows['ids'][select]
        inputs=[samples[i] for i in ix]
        point[role]=(ix,cache(model,tok,inputs))
        print(rotation,role,'compact states',len(ix),round(perf_counter()-start,3),flush=True)
    where=data['oof']['support']
    fit_data=data['oof']
    expected=torch.as_tensor(fit_data['teacher'][where],device='cuda',dtype=torch.float32).clamp(-8.,8.)
    gold=torch.as_tensor(fit_data['y'][where],device='cuda',dtype=torch.float32)
    q=torch.as_tensor(fit_data['queried'][where],device='cuda',dtype=torch.float32)
    dest=DEST/rotation
    dest.mkdir(parents=True,exist_ok=True)
    archived={}
    receipts={}
    for mode,args in ARMS.items():
        fitted,receipt=fit(point['oof'][1],gold,expected,q,mode,model.width)
        path=dest/f'{mode}.pt'
        torch.save(fitted.cpu().state_dict(),path)
        receipts[mode]=receipt|{'sha256':checksum(path)}
        fitted.cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for role,(ids,emb) in point.items():
                columns=args[0]*model.width
                out=[fitted(x[:,:columns]).float().cpu().numpy() for x in emb.split(2048)]
                archived[role+'_ids']=ids
                archived[role+'_'+mode]=np.concatenate(out)
        print(rotation,mode,round(receipt['seconds'],3),flush=True)
    score=dest/'scores.npz'
    np.savez_compressed(score,**archived)
    root=f.HERE/rotation/str(f.SEED)
    limits=json_at(root/'fit.json')['modes']['factor']
    assert checksum(root/limits['file'])==limits['sha256']
    write_json(dest/'receipt.json',dict(rows=int(where.sum()),epochs=EPOCHS,seed=SEED,batch=BATCH,
                    code_sha256=checksum(HERE/'distill34.py'),runtime_sha256=checksum(HERE/'compact34.py'),
                    regional_teacher_sha256=region['predictions_sha256'],phase_sources=proof['phases'],
                    compact_sha256=limits['sha256'],heads=receipts,predictions_sha256=checksum(score)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    main(parser.parse_args().rotation)
