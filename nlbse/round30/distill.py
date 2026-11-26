"""Compare hard targets, soft existing scores, and a file-disjoint alternative teacher."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from teacher_probe import HERE, ROOT, DEST, PARTS, f, checksum, json_at, write_json
from student_runtime import head, stack

OUT=HERE/'student'
STYLES=('hard','existing','frozen','tuned','tuned_selected')
BATCH=512
SEED=92


def load_target(folder, rotation):
    where=folder/rotation
    cfg=json_at(where/'fit.json')
    path=where/'oof.npz'
    if checksum(path)!=cfg['predictions_sha256']:
        raise ValueError('Distillation targets changed')
    with np.load(path) as z:
        return {k:z[k] for k in ('ids','y','fold','calibrated')},cfg


def train(rotation, mode):
    older,proof=load_target(DEST,rotation)
    ids,labels=older['ids'],older['y']
    baseline=older['calibrated'][:,3]
    enhanced=older['calibrated'][:,4]
    tunings=None
    if mode=='tuned':
        tunings,other=load_target(HERE/'tuned_teacher',rotation)
        if not all(np.array_equal(older[k],tunings[k]) for k in ('ids','y','fold')):
            raise ValueError('The teacher folds or labels changed')
        enhanced=tunings['calibrated'][:,4]
    lines,gold,_,_,_=f.inputs()
    assert np.array_equal(gold[ids,0],labels)
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    since=time.perf_counter()
    matrix,q0=stack(model,tok,[lines[i] for i in ids])
    elapsed=time.perf_counter()-since
    y=torch.as_tensor(labels,dtype=torch.float32,device='cuda')
    prob=torch.as_tensor(np.column_stack((baseline,enhanced)),dtype=torch.float32,device='cuda').clamp(.0001,.9999)
    hidden=prob.logit()
    soften=1.7
    targets=(hidden/soften).sigmoid()
    original=-(torch.where(y==1,prob[:,0],1.-prob[:,0])).log()
    amended=-(torch.where(y==1,prob[:,1],1.-prob[:,1])).log()
    benefit=((original-amended)*3).clamp(0.,1.).detach()
    ceiling=min(3.5,float(((len(y)-y.sum())/y.sum()).sqrt()))
    here=OUT/rotation
    here.mkdir(parents=True,exist_ok=True)
    results={}
    file=here/f'{mode}.json'
    for label in (STYLES[:3] if mode=='frozen' else STYLES[3:]):
        torch.manual_seed(SEED)
        rng=torch.Generator().manual_seed(426)
        r=head(model.width).cuda()
        opt=torch.optim.AdamW(r.parameters(),lr=.00125,weight_decay=.022)
        total=3*((len(ids)+BATCH-1)//BATCH)
        lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total,eta_min=.00015)
        stat=[]
        start=time.perf_counter()
        for j in range(3):
            r.train()
            order=torch.randperm(len(ids),generator=rng).split(BATCH)
            acc=torch.zeros(2,device='cuda')
            for batch in order:
                idx=batch.cuda()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    extra=r(matrix[idx])[:,0].float()
                    guess=q0[idx]+extra
                    true=F.binary_cross_entropy_with_logits(guess,y[idx],pos_weight=y.new_tensor(ceiling))
                    if label=='hard':
                        hint=guess.new_tensor(0.)
                    else:
                        group=0 if label=='existing' else 1
                        teach=F.binary_cross_entropy_with_logits(guess/soften,targets[idx,group],reduction='none')
                        if label=='tuned_selected':
                            active=benefit[idx]
                            teach=teach*active
                            hint=teach.sum()/active.sum().clamp_min(1.)
                        else:
                            hint=teach.mean()
                    loss=true+.028*extra.square().mean()+.42*soften**2*hint
                loss.backward()
                torch.nn.utils.clip_grad_norm_(r.parameters(),.8)
                opt.step()
                lr.step()
                acc+=torch.stack((true.detach(),hint.detach()))
            stat.append((acc/len(order)).tolist())
        pt=here/f'{label}.pt'
        torch.save(r.cpu().state_dict(),pt)
        results[label]=dict(seconds=time.perf_counter()-start,loss=stat,sha256=checksum(pt))
        print(rotation,label,results[label],flush=True)
    result=dict(rows=len(ids),positives=int(labels.sum()),encoding_seconds=elapsed,
                matched_teacher_sha256=proof['predictions_sha256'],mode=results,
                code_sha256=checksum(Path(__file__)),seed=SEED,width=model.width)
    if tunings is not None:
        result['tuned_sha256']=other['predictions_sha256']
        result['tuned_weight_positive']=int((benefit>0).sum())
    write_json(file,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    p.add_argument('source',choices=('frozen','tuned'))
    a=p.parse_args()
    train(a.part,a.source)
