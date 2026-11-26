"""Transfer the teacher's change in log odds while preserving the base score."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from distill import HERE, OUT, DEST, PARTS, load_target, f, checksum, write_json
from disagreement import clip_logit
from student_runtime import head, stack

MODES=('delta','delta_selected')


def fit(rotation):
    sources=[load_target(path,rotation) for path in (DEST,HERE/'tuned_teacher')]
    (a,older),(z,teacher)=sources
    if not all(np.array_equal(a[k],z[k]) for k in ('ids','y','fold')):
        raise ValueError('The file-disjoint teachers disagree on the fit cohort')
    y=a['y']
    ids=a['ids']
    before=clip_logit(a['calibrated'][:,3])
    after=clip_logit(z['calibrated'][:,4])
    desired=np.clip(after-before,-1.5,1.5)
    better=np.sign(desired)==(2*y.astype(np.int32)-1)
    corpus,_,_,_,_=f.inputs()
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    x,_=stack(model,tok,[corpus[i] for i in ids])
    destination=OUT/rotation
    result={}
    for name in MODES:
        truth=desired if name=='delta' else np.where(better,desired,0.)
        t=torch.as_tensor(truth,device='cuda',dtype=torch.float32)
        importance=.25+1.25*np.exp(-np.abs(before-clip_logit(np.asarray(.18))))
        scale=torch.as_tensor(importance,device='cuda',dtype=torch.float32)
        torch.manual_seed(127)
        r=head(model.width).cuda()
        opt=torch.optim.AdamW(r.parameters(),lr=.0012,weight_decay=.02)
        t0=time.perf_counter()
        loss_log=[]
        for epoch in range(4):
            r.train()
            batches=torch.randperm(len(ids),device='cuda').split(512)
            epoch_loss=t.new_tensor(0.)
            for selected in batches:
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    q=r(x[selected])[:,0].float()
                    term=F.smooth_l1_loss(q,t[selected],beta=.25,reduction='none')
                    loss=(term*scale[selected]).mean()+.025*q.square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(r.parameters(),.8)
                opt.step()
                epoch_loss+=loss.detach()
            loss_log.append(float(epoch_loss/len(batches)))
        path=destination/f'{name}.pt'
        torch.save(r.cpu().state_dict(),path)
        result[name]=dict(seconds=time.perf_counter()-t0,sha256=checksum(path),loss=loss_log)
        print(rotation,name,result[name],flush=True)
    write_json(destination/'delta.json',dict(mode=result,rows=len(ids),helpful=int(better.sum()),
                source=older['predictions_sha256'],teacher=teacher['predictions_sha256'],code_sha256=checksum(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    fit(p.parse_args().part)
