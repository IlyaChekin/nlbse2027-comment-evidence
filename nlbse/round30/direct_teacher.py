"""Measure the extra cost and transfer of the conditional teacher when called directly."""
import argparse
from itertools import product
from statistics import median
import time

import joblib
import numpy as np
import torch

from teacher_probe import HERE, PARTS, f, checksum, json_at, write_json
from embed import tokens
from reference29 import reference, match, strict_cut
from disagreement import trained_logistic
from students import load_model, lexical_features, batch_plan, batch, ordered_batches, batch_gflops

DEST=HERE/'direct'
WIDTHS=(.25,.5,1.,1.75)


@torch.inference_mode()
def score(net,tok,texts,balance):
    p=np.empty(len(texts),dtype=np.float32)
    wall=time.perf_counter()
    forward=0.
    volume=0.
    def prepare(first):
        local=texts[first:first+2048]
        return tokens(local,tok),lexical_features(local)
    with torch.autocast('cuda',dtype=torch.float16):
        for i,(seq,lex) in ordered_batches(range(0,len(texts),2048),prepare,True):
            side=torch.as_tensor(lex,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,96):
                x,mask=batch(group,seq,tok.pad_token_id)
                start=time.perf_counter()
                p[i+np.asarray(group)]=net(x,mask,side[group])[:,0].float().cpu().numpy()-np.log(balance)
                forward+=time.perf_counter()-start
                volume+=batch_gflops('codebert',net,*x.shape)
    return p,dict(seconds=forward,gflops=volume,wall=time.perf_counter()-wall)


def run(rotation):
    d,t,fit,cal,review,base,target,p,q=reference(rotation)
    teacher=HERE/'tuned_teacher'/rotation
    fit_info=json_at(teacher/'fit.json')
    checkpoint=teacher/'upper_0.pt'
    if checksum(checkpoint)!=fit_info['folds']['0']['sha256']:
        raise ValueError('Stored independent model weights changed')
    model,tok=load_model('unixcoder9')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    needed=model.load_state_dict(saved,strict=False)
    if needed.unexpected_keys:
        raise ValueError('Teacher upper layers do not match the encoder')
    model.eval()
    b=fit_info['folds']['0']['balance']
    dp=[s for s,k in zip(d.texts,d.active,strict=True) if k]
    tp=dp if t is d else [s for s,k in zip(t.texts,t.active,strict=True) if k]
    a,_=score(model,tok,dp,b)
    c=a if t is d else score(model,tok,tp,b)[0]
    xd=np.column_stack((p[d.active],d.inputs[d.active,:3],a))
    xt=np.column_stack((q[t.active],t.inputs[t.active,:3],c))
    dy,ty=d.frame['y'],t.frame['y']
    use=fit&d.active
    readout=trained_logistic(xd[fit[d.active]],dy[use],c=.7)
    rd=readout.decision_function(xd)
    rt=readout.decision_function(xt)
    held=np.ones(len(ty),dtype=bool) if review is None else review
    folder=DEST/rotation
    folder.mkdir(parents=True,exist_ok=True)
    info={}
    forecasts=dict(ids=t.frame['ids'][held],truth=ty[held],base=target[held])
    for width in WIDTHS:
        where=d.active&(abs(p)<width)
        query=t.active&(abs(q)<width)
        y=dy[where&cal]
        if not len(y):
            continue
        cost={}
        if review is not None:
            sample=[s for s,k in zip(d.texts,where&cal,strict=True) if k]
            score(model,tok,sample,b)
            runs=[score(model,tok,sample,b)[1] for _ in range(3)]
            cost={k:median(t[k] for t in runs) for k in runs[0]}
        column=rd[where[d.active]]
        column2=rt[query[t.active]]
        for amount,scope in product((.35,.7),('exchange','promote')):
            u=(1.-amount)*p[where]+amount*column
            v=(1.-amount)*q[query]+amount*column2
            if scope=='promote':
                u=np.where(base[where],p[where],np.maximum(u,p[where]))
                v=np.where(target[query],q[query],np.maximum(v,q[query]))
            name=f'{width:g}_{amount:g}_{scope}'
            row=dict(cost=cost,cal_calls=int((where&cal).sum()),held_calls=int((query&held).sum()),cuts={})
            if review is None:
                adj,local=match(ty,t.frame['project'],target,query,v)
                row.update(matched=f.comparison(ty,target,adj),projects=local)
                forecasts[name+'_matched']=adj
            for label in ('fp','tp','fixed'):
                cut=strict_cut(y,base[where&cal],u[cal[where]],label)
                p2,q2=base.copy(),target.copy()
                p2[where],q2[query]=u>cut,v>cut
                found=f.comparison(dy[cal],base[cal],p2[cal])
                record=dict(cut=cut,cal=found,frozen=f.comparison(ty[held],target[held],q2[held]))
                if cost:
                    debit=.2*f.PUBLIC_N/cal.sum()*(cost['seconds']/1000+cost['gflops']/3e6)
                    record['proxy']=.3*(found['f1_after']-found['f1_before'])-debit
                row['cuts'][label]=record
                forecasts[name+'_'+label]=q2[held]
            info[name]=row
    file=folder/'predictions.npz'
    np.savez_compressed(file,**forecasts)
    path=folder/'readout.joblib'
    joblib.dump(readout,path)
    write_json(folder/'results.json',dict(options=info,head_sha256=checksum(path),teacher_sha256=checksum(checkpoint),
                     predictions_sha256=checksum(file)))
    print(rotation,len(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=PARTS)
    run(p.parse_args().part)
