"""Distill the adviser into a small residual head without a second encoder call."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round22"))
from frozen_head import extract
from project_eval import f,PARTS,calibrate,evaluate
from train import SEED,cohorts,fit_ids,open_student,checksum,json_at,write_json

R22=ROOT/"round22"
MODES=("truth_head","advised_head")
WIDTH=64
SCALE=.50


def small_head(width):
    model=nn.Sequential(nn.Linear(width,WIDTH),nn.GELU(),nn.Linear(WIDTH,1))
    nn.init.zeros_(model[-1].bias)
    nn.init.zeros_(model[-1].weight)
    return model


class Adapter(nn.Module):
    def __init__(self,head,residual):
        super().__init__()
        self.head=head
        self.residual=residual

    def forward(self,pooled):
        x=self.head[0](pooled)
        output=self.head[1](x)
        delta=self.residual(x)[:,0]*SCALE
        output[:,20]=output[:,20]+delta
        return output


def fit(rotation):
    text,labels,teacher,_,source=f.base.partition(rotation)
    weak,y,cohort=cohorts()
    ids=fit_ids(rotation)
    if not np.array_equal(y[ids],labels[:,0]):
        raise ValueError("Negative head saw misaligned teacher labels")
    include=weak[ids]==0
    text=[s for s,k in zip(text,include,strict=True) if k]
    ys=torch.as_tensor(labels[include,0],device="cuda",dtype=torch.float32)
    adviser=torch.as_tensor(teacher[include,0],device="cuda",dtype=torch.float32).clamp(1e-5,1-1e-5)
    adviser=(adviser.logit()/1.7).sigmoid()
    tok=f.small.tokenizer()
    net=open_student(rotation,SEED,"factor",tok)
    started=time.perf_counter()
    features=extract(net,f.tokenize(text,tok),tok)
    main=net.rule_head[1]
    logits=(features.float() @ main.weight[20].detach()+main.bias[20].detach()).float()
    weight=((len(ys)-ys.sum())/ys.sum()).sqrt().clamp(1,6)
    folder=HERE/"latent_expert"/rotation
    folder.mkdir(parents=True,exist_ok=True)
    receipts={}
    print(rotation,"features",len(ys),round(time.perf_counter()-started,2),flush=True)
    for mode in MODES:
        torch.manual_seed(SEED)
        model=small_head(features.shape[1]).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.02)
        for epoch in range(8):
            for rows in torch.randperm(len(features),device="cuda").split(512):
                correction=model(features[rows].float()).flatten()
                p=logits[rows]+SCALE*correction
                loss=F.binary_cross_entropy_with_logits(p,ys[rows],pos_weight=weight)
                if mode=="advised_head":
                    loss=.55*loss+.45*1.7**2*F.binary_cross_entropy_with_logits(p/1.7,adviser[rows])
                loss+=.035*correction.square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        path=folder/f"{mode}.pt"
        torch.save(model.state_dict(),path)
        receipts[mode]=dict(checkpoint=path.name,sha256=checksum(path))
    write_json(folder/"fit.json",dict(modes=receipts,rows=len(ys),positive=int(ys.sum()),
                                      cohort_sha256=cohort,teacher_digests=source,rank=WIDTH,
                                      mix=SCALE,seeds=[SEED],code_sha256=checksum(Path(__file__))))


def collect(rotation):
    folder=HERE/"latent_expert"/rotation
    tok=f.small.tokenizer()
    text,_,_,_,_=f.inputs()
    data={}
    for role in ("dev","test"):
        with np.load(R22/rotation/f"{role}.npz") as z:
            data[role]=dict(z)
    for mode in MODES:
        old=open_student(rotation,SEED,"factor",tok)
        found=json_at(folder/"fit.json")["modes"][mode]
        checkpoint=folder/found["checkpoint"]
        if checksum(checkpoint)!=found["sha256"]:
            raise ValueError("Stored residual changed")
        residual=small_head(2*old.width).cuda().eval()
        residual.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
        old.rule_head=Adapter(old.rule_head,residual).eval()
        pipe=SimpleNamespace(encoder=old,tok=tok,mode="control")
        for x in data.values():
            q,a,cost=f.infer_student(pipe,[text[i] for i in x["ids"]],f.pinned()[0],columns=22,return_all=True)
            candidate=np.column_stack((q[:,0],a[:,0],a[:,-2:]))
            if not np.array_equal(candidate[:,[0,1,3]],x["reference"][:,[0,1,3]]):
                raise ValueError("The learned correction changed another head")
            # One width-64 hidden projection and output dot product per source row.
            total=cost["analytic_gflops"]+2*len(q)*WIDTH*(2*old.width+1)/1e9
            x[mode]=candidate
            x[mode+"_cost"]=np.asarray([cost["forward_seconds"],total])
    for k,frame in data.items():
        np.savez_compressed(folder/f"{k}.npz",**frame)
    evaluate(rotation,MODES,folder=folder)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    p.add_argument("stage",choices=("fit","collect"))
    args=p.parse_args()
    {"fit":fit,"collect":collect}[args.stage](args.rotation)
