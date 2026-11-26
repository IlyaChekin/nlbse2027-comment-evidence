"""Batch uncertain examples across a full input block without recomputing the prefix."""

import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from fused_attention import attention_bias
from hybrid import (ACTIONS, HERE, EARLY, LATE_COST, PREFIX_COST, Pipeline, as_common_scale,
                    benchmark, chunks, finish_fused, infer, lexical_features, ordered_batches,
                    pack, pool, prefix_fused, use_inputs)
from windows import BLOCK


class PackedPipeline(Pipeline):
    def __init__(self,rules=None,batcher=chunks):
        super().__init__()
        self.batcher=batcher
        if rules:
            self.gate["options"]=json.loads(rules.read_text())["options"]
        self.parameters={k:torch.as_tensor(v,device="cuda") for k,v in self.student.items()}
        self.cut=torch.as_tensor(self.thresholds,device="cuda",dtype=torch.float32)
        self.late_cost=torch.as_tensor(LATE_COST,device="cuda",dtype=torch.float32)
        self.router=[torch.as_tensor(z,device="cuda") for z in (self.shift,self.scale,self.weight,self.bias)]

    def prefix_scores(self, pooled, native, side, width, rule):
        params=self.parameters
        with torch.autocast("cuda",enabled=False):
            tags=torch.as_tensor(side,device="cuda")
            x=torch.cat((pooled.float(),native.float(),tags),dim=1).half().float()
            scaled=((x-params["center"])/params["scale"]).clamp(-10,10)
            first=F.linear(scaled,params["weight0"],params["bias0"]).relu()
            p=F.linear(first,params["weight1"],params["bias1"]).sigmoid()
            if rule["kind"]=="margin":
                action=((p[:,0]-self.cut[0]).abs()<rule["vulnerability_margin"]).int()*2
            elif rule["kind"]=="asymmetric":
                action=((p[:,0]>rule["lower"])&(p[:,0]<rule["upper"])).int()*rule["late"]
            else:
                offsets,scale,weight,bias=self.router
                raw=torch.logit(p.clamp(1e-6,1-1e-6))
                margin=raw-torch.logit(self.cut)
                features=torch.cat((raw,margin,margin.abs(),p*(1-p),x[:,-7:],x[:,-1:]*margin.abs()),dim=1)
                gain=F.linear(((features-offsets)/scale).clamp(-10,10),weight,bias)
                prices=self.late_cost[torch.as_tensor(width,device="cuda")]
                gain=torch.cat((torch.zeros_like(gain[:,:1]),gain),dim=1)
                action=(gain-rule["penalty"]*prices/rule["unit"]).argmax(dim=1)
            return torch.column_stack((p,action))

    @torch.inference_mode()
    def adaptive(self,texts,rule):
        start=time.perf_counter()
        stack=self.model.encoder.encoder
        output=np.empty((len(texts),2),dtype=np.float32)
        n_actions=np.zeros(3,dtype=np.int64)
        total,elapsed=0.,0.
        def prepare_at(at):
            local=texts[at:at+BLOCK]
            return use_inputs(local,self.tokenizer,wide=True),lexical_features(local)

        with torch.autocast("cuda",dtype=torch.float16):
            for at,(data,context) in ordered_batches(range(0,len(texts),BLOCK),prepare_at,True):
                seq=data.views["head"]
                sizes=np.asarray([len(row) for row in seq])
                h=[None]*len(seq)
                parts, order=[], []
                compute_start=time.perf_counter()
                pack_seconds=0.
                for rows in self.batcher(seq,range(len(seq))):
                    indices=np.asarray(rows)
                    t=time.perf_counter()
                    x,mask=pack(seq,rows,data.pad,"cuda")
                    pack_seconds+=time.perf_counter()-t
                    states,_=prefix_fused(self.model,x,mask,EARLY,self.cache)
                    pooled=pool(stack,states,mask)
                    parts.append(self.prefix_scores(pooled,self.model.head(pooled),context[indices],sizes[indices],rule))
                    order.extend(rows)
                    for j,i in enumerate(rows):
                        h[i]=states[j,:sizes[i]]
                    total+=len(rows)*PREFIX_COST[x.shape[1]]
                totals=len(seq)*2*(self.parameters["weight0"].numel()+self.parameters["weight1"].numel())/1e9
                total+=totals
                packed=torch.cat(parts,dim=0).cpu().numpy()
                aligned=np.empty_like(packed)
                aligned[order]=packed
                options=aligned[:,2].astype(int)
                n_actions+=np.bincount(options,minlength=3)
                pb=as_common_scale(aligned[:,:2],self.thresholds)
                for j,name in enumerate(ACTIONS,1):
                    policy,head,cut=self.actions[name]
                    for rows in self.batcher(seq,np.flatnonzero(options==j)):
                        state=pad_sequence([h[i] for i in rows],batch_first=True)
                        width=state.shape[1]
                        length=torch.as_tensor(sizes[rows],device="cuda")
                        mask=(torch.arange(width,device="cuda")[None,:]<length[:,None]).long()
                        bias=attention_bias(stack,mask,self.cache)
                        logits=finish_fused(self.model,state,bias,mask,policy,EARLY).float().cpu().numpy()
                        pb[rows]=as_common_scale(head(np.column_stack((logits,context[rows]))),cut)
                        total+=len(rows)*LATE_COST[width,j]
                elapsed+=time.perf_counter()-compute_start-pack_seconds
                output[at:at+len(seq)]=pb
        return output,dict(wall_seconds=time.perf_counter()-start,forward_seconds=elapsed,
                           analytic_gflops=total,actions=n_actions.tolist())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest="stage",required=True)
    cmd=modes.add_parser("benchmark")
    cmd.add_argument("--output",type=Path,default=HERE/"packed")
    cmd.add_argument("--repeat",type=int,default=3)
    cmd.add_argument("--rules",type=Path)
    cli=modes.add_parser("predict")
    cli.add_argument("--input",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--mode",default="score")
    cli.add_argument("--rules",type=Path)
    args=parser.parse_args()
    runner=partial(PackedPipeline,rules=args.rules)
    if args.stage=="benchmark":
        benchmark(args.output,args.repeat,runner)
    else:
        infer(args.input,args.output,args.mode,runner)
