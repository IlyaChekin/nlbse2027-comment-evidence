"""Reuse the fused four-block prefix and route to selective late token updates."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
from scipy.special import expit
import torch

from cache_hybrid import ACTIONS, TOKEN_CONFIG
from fused_attention import (HERE, PORTABLE, EARLY, DeferredOffsets, JointHead, Policy,
                             as_common_scale, dataset, finish_fused, lexical_features,
                             load_checkpoint, ordered_batches, pool, prefix_fused,
                             predict as fixed_predict, save, use_inputs)
from tune_hybrid import apply_rule, costs, features_for_router, normalized
from analyze import changes, group_codes, predict_bits
from common import sha256
from pipeline import PUBLIC, load, write_json
from score_runtime import competition_formula
from round2_report import bootstrap_delta
from windows import BLOCK, chunks, pack

PREFIX_COST, LATE_COST = costs(np.arange(769))
REFERENCE = "query_saliency_50"


class Pipeline:
    def __init__(self, device="cuda"):
        self.model, tok = load_checkpoint(PORTABLE,device)
        self.tokenizer=DeferredOffsets(tok)
        self.tokens=json.loads(TOKEN_CONFIG.read_text())
        self.configurations=self.tokens["configurations"]
        if self.tokens["encoder_sha256"] != sha256(PORTABLE/"model.safetensors"):
            raise ValueError("The token head refers to a different encoder")
        self.gate=json.loads((HERE/"frozen/policy.json").read_text())
        if self.gate["token_config_sha256"]!=sha256(TOKEN_CONFIG):
            raise ValueError("The gate was fitted with a different late readout")
        with np.load(HERE/"frozen"/(self.gate["student"]+".npz")) as stored:
            self.student=dict(stored)
        self.thresholds=np.asarray(self.gate["thresholds"])
        view=self.gate["gate"]
        self.shift,self.scale,self.weight,self.bias=(
            np.asarray(view[k],dtype=np.float32) for k in ("shift","scale","weight","bias"))
        self.actions={k:(Policy(**self.configurations[k]["spec"]),
                         JointHead.from_dict(self.configurations[k]["head"]),
                         self.configurations[k]["thresholds"]) for k in ACTIONS}
        self.cache={}

    @property
    def methods(self):
        return ["fused50",*self.gate["options"]]

    def predict(self,texts,mode):
        if mode=="fused50":
            p,cost=fixed_predict(self.model,self.tokenizer,texts,self.configurations[REFERENCE],
                                 REFERENCE,True,self.cache)
            return p,dict(cost,actions=[0,len(texts),0])
        return self.adaptive(texts,self.gate["options"][mode]["rule"])

    @torch.inference_mode()
    def adaptive(self, texts, rule):
        model,tokenizer=self.model,self.tokenizer
        stack=model.encoder.encoder
        p=np.empty((len(texts),2),dtype=np.float32)
        start=time.perf_counter()
        elapsed,volume=0.,0.
        counts=np.zeros(3,dtype=np.int64)
        student=self.student

        def prepare_at(at):
            block=texts[at:at+BLOCK]
            return use_inputs(block,tokenizer,wide=True), lexical_features(block)

        with torch.autocast("cuda",dtype=torch.float16):
            for at,(data,side) in ordered_batches(range(0,len(texts),BLOCK),prepare_at,True):
                sequences=data.views["head"]
                size=np.asarray([len(value) for value in sequences])
                for group in chunks(sequences,range(len(sequences))):
                    group=np.asarray(group)
                    x,mask=pack(sequences,group,data.pad,"cuda")
                    tick=time.perf_counter()
                    states,bias=prefix_fused(model,x,mask,EARLY,self.cache)
                    pooled=pool(stack,states,mask)
                    native=model.head(pooled)
                    shape=torch.cat((pooled.float(),native.float()),dim=-1).half().cpu().numpy()
                    features=np.column_stack((shape,side[group].astype(np.float16))).astype(np.float32)
                    reading=normalized(features,student["center"],student["scale"])
                    scores=expit(np.maximum(reading@student["weight0"].T+student["bias0"],0.)
                                 @student["weight1"].T+student["bias1"])
                    gate_x=normalized(features_for_router(features,scores,self.thresholds),self.shift,self.scale)
                    gains=np.column_stack((np.zeros(len(group)),gate_x@self.weight.T+self.bias))
                    act=apply_rule(rule,scores,self.thresholds,gains,LATE_COST[size[group]])
                    counts+=np.bincount(act,minlength=3)
                    decision=as_common_scale(scores,self.thresholds)
                    volume+=len(group)*PREFIX_COST[x.shape[1]]
                    volume+=len(group)*2*(student["weight0"].size+student["weight1"].size)/1e9
                    for action,name in enumerate(ACTIONS,1):
                        selected=np.flatnonzero(act==action)
                        if not len(selected):
                            continue
                        ii=torch.as_tensor(selected,device="cuda")
                        width=int(max(size[group[selected]]))
                        policy,head,cut=self.actions[name]
                        logits=finish_fused(model,states[ii,:width],bias[ii,:,:width,:width],
                                            mask[ii,:width],policy,EARLY).float().cpu().numpy()
                        prob=head(np.column_stack((logits,side[group[selected]])))
                        decision[selected]=as_common_scale(prob,cut)
                        volume+=len(selected)*LATE_COST[width,action]
                    elapsed+=time.perf_counter()-tick
                    p[at+group]=decision
        return p,dict(wall_seconds=time.perf_counter()-start,forward_seconds=elapsed,
                      analytic_gflops=volume,actions=counts.tolist())


def benchmark(output,repeats,pipeline=Pipeline):
    proc=pipeline()
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    previous=predict_bits(HERE.parent/"round8/tokens_final"/REFERENCE/"predictions.csv",ids,{"thresholds":(.5,.5)})
    predictions={}
    history={name:[] for name in proc.methods}
    for mode in proc.methods:
        proc.predict(texts[:2048],mode)
    for rep in range(repeats):
        for mode in proc.methods if rep%2==0 else proc.methods[::-1]:
            p,meta=proc.predict(texts,mode)
            if mode in predictions and not np.array_equal(p>.5,predictions[mode]>.5):
                raise ValueError(f"Non-deterministic ordered binary decisions for {mode}")
            predictions[mode]=p
            history[mode].append(meta)
            print(rep+1,mode,meta,flush=True)
    results={}
    for mode,p in predictions.items():
        a=history[mode]
        cost={field:median(x[field] for x in a) for field in ("wall_seconds","forward_seconds","analytic_gflops")}
        cost["actions"]=a[0]["actions"]
        save(output/mode,y,ids,cal,p,cost,(.5,.5))
        item=json.loads((output/mode/"metrics.json").read_text())
        item["local_formula"]=competition_formula(item["scores"]["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])
        item["paired_interval_95_vs_release"]=bootstrap_delta(y,previous,p>.5,groups)
        item["changes_vs_release"]=changes(y,previous,p>.5)
        results[mode]=item
    write_json(output/"analysis.json",dict(device=torch.cuda.get_device_name(),results=results,trials=history))


def infer(source,output,mode,pipeline=Pipeline):
    with source.open(encoding="utf-8-sig",newline="") as handle:
        r=csv.DictReader(handle)
        if "Function" not in (r.fieldnames or ()):
            raise ValueError("Input CSV needs a Function column")
        texts=[row["Function"] for row in r]
    if not texts or any(t is None for t in texts):
        raise ValueError("Missing function content")
    predictor=pipeline()
    if mode not in predictor.methods:
        raise ValueError(f"Choose one of {predictor.methods}")
    p,_=predictor.predict(texts,mode)
    with output.open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>.5).astype(int).tolist())


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_subparsers(dest="command",required=True)
    run=modes.add_parser("benchmark")
    run.add_argument("--output",type=Path,default=HERE/"public")
    run.add_argument("--repeat",type=int,default=3)
    cli=modes.add_parser("predict")
    cli.add_argument("--input",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--mode",default="fused50")
    args=ap.parse_args()
    if args.command=="benchmark":
        benchmark(args.output,args.repeat)
    else:
        infer(args.input,args.output,args.mode)
