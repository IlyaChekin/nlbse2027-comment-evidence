"""Test intermediate depth from the existing weights, with a grouped four-to-five exit."""

import argparse
from functools import partial
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round9"))
from fused_attention import (PORTABLE, DeferredOffsets, JointHead, attention_bias,
                             dataset, finish_fused, fused_block, lexical_features, load_checkpoint,
                             ordered_batches, padded_stride_bias, pool, prefix_fused, save, use_inputs)
from frontier import as_common_scale, calibrated, metric
from cache_hybrid import Policy
from packed_suffix import PackedPipeline
from length_batches import length_buckets
from compare import dense_gflops
from analyze import group_codes, predict_bits
from score_runtime import competition_formula
from pipeline import PUBLIC,load,write_json
from round2_report import bootstrap_delta
from windows import BLOCK,pack

FIXED=("prefix5","last5","prefix5_plus_half")
ROOT=HERE.parent


def late_logits(model,state,bias,mask,name):
    stack=model.encoder.encoder
    # Prefix positions 0–3 already ran with all input tokens.
    block=stack.block[5 if name=="last5" else 4]
    state=fused_block(block,state,padded_stride_bias(bias))
    if name=="prefix5_plus_half":
        return finish_fused(model,state,bias,mask,Policy("query","even",.5),5)
    return model.head(pool(stack,state,mask))


def volume(n,l,name):
    d,f=768,3072
    dense=n*(2*l*(4*d*d+2*d*f)+4*l*l*d)/1e9
    if name=="prefix5_plus_half":
        q=min(l,max(8,(l+1)//2))
        sparse=n*(4*(l+q)*d*d+4*q*d*f+4*l*q*d)/1e9
        return dense+sparse
    return dense


@torch.inference_mode()
def score(proc,texts,specs):
    stack=proc.model.encoder.encoder
    prefix,forward=0.,0.
    costs={k:0. for k in specs}
    p={key:np.empty((len(texts),2),dtype=np.float32) for key in specs}
    raw={key:np.empty((len(texts),7),dtype=np.float32) for key,v in specs.items() if v is None}
    widths=np.empty(len(texts),dtype=np.int32)
    total=time.perf_counter()
    bucket=partial(length_buckets,budget=8192)

    def prepare_at(start):
        subset=texts[start:start+BLOCK]
        return use_inputs(subset,proc.tokenizer,wide=True),lexical_features(subset)

    with torch.autocast("cuda",dtype=torch.float16):
        for at,(data,tags) in ordered_batches(range(0,len(texts),BLOCK),prepare_at,True):
            seq=data.views["head"]
            sizes=np.asarray([len(x) for x in seq])
            widths[at:at+len(seq)]=sizes
            state_rows=[None]*len(seq)
            paths={}
            start=time.perf_counter()
            for g in bucket(seq,range(len(seq))):
                x,mask=pack(seq,g,data.pad,"cuda")
                h,bias=prefix_fused(proc.model,x,mask,4,proc.cache)
                prefix+=4*volume(*x.shape,"prefix5")
                for j,row in enumerate(g):
                    state_rows[row]=h[j,:sizes[row]]
                for key,v in specs.items():
                    if v is None:
                        z=late_logits(proc.model,h,bias,mask,key).float().cpu().numpy()
                        raw[key][at+np.asarray(g)]=np.column_stack((z,tags[g]))
                        costs[key]+=volume(*x.shape,key)
                    elif "rule" in v:
                        pooled=pool(stack,h,mask)
                        early=proc.prefix_scores(pooled,proc.model.head(pooled),tags[g],sizes[g],v["rule"])
                        paths.setdefault(key,[]).append((g,early))
            for key,v in specs.items():
                if v is None:
                    continue
                early=np.empty((len(seq),3),dtype=np.float32)
                if key in paths:
                    for g,part in paths[key]:
                        early[g]=part.cpu().numpy()
                    decision=as_common_scale(early[:,:2],proc.thresholds)
                    selected=np.flatnonzero(early[:,2]>0)
                else:
                    decision=np.empty((len(seq),2),dtype=np.float32)
                    selected=range(len(seq))
                head=JointHead.from_dict(v["head"])
                for g in bucket(seq,selected):
                    h=pad_sequence([state_rows[i] for i in g],batch_first=True)
                    lengths=torch.as_tensor(sizes[g],device="cuda")
                    mask=(torch.arange(h.shape[1],device="cuda")[None,:]<lengths[:,None]).long()
                    bias=attention_bias(stack,mask,proc.cache)
                    z=late_logits(proc.model,h,bias,mask,v["depth"]).float().cpu().numpy()
                    decision[g]=as_common_scale(head(np.column_stack((z,tags[g]))),v["thresholds"])
                    costs[key]+=volume(*mask.shape,v["depth"])
                p[key][at:at+len(seq)]=decision
            forward+=time.perf_counter()-start
    return p,raw,widths,dict(wall_seconds=time.perf_counter()-total,forward_seconds=forward,
                              analytic={k:prefix+v for k,v in costs.items()})


def tune(output):
    output.mkdir(parents=True,exist_ok=True)
    proc=PackedPipeline()
    _,_,texts,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round9/frozen/holdout.npz") as saved:
        if ids!=saved["rows"].tolist():
            raise ValueError("Mismatched heldout rows")
        early=saved["early"]
        fold=saved["folds"]
    fit,cal,review=fold<=2,fold==3,fold==4
    _,raw,widths,_=score(proc,texts,{k:None for k in FIXED})
    out={}
    candidates=[]
    n=len(y)
    prefix=np.asarray([volume(1,int(w),"prefix5")*4 for w in widths])
    prefix_only=early>proc.thresholds
    for depth,z in raw.items():
        head=JointHead.fit(z[fit],y[fit],.6)
        p=head(z)
        cuts=calibrated(y[cal],p[cal])
        late=p>cuts
        record=dict(depth=depth,head=head.to_dict(),thresholds=cuts,
                    calibration=metric(y[cal],late[cal]),review=metric(y[review],late[review]))
        out[depth]=record
        surcharge=np.asarray([volume(1,int(w),depth) for w in widths])
        base=np.sum(prefix[cal]+surcharge[cal])
        for small in (.015,.025,.05,.075,.1,.15,.20):
            for large in (.72,.78,.84,.86,.9,.94):
                needs=(early[:,0]>small)&(early[:,0]<large)
                test=np.where(needs[:,None],late,prefix_only)
                f1=metric(y[cal],test[cal])["mean"]["f1"]
                ratio=(prefix[cal]+needs[cal]*surcharge[cal]).sum()/base
                # Fixed per-row cost estimates only select rules. Public timings are measured.
                objective=.6*f1-.2*485224/3e6*ratio
                candidates.append(dict(depth=depth,lower=small,upper=large,f1=f1,score=objective,
                                       review=metric(y[review],test[review]),late_rate=float(needs[cal].mean())))
    chosen=max(candidates,key=lambda k:k["score"])
    safe=max(candidates,key=lambda k:(k["f1"],k["score"]))
    targets={}
    for key,x in (("depth_score",chosen),("depth_quality",safe)):
        thresholds=dict(kind="asymmetric",lower=x["lower"],upper=x["upper"],late=2)
        targets[key]=dict(out[x["depth"]],rule=thresholds,selection=x)
    best=max(FIXED,key=lambda k:out[k]["calibration"]["mean"]["f1"])
    targets["fixed_depth"]=out[best]
    write_json(output/"policy.json",dict(selected=targets,heads=out,grid=candidates))
    print("depths",{k:v["review"]["mean"]["f1"] for k,v in out.items()},"selected",
          {k:(v["depth"],v.get("selection",{}).get("late_rate"),v.get("selection",v)["review"]["mean"]["f1"])
           for k,v in targets.items()},flush=True)


def public(output,repeat):
    config=json.loads((HERE/"depth_frozen/policy.json").read_text())["selected"]
    proc=PackedPipeline()
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    base=predict_bits(ROOT/"round9/combined/balanced_length/predictions.csv",ids,{"thresholds":(.5,.5)})
    order=tuple(config)
    times={k:[] for k in order}
    p={}
    for key in order:
        score(proc,texts[:2048],{key:config[key]})
    for attempt in range(repeat):
        for key in order if attempt%2==0 else order[::-1]:
            prob,_,_,timing=score(proc,texts,{key:config[key]})
            if key in p and not np.array_equal(p[key]>.5,prob[key]>.5):
                raise ValueError("Depth decisions changed on repeat")
            p[key]=prob[key]
            timing["analytic_gflops"]=timing.pop("analytic")[key]
            times[key].append(timing)
            print(key,attempt+1,timing,flush=True)
    outputs={}
    for key,value in p.items():
        timing={k:median(r[k] for r in times[key]) for k in times[key][0]}
        save(output/key,y,ids,cal,value,timing,(.5,.5))
        row=json.loads((output/key/"metrics.json").read_text())
        row["local_formula"]=competition_formula(row["scores"]["mean"]["f1"],timing["forward_seconds"],
                                                timing["analytic_gflops"])
        row["interval_95_vs_previous"]=bootstrap_delta(y,base,value>.5,groups)
        outputs[key]=row
    write_json(output/"analysis.json",dict(results=outputs,times=times))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","public"))
    ap.add_argument("--repeat",type=int,default=2)
    args=ap.parse_args()
    if args.stage=="tune":
        tune(HERE/"depth_frozen")
    else:
        public(HERE/"depth_public",args.repeat)
