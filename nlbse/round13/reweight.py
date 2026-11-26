"""Matched continuation with extra supervised weight for short teacher misses."""
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from rotate import HERE, PLAN, checksum, load_backbones, read_partition
from evaluate import OUT as STUDY, counts, exact_budget, inputs, raw, summarize_row
from pipeline import write_json

OUT=HERE/"reweight"


def fit(mode):
    back=load_backbones()
    torch.manual_seed(42)
    rng=random.Random(42)
    texts,y,_,_,parts=inputs()
    ids=parts["fit"]
    train=[texts[i] for i in ids]
    label=y[ids]
    source=HERE/"backbones/unix/teacher.npz"
    with np.load(source) as stored:
        if not np.array_equal(stored["ids"],ids) or not np.array_equal(stored["y"],label):
            raise ValueError("The teacher no longer matches the project rotation")
        target=stored["probabilities"]
    model,tok=back.frozen("compact")
    for layer in model.local:
        layer.use_shifted=False
    model.train()
    sequence=back.small.encode(train,tok)
    features=torch.as_tensor(back.students.lexical_features(train),device="cuda")
    truth=torch.as_tensor(label,device="cuda",dtype=torch.float32)
    t=torch.as_tensor(target,device="cuda").clamp(1e-5,1-1e-5)
    t=(t.logit()/1.7).sigmoid()
    short=(np.asarray([len(s) for s in train])<=json.loads(PLAN.read_text())["short_chars_max"]) & (label[:,0]==1)
    missed=short&(target[:,0]<.05)
    short_tensor=torch.as_tensor(short,device="cuda")
    missed_tensor=torch.as_tensor(missed,device="cuda")
    pos=torch.as_tensor(np.clip((len(label)-label.sum(0))/label.sum(0),1,5),device="cuda",dtype=torch.float32)
    rates=[{"params":model.embed.parameters(),"lr":.00016},
           {"params":[p for n,p in model.named_parameters() if not n.startswith("embed.")],"lr":.00035}]
    opt=torch.optim.AdamW(rates,weight_decay=.025,fused=True)
    order=list(range(len(ids)))
    rng.shuffle(order)
    stages=list(back.students.batch_plan(order,sequence,128))
    rng.shuffle(stages)
    schedule=get_linear_schedule_with_warmup(opt,int(.04*len(stages)),len(stages))
    start=time.perf_counter()
    for step,rows in enumerate(stages,1):
        x,mask=back.students.batch(rows,sequence,tok.pad_token_id)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=model(x,mask,features[rows])
            gold=F.binary_cross_entropy_with_logits(p,truth[rows],pos_weight=pos,reduction="none")
            soft=F.binary_cross_entropy_with_logits(p/1.7,t[rows],reduction="none")
            loss=.7*gold+.3*1.7**2*soft
            if mode=="focused":
                weighted=torch.where(short_tensor[rows],loss[:,0]*1.6,loss[:,0])
                weighted=torch.where(missed_tensor[rows],gold[:,0]*2.3,weighted)
                loss=.5*(weighted.mean()+loss[:,1].mean())
            else:
                loss=loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        schedule.step()
        if step%400==0:
            print(mode,step,len(stages),round(time.perf_counter()-start,2),flush=True)
    path=OUT/mode
    path.mkdir(parents=True,exist_ok=True)
    f=path/"checkpoint.pt"
    tmp=f.with_suffix(".part")
    torch.save(model.state_dict(),tmp)
    tmp.replace(f)
    write_json(path/"config.json",dict(mode=mode,rows=len(label),steps=len(stages),seed=42,epochs=1,
             short_positive=int(short.sum()),teacher_confident_short_miss=int(missed.sum()),teacher_sha256=checksum(source),
             model_sha256=checksum(f),seconds=time.perf_counter()-start))
    print(mode,"retained",int(short.sum()),int(missed.sum()),flush=True)


def score(mode,part):
    back=load_backbones()
    rows=read_partition()[part]
    texts,labels,_,_,_=inputs()
    out=OUT/mode
    path=out/"checkpoint.pt"
    if checksum(path)!=json.loads((out/"config.json").read_text())["model_sha256"]:
        raise ValueError("Continuation checkpoint changed")
    model,tok=back.frozen("compact")
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    p,c=back.score("compact",model,tok,[texts[i] for i in rows])
    np.savez_compressed(out/f"{part}.npz",ids=rows,y=labels[rows],p=p)
    print(mode,part,c,flush=True)


def compare():
    texts,y,projects,_,parts=inputs()
    threshold=json.loads((STUDY/"policy.json").read_text())["compact_threshold"]
    short_cap=json.loads(PLAN.read_text())["short_chars_max"]
    recipe={}
    reference={k:raw(k) for k in ("dev","test")}
    for name in ("matched","focused"):
        t=parts["dev"]
        with np.load(OUT/name/"dev.npz") as f:
            if not np.array_equal(f["ids"],t):
                raise ValueError("Development predictions out of order")
            probability=f["p"][:,0]
        b=counts(y[t,0],reference["dev"]["compact"]>threshold)["fp"]
        _,cut=exact_budget(probability,y[t,0],b)
        recipe[name]=cut
    write_json(OUT/"policy.json",dict(dev_thresholds=recipe,original_threshold=threshold))
    t=parts["test"]
    gold=y[t,0]
    original=reference["test"]["compact"]
    locked=original>threshold
    fp=counts(gold,locked)["fp"]
    original_equal,_=exact_budget(original,gold,fp)
    short=np.asarray([len(texts[i])<=short_cap for i in t])
    short_fp=counts(gold[short],locked[short])["fp"]
    short_original,_=exact_budget(original[short],gold[short],short_fp)
    conf=short&(original<.05)&(reference["test"]["unix"]<.05)
    output={}
    for name in recipe:
        with np.load(OUT/name/"test.npz") as f:
            if not np.array_equal(f["ids"],t):
                raise ValueError("Test predictions out of order")
            p=f["p"][:,0]
        equal,_=exact_budget(p,gold,fp)
        s,_=exact_budget(p[short],gold[short],short_fp)
        fixed=p>recipe[name]
        output[name]=dict(equal_fp=summarize_row(gold,original_equal,equal,conf),
            short_equal_fp=summarize_row(gold[short],short_original,s,conf[short]),
            frozen=summarize_row(gold,locked,fixed,conf))
        with (OUT/name/"predictions.csv").open("w",newline="") as out:
            w=csv.writer(out)
            w.writerow(("row","project","y","short","joint_low","base_score","candidate_score","base_pred","base_exact_fp","pred_dev_cut","pred_exact_fp"))
            w.writerows(zip(t,projects[t],gold,short.astype(int),conf.astype(int),original,p,locked.astype(int),
                            original_equal.astype(int),fixed.astype(int),equal.astype(int),strict=True))
    write_json(OUT/"analysis.json",output)
    print(output,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("fit","score","compare"))
    ap.add_argument("--mode",choices=("matched","focused"))
    ap.add_argument("--part",choices=("dev","test"))
    args=ap.parse_args()
    if args.stage=="fit":
        fit(args.mode)
    elif args.stage=="score":
        score(args.mode,args.part)
    else:
        compare()
