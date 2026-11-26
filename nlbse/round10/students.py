"""Matched supervised and distilled comparison of two code encoders."""

import argparse
import csv
import json
from pathlib import Path
import random
from statistics import median
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, RobertaModel, T5EncoderModel, get_linear_schedule_with_warmup

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round9"))
from cache_hybrid import ACTIONS
from frontier import calibrated, lexical_features, ordered_batches
from common import dataset, sha256
from pipeline import PUBLIC, TRAIN, load, metric, select_indices, write_json
from neural import batch_gflops, batch_plan, batch
from protocol import save
from score_runtime import competition_formula
from analyze import group_codes,predict_bits
from round2_report import bootstrap_delta

ROOT=HERE.parent
FAMILIES={"t5_small":"codet5_small","unixcoder9":"unixcoder9"}
BATCH=64
CAP=512


class Model(nn.Module):
    def __init__(self,family,pretrained=True):
        super().__init__()
        source=ROOT/"models"/FAMILIES[family]
        if family=="t5_small":
            cfg=AutoConfig.from_pretrained(source,local_files_only=True)
            encoder=(T5EncoderModel.from_pretrained(source,config=cfg,local_files_only=True)
                     if pretrained else T5EncoderModel(cfg))
            self.layers=list(range(len(encoder.encoder.block)))
            width=encoder.config.d_model
        else:
            cfg=AutoConfig.from_pretrained(source,local_files_only=True)
            cfg._attn_implementation="sdpa"
            encoder=(RobertaModel.from_pretrained(source,config=cfg,local_files_only=True,
                                                  add_pooling_layer=False)
                     if pretrained else RobertaModel(cfg,add_pooling_layer=False))
            self.layers=[0,2,4,6,8,10]
            encoder.encoder.layer=nn.ModuleList(encoder.encoder.layer[i] for i in self.layers)
            encoder.config.num_hidden_layers=len(self.layers)
            width=encoder.config.hidden_size
        self.config=encoder.config
        self.encoder=encoder
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Dropout(.12),nn.Linear(width,2))
        self.side=nn.Linear(5,2,bias=False)
        self.family=family

    def forward(self,x,mask,side):
        hidden=self.encoder(input_ids=x,attention_mask=mask).last_hidden_state
        pooled=(hidden*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1,keepdim=True)
        return self.head(pooled)+self.side(side)


def tokenize(texts,tokenizer,family):
    seq=[]
    for start in range(0,len(texts),1024):
        part=texts[start:start+1024]
        if family=="unixcoder9":
            role=tokenizer.convert_tokens_to_ids("<encoder-only>")
            if role==tokenizer.unk_token_id:
                raise ValueError("Official UniXcoder encoder-only token is missing")
            raw=tokenizer(part,truncation=True,max_length=CAP-4,add_special_tokens=False)["input_ids"]
            prefix=[tokenizer.bos_token_id,role,tokenizer.eos_token_id]
            seq.extend([*prefix,*x,tokenizer.eos_token_id] for x in raw)
        else:
            seq.extend(tokenizer(part,truncation=True,max_length=CAP)["input_ids"])
    return seq


def load_model(family,pretrained=True):
    model=Model(family,pretrained).cuda()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/"models"/FAMILIES[family],local_files_only=True)
    return model,tokenizer


def get_fit():
    official,y,_=load(TRAIN)
    rows,_=select_indices(pilot=False)
    subset=HERE/"teacher.npz"
    if subset.is_file():
        expected=json.loads((HERE/"teacher.json").read_text())["sha256"]
        if sha256(subset)!=expected:
            raise ValueError("The exported teacher changed")
    else:
        subset=ROOT/"round9/cache/fit.npz"
    with np.load(subset) as cached:
        if rows!=cached["ids"].tolist() or not np.array_equal(y[rows],cached["y"]):
            raise ValueError("Teacher examples do not align")
        teacher=cached[ACTIONS[1]]
    return [official[i] for i in rows],y[rows],teacher


def fit(family,epochs):
    torch.manual_seed(42)
    rng=random.Random(42)
    texts,y,teacher=get_fit()
    model,tokenizer=load_model(family)
    out=HERE/"runs"/family
    out.mkdir(parents=True,exist_ok=True)
    config=dict(family=family,cap=CAP,batch=BATCH,seed=42,epochs=epochs,rows=len(y),
                source_layers=model.layers,teacher_weight=.26,temperature=1.7,
                upstream_sha256=sha256(ROOT/"models"/FAMILIES[family]/"pytorch_model.bin"))
    write_json(out/"config.json",config)
    seq=tokenize(texts,tokenizer,family)
    side=torch.as_tensor(lexical_features(texts),device="cuda")
    labels=torch.as_tensor(y,dtype=torch.float32,device="cuda")
    with torch.no_grad():
        p=torch.as_tensor(teacher,device="cuda").clamp(1e-5,1-1e-5)
        targets=torch.sigmoid(torch.logit(p)/config["temperature"])
    weights=torch.as_tensor(np.maximum(1,np.minimum(5,(len(y)-y.sum(0))/y.sum(0))),
                            dtype=torch.float32,device="cuda")
    opt=torch.optim.AdamW([
        {"params":model.encoder.parameters(),"lr":3e-5},
        {"params":list(model.head.parameters())+list(model.side.parameters()),"lr":1.4e-4},
    ],weight_decay=.01,fused=True)
    order=list(range(len(y)))
    steps=epochs*len(list(batch_plan(order,seq,BATCH)))
    schedule=get_linear_schedule_with_warmup(opt,int(steps*.05),steps)
    start=time.perf_counter()
    step=0
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        runs=list(batch_plan(order,seq,BATCH))
        rng.shuffle(runs)
        total=torch.zeros((),device="cuda")
        for ids in runs:
            x,mask=batch(ids,seq,tokenizer.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=model(x,mask,side[ids])
                supervised=F.binary_cross_entropy_with_logits(p,labels[ids],pos_weight=weights)
                soft=F.binary_cross_entropy_with_logits(p/config["temperature"],targets[ids])
                loss=(1-config["teacher_weight"])*supervised+config["teacher_weight"]*config["temperature"]**2*soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            schedule.step()
            total+=loss.detach()
            step+=1
            if step%250==0:
                print(family,step,steps,round(time.perf_counter()-start,2),flush=True)
        filename=out/f"epoch{epoch+1}.pt"
        temp=filename.with_suffix(".part")
        torch.save(model.state_dict(),temp)
        temp.replace(filename)
        print(family,"epoch",epoch+1,round(total.item()/len(runs),5),flush=True)
    write_json(out/"training.json",dict(seconds=time.perf_counter()-start,steps=step,epochs=epochs))


@torch.inference_mode()
def scores(model,tokenizer,texts,family, *, prepare=None, readout=None, fused=False):
    model.eval()
    p=np.empty((len(texts),2),dtype=np.float32)
    volume,forward=0.,0.
    if fused:
        if family != "t5_small":
            raise ValueError("The fused implementation requires a T5 encoder")
        from fused_attention import pool, prefix_fused
        bias_cache={}
    start=time.perf_counter()
    def prepare_at(offset):
        local=texts[offset:offset+2048]
        if prepare is not None:
            return prepare(local,tokenizer)
        return tokenize(local,tokenizer,family),lexical_features(local),None
    with torch.autocast("cuda",dtype=torch.float16):
        for offset,(seq,lex,extra) in ordered_batches(range(0,len(texts),2048),prepare_at,True):
            side=torch.as_tensor(lex,device="cuda")
            for indices in batch_plan(list(range(len(seq))),seq,BATCH):
                x,mask=batch(indices,seq,tokenizer.pad_token_id)
                tick=time.perf_counter()
                if fused:
                    hidden,_=prefix_fused(model,x,mask,len(model.layers),bias_cache)
                    logits=model.head(pool(model.encoder.encoder,hidden,mask))+model.side(side[indices])
                else:
                    logits=model(x,mask,side[indices])
                p[offset+np.asarray(indices)] = logits.float().sigmoid().cpu().numpy()
                forward+=time.perf_counter()-tick
                volume+=batch_gflops("codet5p" if family=="t5_small" else "codebert",model,*x.shape)
            if readout is not None:
                readout(p[offset:offset+len(seq)],extra)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite probabilities in FP16 deployment")
    return p,dict(wall_seconds=time.perf_counter()-start,forward_seconds=forward,analytic_gflops=volume)


def tune(family):
    out=HERE/"runs"/family
    model,tokenizer=load_model(family,pretrained=False)
    _,_,texts,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as grouped:
        if ids!=grouped["rows"].tolist():
            raise ValueError("Evaluation rows changed")
        fold=grouped["folds"]
    cal,review=fold==3,fold==4
    records={}
    for path in sorted(out.glob("epoch*.pt")):
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        p,timing=scores(model,tokenizer,texts,family)
        t=calibrated(y[cal],p[cal])
        records[path.name]=dict(thresholds=t,calibration=metric(y[cal],p[cal]>t),
                                 review=metric(y[review],p[review]>t),timing=timing,
                                 sha256=sha256(path))
        np.savez_compressed(out/(path.stem+"_holdout.npz"),p=p,ids=ids,fold=fold,y=y)
        print(family,path.name,records[path.name]["review"]["mean"],flush=True)
    chosen=max(records,key=lambda k:records[k]["calibration"]["mean"]["f1"])
    write_json(out/"selected.json",dict(checkpoint=chosen,settings=records[chosen],candidates=records))


def load_selected(family):
    out=HERE/"runs"/family
    config=json.loads((out/"selected.json").read_text())
    path=out/config["checkpoint"]
    if sha256(path)!=config["settings"]["sha256"]:
        raise ValueError("The checkpoint changed after calibration")
    model,tokenizer=load_model(family,pretrained=False)
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return model,tokenizer,config["settings"]["thresholds"]


def public(family,repeat):
    out=HERE/"runs"/family
    model,tokenizer,t=load_selected(family)
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    profiles=[]
    scores(model,tokenizer,texts[:2048],family)
    reference=None
    for _ in range(repeat):
        p,cost=scores(model,tokenizer,texts,family)
        if reference is not None and not np.array_equal(p>t,reference):
            raise ValueError("Model decisions changed across runs")
        reference=p>t
        profiles.append(cost)
        print(family,"public",cost,flush=True)
    totals={k:median(v[k] for v in profiles) for k in profiles[0]}
    save(out/"public",y,ids,cal,p,totals,t)
    metrics=json.loads((out/"public/metrics.json").read_text())
    metrics["local_formula"]=competition_formula(metrics["scores"]["mean"]["f1"],
                                                 totals["forward_seconds"],totals["analytic_gflops"])
    old=ROOT/"round9/combined/balanced_length/predictions.csv"
    anchor=predict_bits(old,ids,{"thresholds":(.5,.5)})
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    metrics["interval_vs_balanced"]=bootstrap_delta(y,anchor,reference,groups)
    metrics["repeats"]=profiles
    write_json(out/"public/analysis.json",metrics)


def predict(family,source,target, *, fused=False):
    if source is None or target is None:
        raise ValueError("Supply the input and output CSV paths")
    with source.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV requires a Function column")
        texts=[row["Function"] for row in reader]
    if not texts or any(value is None for value in texts):
        raise ValueError("Every row must supply a function")
    model,tokenizer,t=load_selected(family)
    p,_=scores(model,tokenizer,texts,family,fused=fused)
    with target.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>t).astype(int).tolist())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","tune","public","predict"))
    parser.add_argument("family",choices=FAMILIES)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.stage=="fit":
        fit(args.family,args.epochs)
    elif args.stage=="tune":
        tune(args.family)
    elif args.stage=="public":
        public(args.family,args.repeat)
    else:
        predict(args.family,args.input,args.output)
