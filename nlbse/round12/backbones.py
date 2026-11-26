"""Train matched project-disjoint compact and UniXcoder classifiers from upstream weights."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from project_split import HERE, ROOT, PLAN, checksum, read_partition
from normalize_code import parallel as canonicalize
sys.path[:0]=[str(ROOT/"round11"),str(ROOT/"round10")]
import small
import students
from pipeline import TRAIN, load, write_json

MODELS = HERE/"backbones"
WIDTH = 256
SEED = 42


def initialize(family,official=False):
    if family=="unix":
        model,tok=students.load_model("unixcoder9")
        source=ROOT/"models/unixcoder9/pytorch_model.bin"
        return model,tok,source
    tok=small.tokenizer()
    model=small.Encoder(len(tok),WIDTH).cuda()
    source=ROOT/"models/codet5_small/pytorch_model.bin"
    key="shared.weight"
    if official:
        previous=ROOT/"round10/runs/t5_small"
        config=json.loads((previous/"selected.json").read_text())
        source=previous/config["checkpoint"]
        if checksum(source)!=config["settings"]["sha256"]:
            raise ValueError("Official initialization has changed")
        key="encoder.shared.weight"
    raw=torch.load(source,map_location="cpu",weights_only=True)
    embeddings=raw[key][:len(tok)].cuda()
    with torch.no_grad():
        centered=embeddings-embeddings.mean(0)
        _,_,basis=torch.pca_lowrank(centered,q=WIDTH+16,niter=3,center=False)
        initial=centered@basis[:,:WIDTH]
        model.embed.weight.copy_(initial/initial.std(dim=0).clamp_min(.005))
    return model,tok,source


def train(family,epochs,official=False):
    torch.manual_seed(SEED)
    rng=random.Random(SEED)
    if official:
        if family!="canonical":
            raise ValueError("Official extension is registered for identifier-role augmentation")
        texts,y,official_targets=students.get_fit()
        fit,_=students.select_indices(pilot=False)
    else:
        split=read_partition()
        corpus,labels,_=load(TRAIN)
        fit=split["fit"]
        texts=[corpus[i] for i in fit]
        y=labels[fit]
        del corpus
    normalized=canonicalize(texts) if family=="canonical" else None
    model,tok,source=initialize(family,official)
    out=MODELS/("canonical_official" if official else family)
    out.mkdir(parents=True,exist_ok=True)
    if family=="unix":
        sequences=students.tokenize(texts,tok,"unixcoder9")
        size=64
        groups=[dict(params=model.encoder.parameters(),lr=3e-5),
                dict(params=list(model.head.parameters())+list(model.side.parameters()),lr=1.4e-4)]
        softened=None
    else:
        sequences=small.encode(texts,tok)
        size=128
        groups=[dict(params=model.embed.parameters(),lr=.0007),
                dict(params=[p for n,p in model.named_parameters() if not n.startswith("embed.")],lr=.0012)]
        if official:
            targets=official_targets
        else:
            with np.load(MODELS/"unix/teacher.npz") as archive:
                if not np.array_equal(archive["ids"],fit):
                    raise ValueError("Teacher inputs and fit order differ")
                targets=archive["probabilities"]
        teacher=torch.as_tensor(targets,device="cuda").clamp(1e-5,1-1e-5)
        softened=torch.sigmoid(torch.logit(teacher)/1.7)
    alternative=small.encode(normalized,tok) if normalized is not None else None
    del normalized
    optimizer=torch.optim.AdamW(groups,weight_decay=.01 if family=="unix" else .025,fused=True)
    order=list(range(len(y)))
    steps=epochs*len(list(students.batch_plan(order,sequences,size)))
    schedule=get_linear_schedule_with_warmup(optimizer,int(steps*.05),steps)
    side=torch.as_tensor(students.lexical_features(texts),device="cuda")
    gold=torch.as_tensor(y,device="cuda",dtype=torch.float32)
    pos=torch.as_tensor(np.clip((len(y)-y.sum(0))/y.sum(0),1,5),device="cuda",dtype=torch.float32)
    write_json(out/"config.json",dict(family=family,epochs=epochs,seed=SEED,batch=size,fit_rows=len(fit),
        input_checkpoint=checksum(source),model_dimensions=WIDTH if family!="unix" else 768,
        protocol="official" if official else "project_disjoint",
        partition_sha256=checksum(ROOT/"data/split_manifest.csv") if official else json.loads(PLAN.read_text())["partition_sha256"],
        teacher=(checksum(ROOT/"round10/teacher.npz") if official else
                 checksum(MODELS/"unix/teacher.npz") if softened is not None else None),
        augmented_probability=.45 if family=="canonical" else 0.))
    start=time.perf_counter()
    step=0
    for epoch in range(1,epochs+1):
        rng.shuffle(order)
        view=sequences if alternative is None else [a if rng.random()<.45 else b for a,b in zip(alternative,sequences,strict=True)]
        plan=list(students.batch_plan(order,view,size))
        rng.shuffle(plan)
        model.train()
        for rows in plan:
            x,mask=students.batch(rows,view,tok.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=model(x,mask,side[rows])
                loss=F.binary_cross_entropy_with_logits(output,gold[rows],pos_weight=pos)
                if softened is not None:
                    distill=F.binary_cross_entropy_with_logits(output/1.7,softened[rows])
                    loss=.7*loss+.3*1.7**2*distill
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            schedule.step()
            step+=1
            if step%500==0:
                print(family,step,steps,round(time.perf_counter()-start,1),float(loss.detach()),flush=True)
        pending=out/f"epoch{epoch}.part"
        torch.save(model.state_dict(),pending)
        pending.replace(pending.with_suffix(".pt"))
        print(family,"epoch",epoch,flush=True)
    checkpoint=out/f"epoch{epochs}.pt"
    write_json(out/"selected.json",dict(checkpoint=checkpoint.name,sha256=checksum(checkpoint),
               selection="final predeclared epoch",train_seconds=time.perf_counter()-start))


def frozen(family):
    out=MODELS/family
    selected=json.loads((out/"selected.json").read_text())
    checkpoint=out/selected["checkpoint"]
    if checksum(checkpoint)!=selected["sha256"]:
        raise ValueError("Project-disjoint checkpoint modified")
    if family=="unix":
        net,tok=students.load_model("unixcoder9",pretrained=False)
    else:
        tok=small.tokenizer()
        net=small.Encoder(len(tok),WIDTH).cuda()
        for layer in net.local:
            layer.use_shifted=True
    net.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
    return net,tok


def score(family,net,tok,functions):
    if family=="unix":
        return students.scores(net,tok,functions,"unixcoder9")
    p,_,cost=small.scores(net,tok,functions)
    return p,cost


def export_scores(family,parts):
    split=read_partition()
    texts,y,_=load(TRAIN)
    model,tok=frozen(family)
    out=MODELS/family
    for part in parts:
        ids=split[part]
        p,cost=score(family,model,tok,[texts[i] for i in ids])
        if p.shape!=y[ids].shape or not np.isfinite(p).all():
            raise ValueError("Unexpected prediction shape or values")
        target=out/("teacher.npz" if part=="fit" else f"{part}.npz")
        temp=target.with_suffix(".part.npz")
        np.savez_compressed(temp,ids=ids,probabilities=p,y=y[ids])
        temp.replace(target)
        write_json(target.with_suffix(".json"),dict(role=part,cost=cost,sha256=checksum(target)))
        print(family,part,len(ids),cost,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("train","score"))
    ap.add_argument("family",choices=("unix","compact","canonical"))
    ap.add_argument("--epochs",type=int)
    ap.add_argument("--official",action="store_true")
    ap.add_argument("--parts",nargs="+",choices=("fit","probe","dev","test"),default=["probe","dev","test"])
    args=ap.parse_args()
    if args.stage=="train":
        train(args.family,args.epochs or (2 if args.family=="unix" else 5),args.official)
    else:
        export_scores(args.family,args.parts)
