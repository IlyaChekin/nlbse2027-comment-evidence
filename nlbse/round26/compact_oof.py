"""Build compact conditional scores without training on the scored files."""
import argparse
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"round25"))
from length_eval import R24,PARTS,f,checksum,json_at,write_json
from train import cohorts
from student import rules
from rule_model import RuleEncoder

OUT=HERE/"compact"
WIDTH=256
SEED=42


def signed_source(rotation):
    folder=R24/rotation
    file=folder/"token_oof.npz"
    if checksum(file)!=json_at(folder/"fit.json")["models"]["token"]["predictions_sha256"]:
        raise ValueError("Frozen sparse split and predictions changed")
    with np.load(file) as z:
        return dict(z),checksum(file)


def collect(rotation,epochs):
    split,split_hash=signed_source(rotation)
    x,labels,_,_,_=f.inputs()
    weak,y,cohort_hash=cohorts()
    active=split["ids"]
    if not np.array_equal(y[active],split["y"]) or not np.array_equal(weak[active],split["weak"]):
        raise ValueError("The independent score sources disagree on row order")
    tok=f.small.tokenizer()
    # Stock language-model embeddings contain no labels from any split of this experiment.
    source=ROOT/"models/codet5_small/pytorch_model.bin"
    weights=torch.load(source,map_location="cpu",weights_only=True)["shared.weight"][:len(tok)].cuda()
    torch.manual_seed(SEED)
    weights=weights-weights.mean(0)
    _,_,v=torch.pca_lowrank(weights,q=WIDTH+16,niter=3,center=False)
    init=weights@v[:,:WIDTH]
    init/=init.std(dim=0).clamp_min(.005)
    plain=[x[i] for i in active]
    sequence=f.tokenize(plain,tok)
    side=torch.as_tensor(f.base.lexical_features(plain),device="cuda")
    truth=torch.as_tensor(labels[active],device="cuda",dtype=torch.float32)
    outcome=torch.as_tensor(weak[active],device="cuda")
    events,seen,target_hash=rules()
    known=torch.as_tensor(seen[active],device="cuda")
    rule=torch.as_tensor(events[active],device="cuda",dtype=torch.float32)
    folder=OUT/rotation
    folder.mkdir(parents=True,exist_ok=True)
    records={}
    scores=np.empty((len(active),4),dtype=np.float32)
    for k in sorted(set(split["fold"])):
        held=split["fold"]==k
        allowed=(split["dup"]&(1<<int(k)))==0
        if np.any(held&allowed):
            raise ValueError("Out-of-file compact split overlaps")
        torch.manual_seed(SEED)
        rng=random.Random(SEED)
        model=RuleEncoder(f.small.Encoder(len(tok),WIDTH),conditional=True).cuda()
        with torch.no_grad():
            model.encoder.embed.weight.copy_(init)
        opt=torch.optim.AdamW([dict(params=model.encoder.embed.parameters(),lr=.00065),
                              dict(params=[p for n,p in model.named_parameters()
                                           if n!="encoder.embed.weight"],lr=.0011)],weight_decay=.025,fused=True)
        indices=np.flatnonzero(allowed).tolist()
        size=len(indices)
        target=truth[indices]
        weights_class=((size-target.sum(0))/target.sum(0).clamp_min(1)).clamp(1,5)
        rate=(rule[indices].sum(0)).clamp_min(1)
        weights_aux=((size-rate)/rate).clamp(1,25).sqrt()
        enable=known & (rate>20)
        total_steps=epochs*len(list(f.base.batch_plan(indices,sequence,f.small.BATCH)))
        schedule=get_linear_schedule_with_warmup(opt,int(.04*total_steps),total_steps)
        start=time.perf_counter()
        for e in range(epochs):
            rng.shuffle(indices)
            batches=list(f.base.batch_plan(indices,sequence,f.small.BATCH))
            rng.shuffle(batches)
            model.train()
            for rows in batches:
                tokens,mask=f.base.batch(rows,sequence,tok.pad_token_id)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    q=model(tokens,mask,side[rows])
                    body=truth[rows]
                    main=F.binary_cross_entropy_with_logits(q[:,:2],body,pos_weight=weights_class)
                    aux=F.binary_cross_entropy_with_logits(q[:,2:22],rule[rows],pos_weight=weights_aux,reduction="none")
                    aux=(aux*enable[rows]).sum()/enable[rows].sum().clamp_min(1)
                    flag=outcome[rows]
                    local=q[:,22:].gather(1,flag.clamp_min(0).long()[:,None])[:,0]
                    supervised=F.binary_cross_entropy_with_logits(local,body[:,0],reduction="none")
                    conditional=(supervised*(flag>=0)).sum()/(flag>=0).sum().clamp_min(1)
                    loss=.7*main+.25*aux+.50*conditional
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                schedule.step()
        model.eval()
        for unit in model.encoder.local:
            unit.use_shifted=True
        test=[value for value,choose in zip(plain,held,strict=True) if choose]
        pipe=SimpleNamespace(encoder=model,tok=tok,mode="control")
        p,aux,_=f.infer_student(pipe,test,f.pinned()[0],columns=22,return_all=True)
        scores[held]=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
        file=folder/f"fold_{k}.pt"
        torch.save(model.state_dict(),file)
        records[int(k)]=dict(fit=size,held=len(test),seconds=time.perf_counter()-start,sha256=checksum(file))
        print(rotation,int(k),records[int(k)],flush=True)
    dst=folder/"scores.npz"
    np.savez_compressed(dst,ids=active,y=y[active],fold=split["fold"],values=scores)
    write_json(folder/"manifest.json",dict(rows=len(active),folds=records,sparse_oof_sha256=split_hash,
                    rule_target_sha256=target_hash,weak_cohorts_sha256=cohort_hash,embedding_source_sha256=checksum(source),
                    epochs=epochs,seed=SEED,width=WIDTH,sha256=checksum(dst),code_sha256=checksum(Path(__file__))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    p.add_argument("--epochs",type=int,default=2)
    a=p.parse_args()
    collect(a.rotation,a.epochs)
