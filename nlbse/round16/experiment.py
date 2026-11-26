"""Train matched local-token readouts and compare on project-disjoint data."""

import argparse
import csv
import json
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
sys.path[:0]=[str(ROOT/"round13"),str(ROOT/"round12"),str(ROOT/"round11"),str(ROOT/"round10"),str(ROOT)]
from rotate import checksum
from evaluate import counts, exact_budget, floor_f1, grouped_delta, inputs, raw
from students import batch, batch_plan, group_codes, lexical_features, ordered_batches
from pipeline import write_json
from heads import MODES, LocalEncoder, operations
from margin_rank import RANK_MODES,pair_loss,training_weights
from memory import MEMORY_MODES, MemoryEncoder
import small


PROJECT=HERE/"projects"
OFFICIAL=HERE/"official"
MEMORY=HERE/"memory_projects"
RANKING=HERE/"ranking_projects"


def json_at(path):
    return json.loads(path.read_text())


def checkpoint(scope):
    if scope=="projects":
        folder=ROOT/"round13/backbones/compact"
        row=json_at(folder/"selected.json")
        return folder/row["checkpoint"],row["sha256"]
    folder=ROOT/"round15/official_distill"
    row=json_at(folder/"fit.json")["variants"]["original_teacher"]
    return folder/row["checkpoint"],row["sha256"]


def new_model(tok,mode):
    # Stable width and vocabulary across all arms.
    if mode in RANK_MODES:
        return small.Encoder(len(tok),256).cuda()
    if mode in MEMORY_MODES:
        return MemoryEncoder(len(tok),256,mode).cuda()
    return LocalEncoder(len(tok),256,mode).cuda()


def load_original(tok,mode,source):
    path,digest=source
    if checksum(path)!=digest:
        raise ValueError("Initialization checkpoint changed")
    model=new_model(tok,mode)
    status=model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=mode in RANK_MODES)
    if status.unexpected_keys or any(not k.startswith("evidence.") for k in status.missing_keys):
        raise ValueError("Unexpected differences from the original encoder")
    return model


def fit_data(scope):
    if scope=="official":
        from students import get_fit
        a,b,c=get_fit()
        return a,b,c,checksum(ROOT/"round10/teacher.npz")
    texts,labels,_,_,parts=inputs()
    ids=parts["fit"]
    path=ROOT/"round13/backbones/unix/teacher.npz"
    digest=checksum(path)
    if digest!=json_at(path.with_suffix(".json"))["sha256"]:
        raise ValueError("Project teacher predictions changed")
    with np.load(path) as f:
        if not np.array_equal(f["ids"],ids) or not np.array_equal(f["y"],labels[ids]):
            raise ValueError("Project teacher rows differ")
        return [texts[i] for i in ids],labels[ids],f["probabilities"],digest


def fit(scope,root,modes,seed):
    root.mkdir(parents=True,exist_ok=True)
    texts,labels,teacher,teacher_sha=fit_data(scope)
    source=checkpoint(scope)
    tok=small.tokenizer()
    sequence=small.encode(texts,tok)
    alternate=None
    if scope=="official":
        from normalize_code import parallel
        alternate=small.encode(parallel(texts),tok)
    lex=torch.as_tensor(lexical_features(texts),device="cuda")
    truth=torch.as_tensor(labels,device="cuda",dtype=torch.float32)
    softened=(torch.as_tensor(teacher,device="cuda").clamp(1e-5,1-1e-5).logit()/1.7).sigmoid()
    balance=torch.as_tensor(np.clip((len(labels)-labels.sum(0))/labels.sum(0),1,5),device="cuda",dtype=torch.float32)
    ranking=any(k in RANK_MODES[1:] for k in modes)
    if ranking:
        if scope!="projects":
            raise ValueError("The ranking cache currently covers the project-disjoint training split")
        rank_state,edge,rank_sha=training_weights(ROOT,inputs()[4]["fit"],labels,checksum)
    report={}
    for mode in modes:
        torch.manual_seed(seed)
        rng=random.Random(seed)
        model=load_original(tok,mode,source)
        model.train()
        # Align dropout, the view sequence and minibatch order with the control.
        torch.manual_seed(seed)
        seq=sequence if alternate is None else [b if rng.random()<.45 else a for a,b in zip(sequence,alternate,strict=True)]
        order=list(range(len(seq)))
        rng.shuffle(order)
        batches=list(batch_plan(order,seq,small.BATCH))
        rng.shuffle(batches)
        params=[dict(params=model.embed.parameters(),lr=.00014),
                dict(params=[p for k,p in model.named_parameters() if not k.startswith(("embed.","evidence."))],lr=.00029)]
        if mode not in RANK_MODES:
            params.append(dict(params=model.evidence.parameters(),lr=.0008))
        opt=torch.optim.AdamW(params,weight_decay=.025,fused=True)
        schedule=get_linear_schedule_with_warmup(opt,int(.04*len(batches)),len(batches))
        at=time.perf_counter()
        for step,rows in enumerate(batches,1):
            x,mask=batch(rows,seq,tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=model(x,mask,lex[rows])
                supervised=F.binary_cross_entropy_with_logits(p,truth[rows],pos_weight=balance)
                distill=F.binary_cross_entropy_with_logits(p/1.7,softened[rows])
                loss=.7*supervised+.3*1.7**2*distill
                if mode in RANK_MODES[1:]:
                    loss=loss+.15*pair_loss(p[:,0],truth[rows,0],rank_state[rows],edge,mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            schedule.step()
            if step%400==0:
                print(scope,seed,mode,step,len(batches),round(time.perf_counter()-at,1),flush=True)
        path=root/f"{mode}.pt"
        pending=path.with_suffix(".part")
        torch.save(model.state_dict(),pending)
        pending.replace(path)
        report[mode]=dict(checkpoint=path.name,sha256=checksum(path),steps=len(batches),training_seconds=time.perf_counter()-at)
        write_json(root/"fit.json",dict(scope=scope,seed=seed,epochs=1,fit_rows=len(texts),teacher_sha256=teacher_sha,
                   base_sha256=source[1],training=report,out_of_fold_sha256=rank_sha if ranking else None))
        del model,opt,schedule


def open_model(scope,root,mode,tok):
    if mode=="reference":
        model=load_original(tok,"control",checkpoint(scope))
    else:
        spec=json_at(root/"fit.json")["training"][mode]
        path=root/spec["checkpoint"]
        if checksum(path)!=spec["sha256"]:
            raise ValueError("Trained local readout changed")
        model=new_model(tok,mode)
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    for block in model.local:
        block.use_shifted=True
    model.eval()
    return SimpleNamespace(encoder=model,tok=tok,mode="control" if mode=="reference" or mode in RANK_MODES else mode)


@torch.inference_mode()
def infer(pipe,texts,sequences=None,encoding=None,aux_out=None):
    model=pipe.encoder
    tok=pipe.tok
    result=np.empty((len(texts),2),dtype=np.float32)
    start=time.perf_counter()
    fwd,gflops=0.,0.

    def prepare(first):
        examples=texts[first:first+2048]
        if sequences is None:
            batch_ids=(encoding or small.encode)(examples,tok)
        else:
            batch_ids=sequences[first:first+2048]
        return batch_ids,lexical_features(examples)

    with torch.autocast("cuda",dtype=torch.float16):
        for first,(seq,lex) in ordered_batches(range(0,len(texts),2048),prepare,True):
            result[first:first+len(seq),1]=lex[:,:4].any(axis=1)
            side=torch.as_tensor(lex,device="cuda")
            for rows in batch_plan(list(range(len(seq))),seq,small.BATCH):
                x,mask=batch(rows,seq,tok.pad_token_id)
                at=time.perf_counter()
                logits=model(x,mask,side[rows])
                indices=first+np.asarray(rows)
                if aux_out is None:
                    result[indices,0]=logits[:,0].float().sigmoid().cpu().numpy()
                else:
                    p=logits.float().sigmoid().cpu().numpy()
                    result[indices,0]=p[:,0]
                    aux_out[indices]=p[:,2:]
                    gflops+=len(rows)*(4*model.width*aux_out.shape[1]+6*model.width)/1e9
                fwd+=time.perf_counter()-at
                gflops+=operations(len(rows),x.shape[1],model.width,pipe.mode)
    if not np.isfinite(result).all():
        raise ValueError("Local pooling produced nonfinite predictions")
    return result,dict(wall_seconds=time.perf_counter()-start,forward_seconds=fwd,analytic_gflops=gflops)


def score_project(root,part,modes):
    texts,labels,_,_,parts=inputs()
    ids=parts[part]
    local=[texts[i] for i in ids]
    tok=small.tokenizer()
    for mode in modes:
        pipe=open_model("projects",root,mode,tok)
        p,cost=infer(pipe,local)
        out=root/f"{part}_{mode}.npz"
        np.savez_compressed(out,ids=ids,y=labels[ids],p=p)
        write_json(out.with_suffix(".json"),dict(sha256=checksum(out),cost=cost))
        print(part,mode,cost,flush=True)


def loaded(root,part,mode,ids,y):
    path=root/f"{part}_{mode}.npz"
    if checksum(path)!=json_at(path.with_suffix(".json"))["sha256"]:
        raise ValueError("Local readout scores changed")
    with np.load(path) as z:
        if not np.array_equal(z["ids"],ids) or not np.array_equal(z["y"],y):
            raise ValueError("Local scores disagree with the holdout")
        return z["p"]


def strict_budget(scores,y,budget):
    # Freeze a threshold that preserves tied scores and does not exceed the FP budget.
    negatives=np.sort(scores[y==0])[::-1]
    cut=float(negatives[budget]) if budget<len(negatives) else -1.
    return scores>cut,cut


def comparison(y,old,new):
    a,b=counts(y,old),counts(y,new)
    return dict(before=a,after=b,delta_tp=b["tp"]-a["tp"],delta_fp=b["fp"]-a["fp"],
                f1_before=floor_f1(y,old),f1_after=floor_f1(y,new))


def project_eval(root,modes):
    _,labels,projects,paths,parts=inputs()
    i=parts["dev"]
    initial=raw("dev")["compact"]
    threshold=json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    budget=counts(labels[i,0],initial>threshold)["fp"]
    ref,_=exact_budget(initial,labels[i,0],budget)
    records={}
    for name in modes:
        p=loaded(root,"dev",name,i,labels[i])[:,0]
        equal,_=exact_budget(p,labels[i,0],budget)
        fixed,cut=strict_budget(p,labels[i,0],budget)
        records[name]=dict(cutoff=cut,matched= comparison(labels[i,0],ref,equal),strict=comparison(labels[i,0],ref,fixed))
    primary=max(modes,key=lambda name:(records[name]["matched"]["delta_tp"],name=="control"))
    strongest_head=max((name for name in modes if name!="control"),key=lambda name:records[name]["matched"]["delta_tp"])
    write_json(root/"policy.json",dict(selected=primary,best_new_head=strongest_head,base_threshold=threshold,dev_fp=budget,dev=records))

    score_project(root,"test",modes)
    i=parts["test"]
    initial=raw("test")["compact"]
    orig=initial>threshold
    gold=labels[i,0]
    budget=counts(gold,orig)["fp"]
    rank,_=exact_budget(initial,gold,budget)
    control_p=loaded(root,"test","control",i,labels[i])[:,0]
    control,_=exact_budget(control_p,gold,budget)
    cluster=group_codes(zip(projects[i],[paths[j] for j in i],strict=True))
    output={}
    for name,row in records.items():
        p=loaded(root,"test",name,i,labels[i])[:,0]
        equal,_=exact_budget(p,gold,budget)
        frozen=p>row["cutoff"]
        output[name]=dict(matched=comparison(gold,rank,equal),frozen=comparison(gold,orig,frozen),
                     vs_control=comparison(gold,control,equal),interval=grouped_delta(gold,orig,frozen,cluster),
                     per_project={g:comparison(gold[m],rank[m],equal[m]) for g in np.unique(projects[i]) if (m:=projects[i]==g).any()})
        with (root/f"test_{name}.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","label","p_ref","probability","reference_equal","candidate_equal","control_equal","reference_frozen","candidate_frozen"))
            w.writerows(zip(i,projects[i],gold,initial,p,rank.astype(int),equal.astype(int),control.astype(int),orig.astype(int),
                            frozen.astype(int),strict=True))
        print(name,output[name]["matched"],output[name]["frozen"],flush=True)
    write_json(root/"results.json",dict(test_rows=len(i),fp=budget,selected=primary,best_new_head=strongest_head,modes=output))


def project(seed,study):
    directory,modes={"head":(PROJECT,MODES),"memory":(MEMORY,MEMORY_MODES),"ranking":(RANKING,RANK_MODES)}[study]
    root=directory/str(seed)
    fit("projects",root,modes,seed)
    score_project(root,"dev",modes)
    project_eval(root,modes)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--study",choices=("head","memory","ranking"),default="head")
    args=p.parse_args()
    project(args.seed,args.study)
