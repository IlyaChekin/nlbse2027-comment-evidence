"""Continue the same compact model with matched out-of-fold supervision."""

import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from crossfit import HERE, ROOT, checksum, load_backbones, write_json
from evaluate import inputs, raw, exact_budget, counts, grouped_delta, floor_f1
from mine import OUT as MINING
from students import group_codes


OUT = HERE/"precision"
MODES = ("control", "fp90", "joint_errors")
ADVISOR = "advisor"


def source_data(ids, y):
    path = MINING/"scores.npz"
    manifest = json.loads((MINING/"manifest.json").read_text())
    if checksum(path) != manifest["sha256"]:
        raise ValueError("Out-of-fold probabilities changed")
    with np.load(path) as z:
        if not np.array_equal(ids,z["ids"]) or not np.array_equal(y[ids],z["y"]):
            raise ValueError("Out-of-fold scores differ from the training rows")
        return {name:z[name] for name in ("unix","compact","fp_80","fp_90","fn_05")}, manifest["sha256"]


def fit(mode, seed):
    code = load_backbones()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    texts, y, _, _, parts = inputs()
    ids = parts["fit"]
    dataset = [texts[i] for i in ids]
    gold = y[ids]
    oof, oof_sha = source_data(ids,y)
    model,tok = code.frozen("compact")
    original = ROOT/"round13/backbones/compact/selected.json"
    for block in model.local:
        block.use_shifted=False
    model.train()
    tokens = code.small.encode(dataset,tok)
    side = torch.as_tensor(code.students.lexical_features(dataset),device="cuda")
    label = torch.as_tensor(gold,device="cuda",dtype=torch.float32)
    other = torch.as_tensor(oof["unix"],device="cuda").clamp(1e-5,1-1e-5)
    temperature = 1.7
    target = (other.logit()/temperature).sigmoid()
    pos = torch.as_tensor(np.clip((len(gold)-gold.sum(0))/gold.sum(0),1,5),device="cuda",dtype=torch.float32)
    masks = {k:torch.as_tensor(oof[k],device="cuda") for k in ("fp_80","fp_90","fn_05")}
    if mode == ADVISOR:
        student_p = torch.as_tensor(oof["compact"][:,0],device="cuda")
        teacher_p = other[:,0]
        teacher_correct = torch.where(label[:,0]>0,teacher_p,1-teacher_p)
        student_correct = torch.where(label[:,0]>0,student_p,1-student_p)
        gain = ((teacher_correct-student_correct-.1)/.9).clamp(0,1)
        hard_weight = .7+1.2*gain
        soft_weight = (.3+.6*gain)*(teacher_correct/.33).clamp(0,1)
    optimizer = torch.optim.AdamW([
        {"params":model.embed.parameters(),"lr":.00016},
        {"params":[p for n,p in model.named_parameters() if not n.startswith("embed.")],"lr":.00035},
    ],weight_decay=.025,fused=True)
    indices = list(range(len(ids)))
    rng.shuffle(indices)
    batches = list(code.students.batch_plan(indices,tokens,128))
    rng.shuffle(batches)
    schedule = get_linear_schedule_with_warmup(optimizer,int(.04*len(batches)),len(batches))
    started = time.perf_counter()
    for step, batch in enumerate(batches,1):
        x, m = code.students.batch(batch,tokens,tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            prediction = model(x,m,side[batch])
            supervised = F.binary_cross_entropy_with_logits(prediction,label[batch],pos_weight=pos,reduction="none")
            distilled = F.binary_cross_entropy_with_logits(prediction/temperature,target[batch],reduction="none")
            both = .7*supervised + .3*temperature**2*distilled
            vulnerability = both[:,0]
            if mode == "fp90":
                vulnerability = torch.where(masks["fp_90"][batch],5.*supervised[:,0],vulnerability)
            elif mode == "joint_errors":
                vulnerability = torch.where(masks["fp_80"][batch],2.*supervised[:,0],vulnerability)
                vulnerability = torch.where(masks["fp_90"][batch],4.*supervised[:,0],vulnerability)
                vulnerability = torch.where(masks["fn_05"][batch],3.*supervised[:,0],vulnerability)
            elif mode == ADVISOR:
                vulnerability = hard_weight[batch]*supervised[:,0]+soft_weight[batch]*temperature**2*distilled[:,0]
                shared_error = masks["fp_90"][batch] | masks["fn_05"][batch]
                vulnerability = torch.where(shared_error,3.*supervised[:,0],vulnerability)
            loss = (vulnerability.mean()+both[:,1].mean())/2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        schedule.step()
        if step%400==0:
            print(mode,step,len(batches),round(time.perf_counter()-started,2),flush=True)
    folder = OUT/f"{seed}_{mode}"
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint = folder/"checkpoint.pt"
    pending = checkpoint.with_suffix(".part")
    torch.save(model.state_dict(),pending)
    pending.replace(checkpoint)
    write_json(folder/"config.json",dict(mode=mode,seed=seed,epochs=1,rows=len(gold),nbatches=len(batches),
               out_of_fold_sha256=oof_sha,initial_checkpoint_manifest_sha256=checksum(original),
               checkpoint_sha256=checksum(checkpoint),seconds=time.perf_counter()-started))


def score(mode, seed, part):
    code = load_backbones()
    texts,y,_,_,roles=inputs()
    ids=roles[part]
    folder=OUT/f"{seed}_{mode}"
    path=folder/"checkpoint.pt"
    if checksum(path)!=json.loads((folder/"config.json").read_text())["checkpoint_sha256"]:
        raise ValueError("Reweighted checkpoint changed")
    model,tok=code.frozen("compact")
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    p,c=code.score("compact",model,tok,[texts[i] for i in ids])
    target = folder/f"{part}.npz"
    np.savez_compressed(target,ids=ids,y=y[ids],p=p[:,0])
    write_json(target.with_suffix(".json"),dict(cost=c,sha256=checksum(target)))
    print(mode,part,c,flush=True)


def read_scores(mode, seed, part, ids, label):
    target=OUT/f"{seed}_{mode}"/f"{part}.npz"
    if checksum(target)!=json.loads(target.with_suffix(".json").read_text())["sha256"]:
        raise ValueError("Model predictions changed")
    with np.load(target) as z:
        if not np.array_equal(ids,z["ids"]) or not np.array_equal(label,z["y"]):
            raise ValueError(f"Unexpected rows for {mode}/{part}")
        return z["p"]


def summary(y, base, cand, target):
    a,b=counts(y,base),counts(y,cand)
    return dict(before=a,after=b,delta_tp=b["tp"]-a["tp"],delta_fp=b["fp"]-a["fp"],
                f1_before=floor_f1(y,base),f1_after=floor_f1(y,cand),
                corrected_confident_fp=int((base&~cand&target).sum()),
                removed_fp=int((base&~cand&(y==0)).sum()),added_fp=int((~base&cand&(y==0)).sum()))


def tune(seed, study="precision"):
    _,y,_,_,parts=inputs()
    ids=parts["dev"]
    p=raw("dev")
    gold=y[ids,0]
    threshold=json.loads((ROOT/"round13/study/policy.json").read_text())["compact_threshold"]
    hard=(gold==0)&(p["compact"]>.9)&(p["unix"]>.9)
    budget=counts(gold,p["compact"]>threshold)["fp"]
    base, _=exact_budget(p["compact"],gold,budget)
    control=read_scores("control",seed,"dev",ids,y[ids])
    control_rank,_=exact_budget(control,gold,budget)
    recipe={}
    modes = ("control", ADVISOR) if study==ADVISOR else MODES
    for name in modes:
        q=read_scores(name,seed,"dev",ids,y[ids])
        fixed,t=exact_budget(q,gold,budget)
        recipe[name]=dict(threshold=t,reference=summary(gold,base,fixed,hard),control=summary(gold,control_rank,fixed,hard))
    winner=max(modes,key=lambda n:(recipe[n]["reference"]["delta_tp"],recipe[n]["reference"]["corrected_confident_fp"]))
    tag=f"{study}_{seed}" if study!= "precision" else str(seed)
    write_json(OUT/f"policy_{tag}.json",dict(seed=seed,study=study,tag=tag,chosen=winner,base_threshold=threshold,
                                            dev_budget=budget,modes=recipe))


def test(seed, study="precision"):
    _,y,projects,paths,parts=inputs()
    tag=f"{study}_{seed}" if study!="precision" else str(seed)
    policy=json.loads((OUT/f"policy_{tag}.json").read_text())
    t=parts["test"]
    gold=y[t,0]
    initial=raw("test")
    original=initial["compact"]>policy["base_threshold"]
    hard=(gold==0)&(initial["compact"]>.9)&(initial["unix"]>.9)
    file_ids=group_codes(zip(projects[t],[paths[j] for j in t],strict=True))
    fp=counts(gold,original)["fp"]
    rank,_=exact_budget(initial["compact"],gold,fp)
    control=read_scores("control",seed,"test",t,y[t])
    control_equal,_=exact_budget(control,gold,fp)
    report={}
    for name in policy["modes"]:
        q=read_scores(name,seed,"test",t,y[t])
        equal,_=exact_budget(q,gold,fp)
        frozen=q>policy["modes"][name]["threshold"]
        report[name]=dict(equal=summary(gold,rank,equal,hard), against_control=summary(gold,control_equal,equal,hard),
                          frozen=summary(gold,original,frozen,hard), fixed_cutoff_interval=grouped_delta(gold,original,frozen,file_ids),
                          by_project={g:summary(gold[ix],rank[ix],equal[ix],hard[ix])
                                      for g in np.unique(projects[t]) if (ix:=projects[t]==g).any()})
        csv_name="test_advisor.csv" if study==ADVISOR else "test.csv"
        with (OUT/f"{seed}_{name}"/csv_name).open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(("row","project","label","reference_probability","candidate_probability","reference_frozen",
                             "reference_equal","candidate_frozen","candidate_equal","shared_high_fp"))
            writer.writerows(zip(t,projects[t],gold,initial["compact"],q,original.astype(int),rank.astype(int),
                                 frozen.astype(int),equal.astype(int),hard.astype(int),strict=True))
    write_json(OUT/f"results_{tag}.json",dict(seed=seed,study=study,tag=tag,testing_rows=len(t),
                                             target_hard_fp=int(hard.sum()),modes=report))
    print({k:{"equal":v["equal"],"frozen":v["frozen"]} for k,v in report.items()},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("fit","score","tune","test"))
    ap.add_argument("--mode",choices=(*MODES, ADVISOR))
    ap.add_argument("--part",choices=("dev","test"))
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--study",choices=("precision",ADVISOR),default="precision")
    args=ap.parse_args()
    if args.stage in {"fit","score"} and (not args.mode or (args.stage=="score" and not args.part)):
        ap.error("fit and score need --mode, and score needs --part")
    if args.stage=="fit":
        fit(args.mode,args.seed)
    elif args.stage=="score":
        score(args.mode,args.seed,args.part)
    else:
        {"tune":tune,"test":test}[args.stage](args.seed,args.study)
