"""Continue the released encoder with out-of-fold teacher targets on the official fit."""

import argparse
import csv
import json
import random
from statistics import median
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from crossfit import HERE, ROOT, checksum, read_partition, write_json
from evaluate import counts, exact_budget
from mine import OUT as MINING
from normalize_code import parallel as canonicalize
from pipeline import PUBLIC, TRAIN, load, metric, select_indices
from precision import summary
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from students import dataset, get_fit, lexical_features, batch_plan, batch, predict_bits, group_codes
import efficient_aug
import small


OUT=HERE/"official_distill"
NAMES=("original_teacher","overlap_in_fit","out_of_fold")


def recipe():
    return json.loads((efficient_aug.OUT/"policy.json").read_text())["selected"]["selective"]


def starting_checkpoint_sha256():
    path=ROOT/"round12/backbones/canonical_official/selected.json"
    return json.loads(path.read_text())["sha256"]


def overlap_sources(ids,y):
    path=MINING/"scores.npz"
    reference=ROOT/"round13/backbones/unix/teacher.npz"
    if checksum(path)!=json.loads((MINING/"manifest.json").read_text())["sha256"]:
        raise ValueError("Out-of-fold scores changed")
    if checksum(reference)!=json.loads(reference.with_suffix(".json").read_text())["sha256"]:
        raise ValueError("In-fit scores changed")
    with np.load(path) as o,np.load(reference) as i:
        if not np.array_equal(o["ids"],i["ids"]) or not np.array_equal(o["y"],i["y"]):
            raise ValueError("In-fit and out-of-fold teachers do not cover the same examples")
        keep,local,base=np.intersect1d(o["ids"],ids,return_indices=True)
        if not np.array_equal(o["y"][local],y[base]):
            raise ValueError("Official fit labels differ in the teacher overlap")
        return base,{"overlap_in_fit":i["probabilities"][local,0],"out_of_fold":o["unix"][local,0]},len(keep)


def fit():
    OUT.mkdir(exist_ok=True)
    texts,y,teacher=get_fit()
    ids,_=select_indices(False)
    subset,replacements,n=overlap_sources(ids,y)
    initial=efficient_aug.startup()
    orig=small.encode(texts,initial.tok)
    other=small.encode(canonicalize(texts),initial.tok)
    lexical=torch.as_tensor(lexical_features(texts),device="cuda")
    gold=torch.as_tensor(y,device="cuda",dtype=torch.float32)
    pos=torch.as_tensor(np.clip((len(y)-y.sum(0))/y.sum(0),1,5),device="cuda",dtype=torch.float32)
    record={}
    for mode in NAMES:
        torch.manual_seed(42)
        rng=random.Random(42)
        pipe=efficient_aug.startup()
        model,tok=pipe.encoder,pipe.tok
        for block in model.local:
            block.use_shifted=False
        model.train()
        target=teacher.copy()
        if mode in replacements:
            target[subset,0]=.5*target[subset,0]+.5*replacements[mode]
        t=torch.as_tensor(target,device="cuda").clamp(1e-5,1-1e-5)
        softened=(t.logit()/1.7).sigmoid()
        sequence=[b if rng.random()<.45 else a for a,b in zip(orig,other,strict=True)]
        order=list(range(len(ids)))
        rng.shuffle(order)
        plans=list(batch_plan(order,sequence,128))
        rng.shuffle(plans)
        opt=torch.optim.AdamW([
            {"params":model.embed.parameters(),"lr":.00012},
            {"params":[p for k,p in model.named_parameters() if not k.startswith("embed.")],"lr":.00026},
        ],weight_decay=.025,fused=True)
        schedule=get_linear_schedule_with_warmup(opt,int(.04*len(plans)),len(plans))
        start=time.perf_counter()
        for step,rows in enumerate(plans,1):
            x,m=batch(rows,sequence,tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=model(x,m,lexical[rows])
                supervised=F.binary_cross_entropy_with_logits(p,gold[rows],pos_weight=pos)
                soft=F.binary_cross_entropy_with_logits(p/1.7,softened[rows])
                loss=.7*supervised+.3*1.7**2*soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            schedule.step()
            if step%500==0:
                print(mode,step,len(plans),round(time.perf_counter()-start,1),flush=True)
        path=OUT/f"{mode}.pt"
        pending=path.with_suffix(".part")
        torch.save(model.state_dict(),pending)
        pending.replace(path)
        record[mode]=dict(checkpoint=path.name,sha256=checksum(path),seconds=time.perf_counter()-start,steps=len(plans))
    write_json(OUT/"fit.json",dict(variants=record,fit_rows=len(ids),teacher_overlap=n,teacher_mix=.5,seed=42,
             official_fit_sha256=checksum(ROOT/"data/split_manifest.csv"),mining_sha256=checksum(MINING/"scores.npz"),
             base_sha256=starting_checkpoint_sha256()))


def pipe_for(name,fit_record):
    p=efficient_aug.startup()
    if name!="release":
        cfg=fit_record["variants"][name]
        path=OUT/cfg["checkpoint"]
        if checksum(path)!=cfg["sha256"]:
            raise ValueError("Official continuation checkpoint changed")
        p.encoder.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return p


def tune():
    _,_,texts,y,ids,_=dataset(full=True)
    ids=np.asarray(ids)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if not np.array_equal(f["rows"],ids):
            raise ValueError("Official file calibration IDs differ")
        folds=f["folds"]
    partition=read_partition()
    safe=np.isin(ids,np.concatenate([partition[k] for k in ("probe","dev","test")]))
    if np.isin(ids[safe],partition["fit"]).any():
        raise ValueError("Calibration contains data used to train the out-of-fold teachers")
    cal,review=safe&(folds==3),safe&(folds==4)
    fit_record=json.loads((OUT/"fit.json").read_text())
    base,_=efficient_aug.run(pipe_for("release",fit_record),texts,recipe())
    frozen=base[:,0]>recipe()["threshold"]
    budget=counts(y[cal,0],frozen[cal])["fp"]
    ref,_=exact_budget(base[cal,0],y[cal,0],budget)
    details={"release":dict(threshold=recipe()["threshold"],delta_tp=0,rank_f1=metric(y[cal],np.column_stack((ref,base[cal,1]>.5)))["vulnerability"]["f1"])}
    for name in NAMES:
        q,c=efficient_aug.run(pipe_for(name,fit_record),texts,recipe())
        if not np.array_equal(q[:,1]>.5,base[:,1]>.5):
            raise ValueError("Continued encoder changed the MAT rule")
        match,t=exact_budget(q[cal,0],y[cal,0],budget)
        delta=summary(y[cal,0],ref,match,np.zeros(cal.sum(),dtype=bool))
        details[name]=dict(threshold=t,delta_tp=delta["delta_tp"],rank_f1=delta["f1_after"],development=delta,
                           review=summary(y[review,0],frozen[review],q[review,0]>t,np.zeros(review.sum(),dtype=bool)),cost=c)
        print(name,details[name],flush=True)
    selected=max(details,key=lambda name:(details[name]["delta_tp"],name=="release"))
    write_json(OUT/"policy.json",dict(selected=selected,records=details,calibration_rows=ids[cal].tolist(),review_rows=ids[review].tolist(),
               budget=budget,split_sha256=checksum(ROOT/"round13/partition.npz")))


def public():
    policy=json.loads((OUT/"policy.json").read_text())
    winner=policy["selected"]
    source=json.loads((OUT/"fit.json").read_text())
    text,y,m=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    old=predict_bits(efficient_aug.OUT/"selective/predictions.csv",ids,{"thresholds":(recipe()["threshold"],.5)})
    paths=group_codes(zip(m["Projectname"],m["Filepath"],strict=True))
    engines={name:pipe_for(name,source) for name in dict.fromkeys(("release",winner))}
    trials={k:[] for k in engines}
    decisions={}
    for pipe in engines.values():
        efficient_aug.run(pipe,text[:2048],recipe())
    for rep in range(3):
        for name in (engines if rep%2==0 else reversed(engines)):
            raw,timing=efficient_aug.run(engines[name],text,recipe())
            bits=raw>(policy["records"][name]["threshold"],.5)
            if name=="release" and not np.array_equal(bits,old):
                raise ValueError("Released predictions changed")
            if name in decisions and not np.array_equal(bits,decisions[name]):
                raise ValueError("Public decisions changed between repeats")
            trials[name].append(timing)
            decisions[name]=bits
            print(name,timing,flush=True)
    result={}
    for name,times in trials.items():
        cost={k:median(t[k] for t in times) for k in times[0]}
        bits=decisions[name]
        quality=metric(y,bits)
        changes=summary(y[:,0],old[:,0],bits[:,0],np.zeros(len(y),dtype=bool))
        result[name]=dict(quality=quality,score=competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                          cost=cost,trials=times,vulnerability=changes,interval=bootstrap_delta(y,old,bits,paths))
        with (OUT/f"{name}_public.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(("row","truth_vulnerability","truth_mat","vulnerability","mat"))
            w.writerows(zip(ids,y[:,0],y[:,1],bits[:,0].astype(int),bits[:,1].astype(int),strict=True))
    write_json(OUT/"results.json",result)
    print({k:v["score"] for k,v in result.items()},flush=True)


def fixed_fp():
    policy=json.loads((OUT/"policy.json").read_text())
    source=json.loads((OUT/"fit.json").read_text())
    public=json.loads((OUT/"results.json").read_text())
    names=tuple(public)
    text,y,_=load(PUBLIC)
    ids=list(range(len(y)))
    p={}
    for name in names:
        score,_=efficient_aug.run(pipe_for(name,source),text,recipe())
        with (OUT/f"{name}_public.csv").open(newline="") as f:
            rows=list(csv.DictReader(f))
        expected=np.asarray([[int(r[k]) for k in ("vulnerability","mat")] for r in rows])
        if [int(r["row"]) for r in rows]!=ids or not np.array_equal(score>(policy["records"][name]["threshold"],.5),expected):
            raise ValueError("Frozen predictions differ from the published benchmark")
        p[name]=score[:,0]
    budget=public["release"]["vulnerability"]["before"]["fp"]
    base,_=exact_budget(p["release"],y[:,0],budget)
    out={}
    for name,prob in p.items():
        bits,_=exact_budget(prob,y[:,0],budget)
        out[name]=summary(y[:,0],base,bits,np.zeros(len(y),dtype=bool))
        with (OUT/f"{name}_equal_fp.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(("row","truth_vulnerability","probability","matched","reference_matched"))
            w.writerows(zip(ids,y[:,0],prob,bits.astype(int),base.astype(int),strict=True))
    write_json(OUT/"fixed_fp.json",dict(budget=budget,matched=out))
    print(out,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fit","tune","public","fixed_fp"))
    {"fit":fit,"tune":tune,"public":public,"fixed_fp":fixed_fp}[p.parse_args().stage]()
