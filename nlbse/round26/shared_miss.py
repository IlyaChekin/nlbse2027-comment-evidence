"""Test a compact specialist trained on shared out-of-file errors."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
import random
import time
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from disagreement import (HERE,PARTS,f,checksum,json_at,write_json,encoded_context,
                           match,clip_logit,trained_logistic,LIMIT)
from evaluate_rules import infer
from train import cohorts,fit_ids

OUT=HERE/"specialist"
MODES=("plain","shared")


def focus(rotation,indices):
    directory=HERE/"error_census"/rotation
    path=directory/"evidence.npz"
    expected=json_at(directory/"census.json")
    if checksum(path)!=expected["evidence_sha256"]:
        raise ValueError("Out-of-file specialist targets changed")
    with np.load(path) as z:
        row,i,j=np.intersect1d(indices,z["ids"],return_indices=True)
        p=z["scores"][j]
        y=z["gold"][j].astype(bool)
    hard_negative=(~y)&(p[:,0]>=LIMIT)&(p[:,1]>=LIMIT)
    hard_positive=y&(p[:,0]<LIMIT)&(p[:,1]<LIMIT)
    weight=np.ones(len(indices),dtype=np.float32)
    weight[i[hard_negative]]=3.
    weight[i[hard_positive]]=4.
    return weight,dict(source=expected["evidence_sha256"],covered=len(row),
                     shared_false_alarm=int(hard_negative.sum()),shared_missed=int(hard_positive.sum()))


def fit(rotation):
    selected=np.asarray(fit_ids(rotation))
    weak,y,digest=cohorts()
    ids=selected[weak[selected]==0]
    texts,labels,_,_,_=f.inputs()
    local=[texts[i] for i in ids]
    if not np.array_equal(y[ids],labels[ids,0]):
        raise ValueError("Wrong outcome labels for the specialist")
    tok=f.small.tokenizer()
    sequence=f.tokenize(local,tok)
    side=torch.as_tensor(f.base.lexical_features(local),device="cuda")
    yy=torch.as_tensor(y[ids],device="cuda",dtype=torch.float32)
    weights,origin=focus(rotation,ids)
    reps=torch.from_numpy(weights).cuda()
    balance=min(5.,float(((len(ids)-yy.sum())/yy.sum().clamp_min(1)).sqrt()))
    out=OUT/rotation
    out.mkdir(parents=True,exist_ok=True)
    summary={}
    for mode in MODES:
        torch.manual_seed(42)
        rng=random.Random(42)
        model=f.pipe(rotation,tok).encoder
        for layer in model.encoder.local:
            layer.use_shifted=False
        groups=[dict(params=model.encoder.embed.parameters(),lr=.00008),
                dict(params=[p for n,p in model.named_parameters()
                             if n!="encoder.embed.weight"],lr=.00024)]
        opt=torch.optim.AdamW(groups,fused=True,weight_decay=.025)
        order=list(range(len(ids)))
        rng.shuffle(order)
        phases=list(f.base.batch_plan(order,sequence,f.small.BATCH))
        rng.shuffle(phases)
        lr=get_linear_schedule_with_warmup(opt,int(.04*len(phases)),len(phases))
        time0=time.perf_counter()
        model.train()
        for rows in phases:
            x,mask=f.base.batch(rows,sequence,tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                probability=model(x,mask,side[rows])[:,22]
                loss=F.binary_cross_entropy_with_logits(probability,yy[rows],pos_weight=yy.new_tensor(balance),reduction="none")
                if mode=="shared":
                    w=reps[rows]
                    objective=(loss*w).sum()/w.sum()
                else:
                    objective=loss.mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            lr.step()
        file=out/f"{mode}.pt"
        torch.save(model.state_dict(),file)
        summary[mode]=dict(sec=time.perf_counter()-time0,sha256=checksum(file))
        print(rotation,mode,summary[mode],flush=True)
    write_json(out/"fit.json",dict(rows=len(ids),positive=int(yy.sum()),cohort_sha256=digest,mode=summary,
                    oof_targets=origin,seed=42,code_sha256=checksum(Path(__file__))))


def load_expert(rotation,mode,tok):
    location=OUT/rotation
    path=location/f"{mode}.pt"
    if checksum(path)!=json_at(location/"fit.json")["mode"][mode]["sha256"]:
        raise ValueError("Stored specialist weights changed")
    model=f.pipe(rotation,tok).encoder
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    model.eval()
    return SimpleNamespace(encoder=model,tok=tok,mode="control")


@torch.inference_mode()
def obtain(pipe,examples):
    extra=np.empty((len(examples),22),dtype=np.float32)
    _,cost=infer(pipe,examples,encoding=f.tokenize,aux_out=extra)
    return extra[:,-2],cost


def test(rotation):
    d,b,fit_rows,cal,review,old=encoded_context(rotation)
    if b is None:
        b=d
    control,base=old[0]
    observed,reference=old[-1]
    dev=[s for s,k in zip(d.texts,d.active,strict=True) if k]
    sample=dev if b is d else [s for s,k in zip(b.texts,b.active,strict=True) if k]
    tok=f.small.tokenizer()
    work=OUT/rotation
    records,weights={},{}
    for mode in MODES:
        pipe=load_expert(rotation,mode,tok)
        p,ct=obtain(pipe,dev)
        q,t=(p,ct) if b is d else obtain(pipe,sample)
        batch_cost={}
        m=np.column_stack((base[d.active],clip_logit(p)))
        v=np.column_stack((reference[b.active],clip_logit(q)))
        heads=trained_logistic(m[fit_rows[d.active]],d.frame["y"][fit_rows&d.active])
        scores=clip_logit(heads.predict_proba(m)[:,1])
        results=clip_logit(heads.predict_proba(v)[:,1])
        weights[mode]=heads
        for width,weight in product((.4,1.,2.),(.4,.8)):
            take=np.abs(base[d.active])<width
            ask=np.abs(reference[b.active])<width
            inside=d.active.copy()
            inside[d.active]=take
            accessed=b.active.copy()
            accessed[b.active]=ask
            qa=(1-weight)*base[d.active]+weight*scores
            qb=(1-weight)*reference[b.active]+weight*results
            qdev=qa[take]
            qtest=qb[ask]
            label=d.frame["y"][inside&cal]
            fp=int((control[inside&cal]&(label==0)).sum())
            _,cut=f.strict_budget(qdev[cal[inside]],label,fp)
            adjusted=control.copy()
            adjusted[inside]=qdev>cut
            internal=f.comparison(d.frame["y"][cal],control[cal],adjusted[cal])
            dest=observed.copy()
            dest[accessed]=qtest>cut
            held=np.ones(len(dest),dtype=bool) if review is None else review
            evidence=dict(calibration=internal,frozen=f.comparison(b.frame["y"][held],observed[held],dest[held]),
                          eval_calls=int((accessed&held).sum()),cal_calls=int((inside&cal).sum()),
                          threshold=cut,width=width,weight=weight,measured_query_full_negative_cohort=t)
            if review is None:
                votes,each=match(b.frame["y"],b.frame["project"],observed,accessed,qtest)
                evidence["matched"]=f.comparison(b.frame["y"],observed,votes)
                evidence["per_project"]=each
            else:
                if width not in batch_cost:
                    chosen=[s for s,k in zip(d.texts,inside&cal,strict=True) if k]
                    if chosen:
                        obtain(pipe,chosen)
                        costs=[obtain(pipe,chosen)[1] for _ in range(3)]
                        batch_cost[width]={k:median(c[k] for c in costs) for k in costs[0]}
                    else:
                        batch_cost[width]=dict(forward_seconds=0.,wall_seconds=0.,analytic_gflops=0.)
                used=batch_cost[width]
                boost=.3*(internal["f1_after"]-internal["f1_before"])
                tax=.2*f.PUBLIC_N/cal.sum()*(used["forward_seconds"]/1000+used["analytic_gflops"]/3e6)
                evidence["measured_cal"] = used
                evidence["proxy"] = float(boost-tax)
            key=f"{mode}_{width:g}_{weight:g}"
            records[key]=evidence
            np.savez_compressed(work/f"{key}.npz",ids=b.frame["ids"][held],control=observed[held],
                           frozen=dest[held],active=accessed[held],score=qtest[held[accessed]])
    dump=work/"readouts.joblib"
    joblib.dump(weights,dump,compress=3)
    write_json(work/"results.json",dict(options=records,weights_sha256=checksum(dump)))
    print(rotation,sorted([(r.get("matched",r["frozen"])["delta_tp"],k,r["calibration"]["delta_tp"])
                            for k,r in records.items()],reverse=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    p.add_argument("stage",choices=("fit","evaluate"))
    args=p.parse_args()
    {"fit":fit,"evaluate":test}[args.stage](args.rotation)
