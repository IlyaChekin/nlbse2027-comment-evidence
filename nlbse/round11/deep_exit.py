"""Jointly train two-, four- and six-block exits in one shared CodeT5-small encoder."""
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
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"round10"))
from students import (ROOT, PUBLIC, batch, batch_plan, calibrated, dataset, get_fit, group_codes,
                      lexical_features, load, load_model, load_selected, metric, ordered_batches, predict_bits,
                      save, sha256, tokenize, write_json)
from student_context import feature_block, load_policy, probability
from fused_attention import fused_block, padded_stride_bias, pool, prefix_fused
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT = HERE/"deep_exit"
DEPTHS = (2,4,6)
BATCH = 80
FAMILY = "t5_small"


class Exits(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        width = backbone.config.d_model
        self.early = nn.ModuleList(nn.Sequential(nn.LayerNorm(width+5), nn.Dropout(.08), nn.Linear(width+5,2)) for _ in (2,4))

    def logits(self, hidden, valid, side, stage):
        stack = self.backbone.encoder.encoder
        if stage < 2:
            representation = pool(stack, hidden, valid)
            return self.early[stage](torch.cat((representation,side),dim=-1))
        representation = (hidden*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True)
        return self.backbone.head(representation)+self.backbone.side(side)

    def forward(self, tokens, mask, side):
        result = self.backbone.encoder(input_ids=tokens,attention_mask=mask,output_hidden_states=True)
        return [self.logits(result.hidden_states[d],mask,side,j) for j,d in enumerate(DEPTHS)]


def fit(epochs):
    torch.manual_seed(42)
    rng = random.Random(42)
    backbone,tok,_ = load_selected(FAMILY)
    model = Exits(backbone).cuda()
    texts, y, soft = get_fit()
    ids = tokenize(texts,tok,FAMILY)
    side = torch.as_tensor(lexical_features(texts),device="cuda")
    labels = torch.as_tensor(y,device="cuda",dtype=torch.float32)
    targets = torch.sigmoid(torch.logit(torch.as_tensor(soft,device="cuda").clamp(1e-5,1-1e-5))/1.8)
    pos = torch.as_tensor(np.clip((len(y)-y.sum(0))/y.sum(0),1,5),device="cuda",dtype=torch.float32)
    opt = torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":1.2e-5},
                             {"params":model.early.parameters(),"lr":4e-4}],weight_decay=.01,fused=True)
    order=list(range(len(y)))
    total=epochs*len(list(batch_plan(order,ids,BATCH)))
    schedule=get_linear_schedule_with_warmup(opt,int(total*.04),total)
    ref = json.loads((ROOT/"round10/runs/t5_small/selected.json").read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"config.json",dict(epochs=epochs,batch=BATCH,fit_rows=len(y),seed=42,depths=DEPTHS,
              initial_sha256=ref["settings"]["sha256"],loss_weights=[.2,.25,.55],temperature=1.8))
    began=time.perf_counter()
    step=0
    for epoch in range(1,epochs+1):
        rng.shuffle(order)
        batches=list(batch_plan(order,ids,BATCH))
        rng.shuffle(batches)
        model.train()
        for rows in batches:
            x,mask=batch(rows,ids,tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                heads=model(x,mask,side[rows])
                loss=0.
                for weight,logits in zip((.2,.25,.55),heads,strict=True):
                    hard=F.binary_cross_entropy_with_logits(logits,labels[rows],pos_weight=pos)
                    soft=F.binary_cross_entropy_with_logits(logits/1.8,targets[rows])
                    loss+=weight*(.68*hard+.32*1.8**2*soft)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            schedule.step()
            step+=1
            if step%500==0:
                print(step,total,float(loss.detach()),round(time.perf_counter()-began,1),flush=True)
        temp=OUT/f"epoch{epoch}.part"
        torch.save(model.state_dict(),temp)
        temp.replace(temp.with_suffix(".pt"))
        print("epoch",epoch,flush=True)


def block_flops(width):
    return (2*width*(4*512**2+2*512*2048)+4*width**2*512)/1.e9


@torch.inference_mode()
def evaluate(model,tok,texts,policy=None):
    stack = model.backbone.encoder.encoder
    comment=load_policy()["lexical_control"]
    model.eval()
    all_outputs = policy is None
    p=np.empty((len(texts),3 if all_outputs else 2),dtype=np.float32)
    unit=np.empty(len(texts),dtype=np.float64)
    exits=np.zeros(3,dtype=np.int64)
    volume, forward = 0.,0.
    bias_cache={}
    began=time.perf_counter()

    def prepare(i):
        return feature_block(texts[i:i+2048],tok,deferred_offsets=True)

    with torch.autocast("cuda",dtype=torch.float16):
        for offset,(seq,side,extra) in ordered_batches(range(0,len(texts),2048),prepare,True):
            side=torch.as_tensor(side,device="cuda")
            for rows in batch_plan(list(range(len(seq))),seq,BATCH):
                tokens,mask=batch(rows,seq,tok.pad_token_id)
                ix=offset+np.asarray(rows)
                tick=time.perf_counter()
                states,bias=prefix_fused(model.backbone,tokens,mask,2,bias_cache)
                active=np.arange(len(rows))
                q=side[rows]
                # Every later stage receives only the gathered states of still active examples.
                unit[ix]=block_flops(tokens.shape[1])
                for stage,depth in enumerate(DEPTHS):
                    volume+=2*len(active)*block_flops(mask.shape[1])
                    hidden=stack.final_layer_norm(states) if stage==2 else states
                    local=model.logits(hidden,mask,q,stage).float().sigmoid().cpu().numpy()
                    if all_outputs:
                        p[ix,stage]=local[:,0]
                        needs=np.ones(len(local),dtype=bool)
                    else:
                        needs=(local[:,0]>policy["lower"][stage]) & (local[:,0]<policy["upper"][stage]) if stage<2 else np.zeros(len(local),dtype=bool)
                        stop=~needs
                        p[ix[active[stop]],0]=(local[stop,0]>policy["thresholds"][stage]).astype(np.float32)
                        exits[stage]+=stop.sum()
                    if stage==2 or not needs.any():
                        break
                    active=active[needs]
                    take=torch.as_tensor(np.flatnonzero(needs),device="cuda")
                    width=max(len(seq[rows[i]]) for i in active)
                    states=states[take,:width]
                    mask=mask[take,:width]
                    q=q[take]
                    bias=bias[take,:,:width,:width]
                    fast_bias=padded_stride_bias(bias)
                    for block in stack.block[depth:depth+2]:
                        states=fused_block(block,states,fast_bias)
                if not all_outputs:
                    final_mat=probability(np.zeros((len(rows),2),dtype=np.float32),extra[rows],comment)
                    p[ix,1]=(final_mat>comment["thresholds"][1]).astype(np.float32)
                forward+=time.perf_counter()-tick
    return p,unit,dict(analytic_gflops=volume,wall_seconds=time.perf_counter()-began,forward_seconds=forward,exits=exits.tolist())


def load_state(model,path):
    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))


def calibrate():
    backbone,tok,_=load_selected(FAMILY)
    model=Exits(backbone).cuda()
    _,_,texts,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as split:
        if ids!=split["rows"].tolist():
            raise ValueError("Unexpected grouped holdout rows")
        fold=split["folds"]
    cal,review=fold==3,fold==4
    all_candidates=[]
    target_volume=254812.
    for path in sorted(OUT.glob("epoch*.pt")):
        load_state(model,path)
        p,unit,time_record=evaluate(model,tok,texts)
        t=[calibrated(y[cal],np.column_stack((column[cal],column[cal])))[0] for column in p.T]
        for lo2 in (.01,.025,.05,.075,.1,.15,.2):
            for hi2 in (.73,.82,.9,.95,.98):
                for lo4 in (.015,.035,.06,.10,.15):
                    for hi4 in (.7,.81,.9,.96):
                        stage4=(p[:,0]>lo2)&(p[:,0]<hi2)
                        stage6=stage4&(p[:,1]>lo4)&(p[:,1]<hi4)
                        v=np.where(stage6,p[:,2]>t[2],np.where(stage4,p[:,1]>t[1],p[:,0]>t[0]))
                        paired=np.column_stack((v,y[:,1]))
                        vulnerability=metric(y[cal],paired[cal])["vulnerability"]["f1"]
                        expected=target_volume*((unit[cal]*(2+2*stage4[cal]+2*stage6[cal])).sum()/(6*unit[cal].sum()))
                        rank=.3*vulnerability-.2*expected/3e6
                        all_candidates.append(dict(checkpoint=path.name,lower=[lo2,lo4],upper=[hi2,hi4],thresholds=t,
                            calibration_vulnerability=vulnerability,estimated_gflops=expected,calibration_objective=rank,
                            review_vulnerability=metric(y[review],paired[review])["vulnerability"]["f1"],
                            review_fraction_deeper=[float(stage4[review].mean()),float(stage6[review].mean())]))
        np.savez_compressed(OUT/f"{path.stem}_holdout.npz",p=p,ids=ids,y=y,folds=fold,gflops_per_block=unit)
    selected=max(all_candidates,key=lambda c:c["calibration_objective"])
    fixed=dict(selected,lower=[0.,0.],upper=[1.,1.])
    write_json(OUT/"policy.json",dict(selected=selected,full_depth=fixed,candidates=all_candidates,
                                     checkpoint_sha256=sha256(OUT/selected["checkpoint"])))
    print(selected,flush=True)


def inference_model():
    config=json.loads((OUT/"policy.json").read_text())
    path=OUT/config["selected"]["checkpoint"]
    if sha256(path)!=config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after tuning")
    backbone,tok=load_model(FAMILY,pretrained=False)
    model=Exits(backbone).cuda()
    load_state(model,path)
    return model,tok,config


def public():
    model,tok,config=inference_model()
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    reference=predict_bits(ROOT/"round10/compact_exit/lexical_deferred/predictions.csv",ids,{"thresholds":(.5,.5)})
    results={}
    for mode in ("selected","full_depth"):
        rule=config[mode]
        evaluate(model,tok,texts[:2048],rule)
        timings=[]
        previous=None
        for _ in range(3):
            p,_,profiles=evaluate(model,tok,texts,rule)
            if previous is not None and not np.array_equal(previous,p):
                raise ValueError("Different decisions across public runs")
            previous=p
            timings.append(profiles)
            print(mode,profiles,flush=True)
        cost={k:median(r[k] for r in timings) for k in ("wall_seconds","forward_seconds","analytic_gflops")}
        m=metric(y,p>.5)
        results[mode]=dict(cost=cost,exits=timings[0]["exits"],metrics=m,timings=timings,
                           score=competition_formula(m["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                           interval=bootstrap_delta(y,reference,p>.5,groups))
        save(OUT/mode,y,ids,cal,p,cost,(.5,.5))
    write_json(OUT/"analysis.json",results)


def predict(source,target):
    with source.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV requires a Function column")
        texts=[row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Every row must supply a function")
    model,tok,config=inference_model()
    p,_,_=evaluate(model,tok,texts,config["selected"])
    with target.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>.5).astype(np.int8).tolist())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","tune","public","predict"))
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.stage=="fit":
        fit(args.epochs)
    elif args.stage=="tune":
        calibrate()
    elif args.stage=="public":
        public()
    else:
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        predict(args.input,args.output)
