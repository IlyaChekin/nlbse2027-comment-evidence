"""Learn WeakSATD rule indicators in the existing compact encoder."""
import argparse
from functools import lru_cache
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R19 = ROOT/"round19"
WEAK = R19/"weak_satd"
sys.path[:0] = [str(R19)]
import transfer as base
from rule_model import RuleEncoder
from transfer import batch, batch_plan, checkpoint, checksum, inputs, json_at, load_original, small, write_json


MODES = ("probe", "joint")
AUX_WEIGHT = .45


@lru_cache(maxsize=1)
def rules():
    path = WEAK/"train_parallel.csv"
    source = json_at(path.with_suffix(".json"))
    if checksum(path) != source["sha256"] or checksum(WEAK/"scan.mjs") != source["code_sha256"]:
        raise ValueError("The versioned WeakSATD labels changed")
    result = np.loadtxt(path, skiprows=1, delimiter=",", dtype=np.int16, usecols=range(1,20))
    if result.shape != (source["rows"],19) or not np.array_equal(
        np.loadtxt(path, skiprows=1, delimiter=",", dtype=np.int32, usecols=0), np.arange(source["rows"])
    ):
        raise ValueError("Rule labels do not align with the training rows")
    y = result > 0
    known = result >= 0
    return np.column_stack((y.any(1),y)), np.column_stack((y.any(1)|known.all(1),known)), source["sha256"]


def fit_ids(rotation):
    if rotation == "official":
        ids, _ = base.select_indices(False)
        return np.asarray(ids)
    with np.load(R19/f"{rotation}.npz") as stored:
        return stored["fit"]


def fit(rotation,seed,mode_list=MODES):
    label, teacher, _, digests, tok, plain, normalized, lexical = base.prepared_training(rotation)
    ids = fit_ids(rotation)
    if not np.array_equal(inputs()[1][ids],label):
        raise ValueError("Teacher labels use a different training order")
    rule, known, target_hash = rules()
    y = torch.as_tensor(rule[ids],device="cuda",dtype=torch.float32)
    valid = torch.as_tensor(known[ids],device="cuda")
    main = torch.as_tensor(label,device="cuda",dtype=torch.float32)
    soft = (torch.as_tensor(teacher,device="cuda").clamp(1e-5,1-1e-5).logit()/1.7).sigmoid()
    main_w = ((len(main)-main.sum(0))/main.sum(0).clamp_min(1)).clamp(1,5)
    weak_w = ((valid.sum(0)-y.sum(0))/y.sum(0).clamp_min(1)).sqrt().clamp(1,5)
    keep = valid & (y.sum(0)>=32)
    lex = torch.as_tensor(lexical,device="cuda")
    source = checkpoint("official" if rotation=="official" else "projects")
    rng = random.Random(seed)
    sequences = [b if rng.random()<.45 else a for a,b in zip(plain,normalized,strict=True)]
    order = list(range(len(sequences)))
    rng.shuffle(order)
    batches = list(batch_plan(order,sequences,small.BATCH))
    rng.shuffle(batches)
    modes = {}
    folder = HERE/rotation/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    existing=folder/"fit.json"
    if existing.exists():
        receipt=json_at(existing)
        if receipt["initialization"]!=source[1] or receipt["weak_sha256"]!=target_hash:
            raise ValueError("Existing student variants use different training inputs")
        modes=receipt["modes"]
    for mode in mode_list:
        torch.manual_seed(seed)
        enc = load_original(tok,"control",source)
        model = RuleEncoder(enc,conditional=mode=="factor").cuda()
        head_params = dict(params=model.rule_head.parameters(),lr=.0008)
        if mode in ("joint","factor"):
            params = [dict(params=enc.embed.parameters(),lr=.00012),
                      dict(params=[p for name,p in enc.named_parameters() if not name.startswith("embed.")],
                           lr=.00025),head_params]
            model.train()
        else:
            enc.requires_grad_(False)
            enc.eval()
            params = [head_params]
        opt = torch.optim.AdamW(params,weight_decay=.025,fused=True)
        lr = get_linear_schedule_with_warmup(opt,int(.04*len(batches)),len(batches))
        torch.manual_seed(seed)
        start = time.perf_counter()
        for step, rows in enumerate(batches,1):
            x,mask = batch(rows,sequences,tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output = model(x,mask,lex[rows])
                cell = F.binary_cross_entropy_with_logits(output[:,2:22],y[rows],pos_weight=weak_w,reduction="none")
                l_union = cell[:,0].masked_fill(~keep[rows,0],0).sum()/keep[rows,0].sum().clamp_min(1)
                l_parts = cell[:,1:].masked_fill(~keep[rows,1:],0).sum()/keep[rows,1:].sum().clamp_min(1)
                loss = .6*l_union+.4*l_parts
                if mode in ("joint","factor"):
                    v = output[:,:2]
                    ce = F.binary_cross_entropy_with_logits(v,main[rows],pos_weight=main_w)
                    kd = F.binary_cross_entropy_with_logits(v/1.7,soft[rows])
                    loss = AUX_WEIGHT*loss+.7*ce+.3*1.7**2*kd
                if mode=="factor":
                    expert = output[:,22:].gather(1,y[rows,:1].long()).squeeze(1)
                    condition = F.binary_cross_entropy_with_logits(expert,main[rows,0],reduction="none")
                    loss += .50*(condition*valid[rows,0]).sum()/valid[rows,0].sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            lr.step()
            if step%400==0:
                print(rotation,seed,mode,step,len(batches),round(time.perf_counter()-start,1),flush=True)
        path = folder/f"{mode}.pt"
        temp = path.with_suffix(".part")
        torch.save(model.state_dict(),temp)
        temp.replace(path)
        modes[mode] = dict(file=path.name,sha256=checksum(path),sec=time.perf_counter()-start)
        write_json(folder/"fit.json",dict(rotation=rotation,seed=seed,steps=len(batches),rows=len(ids),
                   initialization=source[1], weak_sha256=target_hash, digests=digests,
                   weak_counts=y.sum(0).int().tolist(),modes=modes))
        del enc,model,opt,lr


def open_student(rotation,seed,mode,tok):
    source = checkpoint("official" if rotation=="official" else "projects")
    folder = HERE/rotation/str(seed)
    artifact = json_at(folder/"fit.json")["modes"][mode]
    path = folder/artifact["file"]
    if checksum(path)!=artifact["sha256"]:
        raise ValueError("Rule student weights changed")
    model = RuleEncoder(load_original(tok,"control",source),conditional=mode=="factor").cuda()
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    for layer in model.encoder.local:
        layer.use_shifted = True
    return model.eval()


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*base.ROTATIONS,"official"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--factor",action="store_true")
    a=p.parse_args()
    fit(a.rotation,a.seed,("factor",) if a.factor else MODES)
