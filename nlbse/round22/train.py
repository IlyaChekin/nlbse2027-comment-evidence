"""Continue a conditional encoder with error weighting confined to rule-negative rows."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "round20"))
import factor
from student import fit_ids, rules, open_student
from transfer import batch, batch_plan, checksum, json_at, small, write_json

SEED = 42
MODES = ("bce", "focal", "tail")


def cohorts():
    target = ROOT / "round21/cohorts.npz"
    receipt = json_at(target.with_suffix(".json"))
    if checksum(target) != receipt["sha256"]:
        raise ValueError("The verified rule cohorts changed")
    with np.load(target) as saved:
        if not np.array_equal(saved["rows"], np.arange(receipt["rows"])):
            raise ValueError("Unexpected row ordering in rule cohorts")
        return saved["weak"], saved["y"], receipt["sha256"]


def negative_loss(logits, gold, eligible, positive_weight, mode):
    loss = F.binary_cross_entropy_with_logits(logits, gold, pos_weight=positive_weight, reduction="none")
    weights = eligible.float()
    if mode != "bce":
        # Detached weights change the examples' emphasis without adding a gradient through their rank.
        p = logits.detach().float().sigmoid()
        if mode == "focal":
            hardness = torch.where(gold.bool(), 1-p, p).square()
        else:
            hardness = torch.where(gold.bool(), (1-p).square(), p.pow(4))
        weights = weights * (1+4*hardness)
    return (loss*weights).sum()/weights.sum().clamp_min(1.)


def run(rotation, modes):
    labels, teacher, _, digests, tok, plain, normalized, lexical = factor.base.prepared_training(rotation)
    ids = fit_ids(rotation)
    weak, check_gold, cohort_hash = cohorts()
    if not np.array_equal(labels[:, 0], check_gold[ids]):
        raise ValueError("Rule cohort labels do not match prepared training rows")
    weak = torch.as_tensor(weak[ids], device="cuda")
    targets, available, rule_hash = rules()
    rule_y = torch.as_tensor(targets[ids], device="cuda", dtype=torch.float32)
    rule_valid = torch.as_tensor(available[ids], device="cuda") & (rule_y.sum(0)>=32)
    rule_w = ((rule_valid.sum(0)-rule_y.sum(0))/rule_y.sum(0).clamp_min(1)).clamp(1,25).sqrt()
    gold = torch.as_tensor(labels, device="cuda", dtype=torch.float32)
    soft = (torch.as_tensor(teacher, device="cuda").clamp(1e-5,1-1e-5).logit()/1.7).sigmoid()
    main_w = ((len(gold)-gold.sum(0))/gold.sum(0).clamp_min(1)).clamp(1,5)
    n_neg = (weak==0)&(gold[:,0]==0)
    n_pos = (weak==0)&(gold[:,0]==1)
    positive_weight = (n_neg.sum()/n_pos.sum().clamp_min(1)).sqrt().clamp(1,6)
    side = torch.as_tensor(lexical, device="cuda")
    rng = random.Random(SEED)
    seq = [canonical if rng.random()<.45 else raw for raw,canonical in zip(plain,normalized,strict=True)]
    order = list(range(len(seq)))
    rng.shuffle(order)
    batches = list(batch_plan(order,seq,small.BATCH))
    rng.shuffle(batches)
    out = HERE / rotation
    out.mkdir(parents=True,exist_ok=True)
    version = json_at(ROOT/"round20"/rotation/str(SEED)/"fit.json")["modes"]["factor"]["sha256"]
    config = dict(seed=SEED,rows=len(ids),steps=len(batches),cohorts_sha256=cohort_hash,
                  source_sha256=version,source_rows=digests,rules_sha256=rule_hash,
                  code_sha256=checksum(Path(__file__)),positive_weight=float(positive_weight),
                  rule_negative_positive=int(n_pos.sum()),rule_negative_negative=int(n_neg.sum()),
                  rates=[.00005,.00012,.00040],weight_decay=.025)
    entries = {}
    receipt = out/"training.json"
    if receipt.exists():
        old = json_at(receipt)
        if old["config"] != config:
            raise ValueError("Existing results came from a different training recipe")
        entries = old["variants"]
    for mode in modes:
        if mode in entries and checksum(out/entries[mode]["file"])==entries[mode]["sha256"]:
            continue
        torch.manual_seed(SEED)
        model = open_student(rotation,SEED,"factor",tok).train()
        # Use the previous training kernel and retain the shifted kernel for inference.
        for block in model.encoder.local:
            block.use_shifted=False
        rates = config["rates"]
        params = [dict(params=model.encoder.embed.parameters(),lr=rates[0]),
                  dict(params=[p for n,p in model.encoder.named_parameters() if not n.startswith("embed.")],lr=rates[1]),
                  dict(params=model.rule_head.parameters(),lr=rates[2])]
        optim = torch.optim.AdamW(params,weight_decay=config["weight_decay"],fused=True)
        schedule = get_linear_schedule_with_warmup(optim,int(.04*len(batches)),len(batches))
        t = time.perf_counter()
        for step, rows in enumerate(batches, 1):
            tokens,mask = batch(rows,seq,tok.pad_token_id)
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                out_logits = model(tokens,mask,side[rows])
                current = gold[rows]
                main = F.binary_cross_entropy_with_logits(out_logits[:,:2],current,pos_weight=main_w)
                kd = F.binary_cross_entropy_with_logits(out_logits[:,:2]/1.7,soft[rows])
                aux = F.binary_cross_entropy_with_logits(out_logits[:,2:22],rule_y[rows],pos_weight=rule_w,reduction="none")
                aux = (aux*rule_valid[rows]).sum()/rule_valid[rows].sum().clamp_min(1)
                has_rule = weak[rows]==1
                special = F.binary_cross_entropy_with_logits(out_logits[:,23],current[:,0],reduction="none")
                positive_loss = (special*has_rule).sum()/has_rule.sum().clamp_min(1)
                negative = negative_loss(out_logits[:,22],current[:,0],weak[rows]==0,positive_weight,mode)
                loss = .7*main+.3*1.7**2*kd+.25*aux+.20*positive_loss+.45*negative
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
            schedule.step()
            if step%400==0:
                print(rotation,mode,step,len(batches),round(time.perf_counter()-t,1),flush=True)
        path = out/f"{mode}.pt"
        staged = path.with_suffix(".part")
        torch.save(model.state_dict(),staged)
        staged.replace(path)
        entries[mode]=dict(file=path.name,sha256=checksum(path),seconds=time.perf_counter()-t)
        write_json(receipt,dict(config=config,variants=entries))
        print(rotation,mode,entries[mode],flush=True)


def pipeline(rotation, mode, tok):
    model = open_student(rotation,SEED,"factor",tok)
    if mode!="reference":
        folder = HERE/rotation
        record = json_at(folder/"training.json")["variants"][mode]
        path = folder/record["file"]
        if checksum(path)!=record["sha256"]:
            raise ValueError("Continuation checkpoint changed")
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return SimpleNamespace(encoder=model.eval(),tok=tok,mode="control")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rotation",choices=(*factor.base.ROTATIONS,"official"))
    ap.add_argument("--modes",nargs="+",choices=MODES,default=MODES)
    args=ap.parse_args()
    run(args.rotation,args.modes)
