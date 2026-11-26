"""Train an anchored linear residual in the frozen rule-negative head."""
import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from train import HERE, ROOT, SEED, batch, batch_plan, checksum, cohorts, fit_ids, json_at, open_student
from train import small, write_json
import factor as f
from rule_model import pooling

MODES = ("head_bce", "head_focal")


@torch.inference_mode()
def extract(model, inputs, tok):
    features = torch.empty((len(inputs),model.width*2),dtype=torch.float16,device="cuda")
    norm = model.rule_head[0]
    for rows in batch_plan(list(range(len(inputs))),inputs,small.BATCH):
        x,mask = batch(rows,inputs,tok.pad_token_id)
        with torch.autocast("cuda",dtype=torch.float16):
            features[rows] = norm(pooling(model.encoder.token_states(x,mask),mask)).to(features.dtype)
    return features


def fit(rotation):
    texts, full_y, _, _, _ = f.inputs()
    eligible, gold, cohort_hash = cohorts()
    ids = fit_ids(rotation)
    ids = ids[eligible[ids]==0]
    y = torch.as_tensor(full_y[ids,0],device="cuda",dtype=torch.float32)
    if not np.array_equal(full_y[ids,0],gold[ids]):
        raise ValueError("Frozen residual target does not align")
    source = [texts[i] for i in ids]
    tok = small.tokenizer()
    tracked = time.perf_counter()
    entries = {}
    origin = ROOT/"round20"/rotation/str(SEED)/"fit.json"
    model = open_student(rotation,SEED,"factor",tok)
    seq = f.tokenize(source,tok)
    latent = extract(model,seq,tok)
    linear = model.rule_head[1]
    prior = (latent.float() @ linear.weight[20].detach() + linear.bias[20].detach()).float()
    # A residual is trained on the raw inference view. Final coefficients are added to the existing head.
    print(rotation,"latent",len(ids),float(y.sum()),time.perf_counter()-tracked,flush=True)
    out = HERE/rotation/"frozen_head"
    out.mkdir(parents=True,exist_ok=True)
    pw = ((len(y)-y.sum())/y.sum()).sqrt().clamp(1,6)
    for mode in MODES:
        torch.manual_seed(SEED)
        w = torch.nn.Parameter(torch.zeros(latent.shape[1],device="cuda"))
        b = torch.nn.Parameter(torch.zeros((),device="cuda"))
        opt = torch.optim.AdamW((w,b),lr=.007,weight_decay=.01)
        for epoch in range(8):
            for i in torch.randperm(len(latent),device="cuda").split(1024):
                p = prior[i]+latent[i].float()@w+b
                risk = F.binary_cross_entropy_with_logits(p,y[i],pos_weight=pw,reduction="none")
                if mode=="head_focal":
                    q = p.detach().sigmoid()
                    risk *= 1+4*torch.where(y[i].bool(),1-q,q).square()
                loss = risk.mean()+.65*w.square().sum()+.15*b.square()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        output = out/f"{mode}.npz"
        np.savez_compressed(output,weight=w.detach().cpu().numpy(),bias=b.detach().cpu().numpy())
        entries[mode]=dict(file=output.name,sha256=checksum(output),mean_abs_training_shift=float(
                        (latent.float()@w.detach()+b.detach()).abs().mean()))
        print(rotation,mode,entries[mode],flush=True)
    write_json(out/"fit.json",dict(variants=entries,rows=len(ids),checkpoint_sha256=json_at(origin)["modes"]["factor"]["sha256"],
                                  cohort_sha256=cohort_hash,script_sha256=checksum(Path(__file__))))


@torch.inference_mode()
def model_for(rotation, mode, tok):
    original = open_student(rotation,SEED,"factor",tok)
    folder = HERE/rotation/"frozen_head"
    info = json_at(folder/"fit.json")["variants"][mode]
    file = folder/info["file"]
    if checksum(file)!=info["sha256"]:
        raise ValueError("Frozen head changes after training")
    with np.load(file) as result:
        original.rule_head[1].weight[20].add_(torch.from_numpy(result["weight"]).cuda(),alpha=.5)
        original.rule_head[1].bias[20].add_(torch.from_numpy(result["bias"]).cuda(),alpha=.5)
    return original


def collect(rotation):
    from types import SimpleNamespace
    from project_eval import evaluate
    text,_,_,_,_ = f.inputs()
    tok = small.tokenizer()
    folder = HERE/rotation
    route,_ = f.pinned()
    data = {}
    for role in ("dev","test"):
        with np.load(folder/f"{role}.npz") as z:
            data[role] = dict(z)
    for mode in MODES:
        pipe = SimpleNamespace(encoder=model_for(rotation,mode,tok),tok=tok,mode="control")
        for role, frame in data.items():
            values = [text[i] for i in frame["ids"]]
            p,a,cost = f.infer_student(pipe,values,route,columns=22,return_all=True)
            result = np.column_stack((p[:,0],a[:,0],a[:,-2:]))
            if not np.array_equal(result[:,[0,1,3]],frame["reference"][:,[0,1,3]]):
                raise ValueError("A one-head residual changed unrelated outputs")
            frame[mode] = result
            frame[mode+"_cost"] = np.array([cost["forward_seconds"],cost["analytic_gflops"]])
    target = folder/"frozen_head"
    for role,frame in data.items():
        np.savez_compressed(target/f"{role}.npz",**frame)
    evaluate(rotation,MODES,folder=target)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*f.base.ROTATIONS,"official"))
    p.add_argument("stage",choices=("fit","collect"))
    a = p.parse_args()
    {"fit":fit,"collect":collect}[a.stage](a.rotation)
