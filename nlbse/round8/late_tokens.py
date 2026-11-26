"""Keep the six frozen blocks and update only selected token states in the final two."""
import argparse
from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from torch.nn import functional as F

from frontier import (EARLY, HERE, PORTABLE, THRESHOLDS, JointHead, as_common_scale, blocks,
                      calibrated, dataset, dimensions, finish, lexical_features, load_checkpoint,
                      metric, ordered_batches, pool, save, sha256, use_inputs, write_json)
from analyze import changes, group_codes, predict_bits
from compare import dense_gflops
from pipeline import load, PUBLIC
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from tokenizer_speed import DeferredOffsets
from windows import BLOCK, chunks, pack


@dataclass(frozen=True)
class Policy:
    attention: str
    selector: str
    keep: float

    @property
    def tag(self):
        return f"{self.attention}_{self.selector}_{int(100*self.keep)}"


FULL = Policy("full", "even", 1.)
CANDIDATES = (
    Policy("query", "saliency", .50), Policy("query", "saliency", .65),
    Policy("query", "saliency", .80), Policy("query", "saliency", .9),
    Policy("query", "even", .65), Policy("query", "even", .80),
    Policy("prune", "saliency", .65), Policy("prune", "saliency", .80),
    Policy("prune", "even", .65), Policy("prune", "even", .80),
)


def selection(policy, state, mask, saliency):
    length = mask.sum(dim=1, keepdim=True)
    kept = torch.minimum(length, torch.ceil(length * policy.keep).to(length.dtype).clamp_min(8))
    width = min(state.shape[1], max(8, ceil(state.shape[1] * policy.keep)))
    ranks = torch.arange(width, device=state.device)
    valid = ranks < kept
    if policy.selector == "even":
        indices = (ranks * (length-1)) // (kept-1).clamp_min(1)
    else:
        coordinates = torch.arange(state.shape[1], device=state.device)
        endpoints = (coordinates==0) | (coordinates==length-1)
        relevance = saliency.masked_fill(endpoints, torch.inf).masked_fill(mask==0, -torch.inf)
        indices = relevance.topk(width, dim=1).indices
        # Sort retained positions together while moving the discarded ranks to the end.
        indices = indices.masked_fill(~valid, state.shape[1]).sort(dim=1).values
    # Values beyond each real sequence point to distinct padded positions.
    return torch.where(valid, indices, length + ranks - kept), valid


def take_tokens(value, indices):
    return value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))


def attention_tensor(linear, value, heads, depth):
    return linear(value).reshape(value.shape[0], value.shape[1], heads, depth).transpose(1,2)


def update_queries(stack, state, full_bias, indices):
    b, width = indices.shape
    head_ids = torch.arange(full_bias.shape[1], device=state.device)
    batch_ids = torch.arange(b, device=state.device)
    bias = full_bias[batch_ids[:,None,None], head_ids[None,:,None], indices[:,None,:], :]
    for block in stack.block[EARLY:]:
        self_attention = block.layer[0]
        attn = self_attention.SelfAttention
        normalized = self_attention.layer_norm(state)
        query = attention_tensor(attn.q, take_tokens(normalized,indices),attn.n_heads,attn.key_value_proj_dim)
        key = attention_tensor(attn.k, normalized,attn.n_heads,attn.key_value_proj_dim)
        value = attention_tensor(attn.v, normalized,attn.n_heads,attn.key_value_proj_dim)
        scores = torch.matmul(query,key.transpose(2,3))
        scores += bias
        probabilities = F.softmax(scores.float(),dim=-1).type_as(scores)
        update = torch.matmul(probabilities,value).transpose(1,2).contiguous().view(b,width,attn.inner_dim)
        initially = take_tokens(state,indices)
        updated = initially + self_attention.dropout(attn.o(update))
        updated = block.layer[-1](updated)
        state = state.scatter(1,indices[...,None].expand(-1,-1,state.shape[-1]),updated)
    return state


def correction_pool(stack, states, mask, indices, valid, refined):
    before = stack.final_layer_norm(states)
    after = stack.final_layer_norm(refined)
    raw = (before * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1,keepdim=True)
    delta = after - take_tokens(before,indices)
    return raw + (delta * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1,keepdim=True)


def reduced_forward(model, state, bias, positions, mask, policy, saliency, dims):
    stack = model.encoder.encoder
    b,l = mask.shape
    suffix = dict(dims, layers=dims["layers"]-EARLY)
    if policy.attention == "full":
        complete = finish(stack,state,bias,positions)
        return model.head(pool(stack,complete,mask)), dense_gflops([(b,l)],suffix)
    indices, valid = selection(policy,state,mask,saliency)
    reduced = indices.shape[1]
    if policy.attention == "query":
        result = update_queries(stack,state,bias,indices)
        d,f = suffix["hidden_size"],suffix["intermediate_size"]
        cost = suffix["layers"] * b * (4*(l+reduced)*d*d + 4*reduced*d*f + 4*l*reduced*d) / 1e9
        return model.head(pool(stack,result,mask)), cost
    b_id = torch.arange(b,device=state.device)
    h_id = torch.arange(bias.shape[1],device=state.device)
    compact_bias = bias[b_id[:,None,None,None], h_id[None,:,None,None],
                        indices[:,None,:,None], indices[:,None,None,:]]
    result = finish(stack,take_tokens(state,indices),compact_bias,positions[:reduced])
    if policy.attention == "drop":
        mixed = state.scatter(1,indices[...,None].expand(-1,-1,state.shape[-1]),result)
        pooled = pool(stack,mixed,mask)
    else:
        pooled = correction_pool(stack,state,mask,indices,valid,result)
    return model.head(pooled),dense_gflops([(b,reduced)],suffix)


@torch.inference_mode()
def logits_for(model, tokenizer, texts, policies, *, wide=False):
    stack = model.encoder.encoder
    need_saliency = any(p.selector=="saliency" for p in policies)
    cfg = dimensions(model)
    prefix = dict(cfg,layers=EARLY)
    output = {p.tag: np.empty((len(texts),2),dtype=np.float32) for p in policies}
    cost = {p.tag:0. for p in policies}
    features = np.empty((len(texts),5),dtype=np.float32)
    tested = False
    forward = 0.
    start = time.perf_counter()

    def prepare_at(first):
        local = texts[first:first+BLOCK]
        return use_inputs(local,tokenizer,wide=wide), lexical_features(local)

    with torch.autocast("cuda",dtype=torch.float16):
        for first,(data,side) in ordered_batches(range(0,len(texts),BLOCK),prepare_at,True):
            seq = data.views["head"]
            features[first:first+len(seq)] = side
            for group in chunks(seq, range(len(seq))):
                x,mask = pack(seq,group,data.pad,"cuda")
                tick = time.perf_counter()
                states,bias,positions = blocks(stack,x,mask)
                # Use the existing classification directions at block four to score contextual states.
                salience = model.head(stack.final_layer_norm(states)).abs().amax(-1) if need_saliency else None
                prefix_cost = dense_gflops([x.shape],prefix)
                for policy in policies:
                    p,tail = reduced_forward(model,states,bias,positions,mask,policy,salience,cfg)
                    output[policy.tag][first+np.asarray(group)] = p.float().cpu().numpy()
                    cost[policy.tag] += prefix_cost + tail
                forward += time.perf_counter()-tick
                if not tested and len(policies)>1:
                    all_tokens = torch.arange(x.shape[1],device=x.device).expand(x.shape[0],-1)
                    complete = finish(stack,states,bias,positions)
                    manual = update_queries(stack,states,bias,all_tokens)
                    expected = model(x,mask)
                    if not torch.equal(model.head(pool(stack,complete,mask)),expected):
                        raise ValueError("Full split graph differs from normal T5")
                    if not torch.equal(manual,complete):
                        raise ValueError("Updating all queries should replay the standard blocks")
                    tested = True
    return output, features, dict(wall_seconds=time.perf_counter()-start,forward_seconds=forward,analytic_gflops=cost)


def tune(out):
    _,_,texts,y,ids,_=dataset(full=True)
    with np.load(HERE/"frozen/holdout.npz") as saved:
        if ids != saved["rows"].tolist() or not np.array_equal(y,saved["gold"]):
            raise ValueError("Late-token tuning received different holdout examples")
        folds=saved["folds"]
    enc, token=load_checkpoint(PORTABLE,"cuda")
    token=DeferredOffsets(token)
    candidates=(FULL,*CANDIDATES)
    raw,features,cost=logits_for(enc,token,texts,candidates,wide=True)
    fit,cal,review=folds<=2,folds==3,folds==4
    out.mkdir(parents=True,exist_ok=True)
    configurations,probabilities={},{}
    for policy in candidates:
        x=np.column_stack((raw[policy.tag],features))
        head=JointHead.fit(x[fit],y[fit],.6)
        scores=head(x)
        cut=calibrated(y[cal],scores[cal])
        probabilities[policy.tag]=as_common_scale(scores,cut)
        configurations[policy.tag]=dict(spec=asdict(policy),head=head.to_dict(),thresholds=cut,
                                        calibration=metric(y[cal],scores[cal]>cut),
                                        review=metric(y[review],scores[review]>cut),
                                        estimated_gflops=cost["analytic_gflops"][policy.tag])
    untagged=np.column_stack((raw[FULL.tag],features[:,4:]))
    no_tag_head=JointHead.fit(untagged[fit],y[fit],.6)
    no_tag_scores=no_tag_head(untagged)
    no_tag_cuts=calibrated(y[cal],no_tag_scores[cal])
    tag_ablation=dict(head=no_tag_head.to_dict(),thresholds=no_tag_cuts,
                      review=metric(y[review],no_tag_scores[review]>no_tag_cuts))
    reference=configurations[FULL.tag]
    baseline_f1=reference["calibration"]["mean"]["f1"]
    def rank(key):
        row=configurations[key]
        f1=row["calibration"]["mean"]["f1"]
        saving=1-row["estimated_gflops"]/reference["estimated_gflops"]
        return .6*f1+.2*564489/3e6*saving
    accepted=[p for p in CANDIDATES if configurations[p.tag]["calibration"]["mean"]["f1"] >= baseline_f1-.003]
    selected=max(accepted or CANDIDATES,key=lambda p:rank(p.tag))
    control=Policy("prune" if selected.attention=="query" else "query", selected.selector,selected.keep)
    bench=[FULL.tag,selected.tag]
    if control.tag in configurations:
        bench.append(control.tag)
    bench.extend(p.tag for p in CANDIDATES if p.keep==selected.keep and p.tag not in bench)
    write_json(out/"policy.json",dict(encoder_sha256=sha256(PORTABLE/"model.safetensors"),
                                         configurations=configurations, selected=selected.tag,
                                         within_budget=bool(accepted), benchmark=bench, reference=FULL.tag, input_cap=768,
                                         tag_ablation=tag_ablation))
    np.savez_compressed(out/"holdout.npz", rows=ids,y=y,folds=folds,**probabilities)
    print("Frozen late-token selection",selected.tag, "review",
          {k:round(v["review"]["mean"]["f1"],5) for k,v in configurations.items()},flush=True)


def predict_variant(model,tokenizer,texts,config,variant):
    record=config["configurations"][variant]
    policy=Policy(**record["spec"])
    started=time.perf_counter()
    raw, features, cost=logits_for(model,tokenizer,texts,[policy],wide=config["input_cap"]==768)
    p=JointHead.from_dict(record["head"])(np.column_stack((raw[policy.tag],features)))
    decision=as_common_scale(p,record["thresholds"])
    cost["analytic_gflops"]=cost["analytic_gflops"][variant]
    cost["wall_seconds"]=time.perf_counter()-started
    return decision,cost


def public(out,tune_path,repeats, compare=None):
    saved=json.loads((tune_path/"policy.json").read_text())
    enc, token=load_checkpoint(PORTABLE,"cuda")
    token=DeferredOffsets(token)
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    origin=predict_bits(HERE.parent/"round5/runs/public/skip_positive/predictions.csv",ids,
                       {"thresholds":THRESHOLDS})
    variants=compare or saved["benchmark"]
    if FULL.tag not in variants or len(set(variants)) != len(variants):
        raise ValueError("Compare distinct variants including the full reference")
    for name in variants:
        predict_variant(enc,token,texts[:1024],saved,name)
    prediction, measurements = {}, {v:[] for v in variants}
    for i in range(repeats):
        for variant in variants if i%2==0 else variants[::-1]:
            p,cost=predict_variant(enc,token,texts,saved,variant)
            if variant in prediction and not np.array_equal(prediction[variant]>.5,p>.5):
                raise ValueError(f"Nondeterministic {variant} results")
            prediction[variant]=p
            measurements[variant].append(cost)
            print(i+1,variant,cost,flush=True)
    full=prediction[FULL.tag]>.5
    results={}
    for variant,p in prediction.items():
        duration={key:median(r[key] for r in measurements[variant]) for key in measurements[variant][0]}
        save(out/variant,y,ids,cal,p,duration,(.5,.5))
        row=json.loads((out/variant/"metrics.json").read_text())
        row["local_formula"]=competition_formula(row["scores"]["mean"]["f1"],
                                                  duration["forward_seconds"],duration["analytic_gflops"])
        row["interval_vs_deployed"]=bootstrap_delta(y,origin,p>.5,groups)
        row["interval_vs_full"]=bootstrap_delta(y,full,p>.5,groups)
        row["changes_vs_full"]=changes(y,full,p>.5)
        results[variant]=row
    raw, side, _=logits_for(enc,token,texts,[FULL],wide=saved["input_cap"]==768)
    no_tag=saved["tag_ablation"]
    diagnostic=JointHead.from_dict(no_tag["head"])(np.column_stack((raw[FULL.tag],side[:,4:])))
    diagnostic=as_common_scale(diagnostic,no_tag["thresholds"])
    save(out/"full_without_tags",y,ids,cal,diagnostic,{"scope":"ablation, untimed"},(.5,.5))
    tag_ablation=dict(scores=metric(y,diagnostic>.5),
                      improvement_with_tags=bootstrap_delta(y,diagnostic>.5,full,groups))
    write_json(out/"analysis.json",dict(selected=saved["selected"],baseline=FULL.tag,
                                         results=results,trials=measurements,repetitions=repeats,
                                         tag_ablation=tag_ablation,device=torch.cuda.get_device_name()))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","public"))
    ap.add_argument("--tune",type=Path,default=HERE/"tokens_frozen")
    ap.add_argument("--output",type=Path,default=HERE/"tokens_public")
    ap.add_argument("--repeats",type=int,default=2)
    ap.add_argument("--compare",nargs="+")
    args=ap.parse_args()
    if args.stage=="tune":
        tune(args.tune)
    else:
        public(args.output,args.tune,args.repeats,args.compare)
