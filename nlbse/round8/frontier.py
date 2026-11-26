"""Fit one CodeT5+ with a calibrated four-block exit and an annotation-aware head."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round6"))
from common import (PORTABLE, THRESHOLDS, dataset, dimensions, forward_rows, load_checkpoint,
                    metric, prepare, run_policy, save, sha256, write_json)
from context_policy import Policy
from compare import dense_gflops
from comment_spans import LEXEME
from omitted_comments import TAG
from pipeline import load, PUBLIC
from round2_report import bootstrap_delta
from analyze import changes, group_codes, predict_bits
from test_context import ordered_batches
from tokenizer_speed import DeferredOffsets
from windows import BLOCK, chunks, pack

EARLY = 4
TAGS = {"todo": 0, "fixme": 1, "xxx": 2, "hack": 3}
BASE = Policy("skip_positive", "cover128", gate="skip")
MODES = ("baseline", "four", "six", "adaptive", "wide")


def lexical_features(texts):
    features = np.zeros((len(texts), 5), dtype=np.float32)
    for i, text in enumerate(texts):
        features[i, 4] = np.log1p(len(text))
        if TAG.search(text):
            for word in LEXEME.finditer(text):
                if word[0].startswith(("//", "/*")):
                    for tag in TAG.finditer(text, *word.span()):
                        features[i, TAGS[tag[0].lower()]] = 1
    return features


def pool(stack, states, mask):
    value = stack.dropout(stack.final_layer_norm(states))
    return (value * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


def blocks(stack, x, mask):
    positions = torch.arange(x.shape[1], device=x.device)
    state = stack.dropout(stack.embed_tokens(x))
    additive = (1.0-mask[:, None, None, :].to(state.dtype)) * torch.finfo(state.dtype).min
    bias = None
    for block in stack.block[:EARLY]:
        state, bias = block(state, attention_mask=additive, position_bias=bias, cache_position=positions)[:2]
    return state, bias, positions


def finish(stack, state, bias, positions):
    for block in stack.block[EARLY:]:
        state, bias = block(state, position_bias=bias, cache_position=positions)[:2]
    return state


def use_inputs(texts, tokenizer, wide=False):
    views = ("head", "extend768") if wide else ("head",)
    inputs = prepare(texts, tokenizer, views, metadata=False)
    if wide:
        inputs.views["head"] = [b or a for a,b in zip(inputs.views["head"], inputs.views["extend768"], strict=True)]
    return inputs


@torch.inference_mode()
def collect_features(model, tokenizer, texts, wide=False):
    n = len(texts)
    stack = model.encoder.encoder
    features = np.empty((n, model.encoder.config.d_model), dtype=np.float32)
    native = np.empty((n, 2), dtype=np.float32)
    lengths = np.empty(n, dtype=np.int16)
    with torch.autocast("cuda", dtype=torch.float16):
        for at in range(0, n, BLOCK):
            data = use_inputs(texts[at:at+BLOCK], tokenizer, wide)
            seq = data.views["head"]
            lengths[at:at+len(seq)] = list(map(len, seq))
            for rows in chunks(seq, range(len(seq))):
                x, mask = pack(seq, rows, data.pad, "cuda")
                state,bias,positions = blocks(stack, x, mask)
                at_rows = at+np.asarray(rows)
                features[at_rows] = pool(stack, state, mask).float().cpu().numpy()
                deep = finish(stack, state, bias, positions)
                native[at_rows] = model.head(pool(stack, deep, mask)).float().cpu().numpy()
                if at==0 and rows[0] == min(range(len(seq)), key=lambda i: len(seq[i])):
                    reference = model(x, mask).float()
                    if not torch.equal(reference, torch.from_numpy(native[at_rows]).to("cuda")):
                        raise ValueError("Explicit six-block encoder differs from the original stack")
    return features, native, lengths


@dataclass
class JointHead:
    weight: np.ndarray
    bias: np.ndarray

    @classmethod
    def fit(cls, x, y, strength):
        shift = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), .025)
        normalized = (x-shift)/scale
        coefficients = [LogisticRegression(C=strength, max_iter=700).fit(normalized, y[:, i]) for i in range(2)]
        weight = np.concatenate([m.coef_ for m in coefficients]).astype(np.float32)/scale
        bias = np.asarray([m.intercept_[0] for m in coefficients], dtype=np.float32) - weight@shift
        return cls(weight, bias)

    def __call__(self, x):
        return expit(x@self.weight.T+self.bias)

    def to_dict(self):
        return {"weight": self.weight.tolist(), "bias": self.bias.tolist()}

    @classmethod
    def from_dict(cls, config):
        return JointHead(np.asarray(config["weight"], dtype=np.float32),
                         np.asarray(config["bias"], dtype=np.float32))


def calibrated(y, probability):
    # Coarse fixed grid and the original cutoffs as tie-breaks avoid extreme thresholds on rare MAT labels.
    selected = []
    for i,(label,prior) in enumerate(zip(("vulnerability","MAT"), THRESHOLDS, strict=True)):
        cuts = np.unique(np.r_[np.arange(.25, .871, .025), prior])
        scored = [(metric(y, probability > (value if i==0 else prior, value if i==1 else prior))[label]["f1"],
                   -abs(float(value)-prior), float(value)) for value in cuts]
        selected.append(max(scored)[2])
    return tuple(selected)


def as_common_scale(p, thresholds):
    # A positive log-odds margin is the same decision in either head.
    return expit(logit(np.clip(p, 1e-7, 1-1e-7)) - logit(thresholds))


def should_exit(margin, levels):
    return (np.abs(np.clip(logit(margin.clip(1e-7, 1-1e-7)), -16, 16)) >= levels).all(axis=1)


def tune(output):
    _,_,texts,y,ids,_ = dataset(full=True)
    with np.load(HERE.parent/"round6/router/train_actions.npz") as split:
        if ids != split["ids"].tolist() or not np.array_equal(y, split["y"]):
            raise ValueError("Existing grouped partition is not aligned")
        folds=split["folds"]
    with np.load(HERE.parent/"round5/holdout_scores.npz") as oracle:
        if ids != oracle["ids"].tolist():
            raise ValueError("Can't match the baseline cache to the held-out examples")
        b=oracle["head"].copy()
        has_window=(oracle["size_cover128"]>0)&(b[:,1]<=THRESHOLDS[1])
        b[has_window,1]=np.maximum(b[has_window,1],oracle["cover128"][has_window,1])
    model, tokenizer=load_checkpoint(PORTABLE,"cuda")
    tokenizer=DeferredOffsets(tokenizer)
    short, deep, length=collect_features(model, tokenizer, texts)
    _, wide, wide_length=collect_features(model, tokenizer, texts, wide=True)
    side=lexical_features(texts)
    features=dict(four=np.column_stack((short,side)), six=np.column_stack((deep,side)),
                  wide=np.column_stack((wide,side)))
    split={"fit": folds<=2, "calibration": folds==3, "review": folds==4}
    fit, cal, rev=split.values()
    heads, thresholds, probs={}, {}, {}
    for name,x in features.items():
        heads[name]=JointHead.fit(x[fit],y[fit], .025 if name=="four" else .6)
        raw=heads[name](x)
        thresholds[name]=calibrated(y[cal],raw[cal])
        probs[name]=as_common_scale(raw,thresholds[name])
    pred=probs["four"][cal]
    late=probs["six"][cal]
    first_width=length[cal]
    dim=dimensions(model)
    unit=np.asarray([dense_gflops([(1,int(i))],dim) for i in first_width])
    baseline_f1=metric(y[cal],b[cal]>THRESHOLDS)["mean"]["f1"]
    policies=[]
    for a in (.65, .95, 1.25, 1.55, 1.85, 2.2, 2.8, 3.5, 4.5, 6.):
        for c in (0., .4, .8, 1.2, 1.8, 2.5):
            exit_=should_exit(pred, (a,c))
            p=np.where(exit_[:,None],pred,late)
            score=metric(y[cal],p>.5)["mean"]["f1"]
            saved=unit[exit_].sum()/unit.sum()*(6-EARLY)/6
            # Project relative savings onto the observed full public scale.
            utility=.6*score+.2*547405/3e6*saved
            policies.append(dict(margin=[a,c], calibration_mean_f1=score,
                                 fraction_exit=float(exit_.mean()), estimated_compute_saving=float(saved),
                                 local_utility=float(utility)))
    eligible=[row for row in policies if row["calibration_mean_f1"]>=baseline_f1-.005 and row["fraction_exit"]>=.20]
    selected=max(eligible or policies,key=lambda x:x["local_utility"])
    probabilities=dict(probs)
    fast=should_exit(probs["four"],selected["margin"])
    probabilities["adaptive"]=np.where(fast[:,None],probs["four"],probs["six"])
    review={"baseline":metric(y[rev],b[rev]>THRESHOLDS)}
    review.update({name:metric(y[rev],p[rev]>.5) for name,p in probabilities.items()})
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"holdout.npz", rows=ids, gold=y, folds=folds, lengths=length,
                        wide_lengths=wide_length, baseline=b, adaptive_exits=fast, **probabilities)
    write_json(output/"policy.json",dict(
        checkpoint_sha256=sha256(PORTABLE/"model.safetensors"),
        num_layers_early=EARLY, selected=selected, calibration_baseline_f1=baseline_f1,
        grid=policies, thresholds=thresholds, heads={k:v.to_dict() for k,v in heads.items()}, review=review,
        rows={k:int(v.sum()) for k,v in split.items()}))
    print("selected",selected,"review", {k:round(v["mean"]["f1"],5) for k,v in review.items()},flush=True)


@torch.inference_mode()
def run(model, tokenizer, texts, config, mode, *, prefetch=False):
    start=time.perf_counter()
    stack=model.encoder.encoder
    dims=dimensions(model)
    short_dims=dict(dims,layers=EARLY)
    finish_dims=dict(dims,layers=dims["layers"]-EARLY)
    full_model=JointHead.from_dict(config["heads"][mode if mode!="adaptive" else "six"])
    early_model=JointHead.from_dict(config["heads"]["four"])
    p=np.empty((len(texts),2),dtype=np.float32)
    volume, stopped, forward=0.,0,0.
    def prepare_at(at):
        text=texts[at:at+BLOCK]
        return use_inputs(text,tokenizer,wide=mode=="wide"), lexical_features(text)

    with torch.autocast("cuda", dtype=torch.float16):
        for at,(data,side) in ordered_batches(range(0,len(texts),BLOCK),prepare_at,prefetch):
            seq=data.views["head"]
            for rows in chunks(seq,range(len(seq))):
                x,mask=pack(seq,rows,data.pad,"cuda")
                tick=time.perf_counter()
                if mode in ("six","wide"):
                    logits=model(x,mask).float().cpu().numpy()
                    values=full_model(np.column_stack((logits,side[rows])))
                    p[at+np.asarray(rows)] = as_common_scale(values,config["thresholds"][mode])
                    volume+=dense_gflops([x.shape],dims)
                else:
                    state,bias,positions=blocks(stack,x,mask)
                    pooled=pool(stack,state,mask).float().cpu().numpy()
                    p_early=early_model(np.column_stack((pooled,side[rows])))
                    variance=as_common_scale(p_early,config["thresholds"]["four"])
                    volume+=dense_gflops([x.shape],short_dims)
                    if mode=="adaptive":
                        take=np.flatnonzero(~should_exit(variance,config["selected"]["margin"]))
                        stopped+=len(rows)-len(take)
                        if len(take):
                            ii=torch.as_tensor(take,device="cuda")
                            width=max(len(seq[rows[i]]) for i in take)
                            h=finish(stack,state[ii,:width],bias[ii,:,:width,:width],positions[:width])
                            score=model.head(pool(stack,h,mask[ii,:width])).float().cpu().numpy()
                            values=full_model(np.column_stack((score,side[np.asarray(rows)[take]])))
                            variance[take]=as_common_scale(values,config["thresholds"]["six"])
                            volume+=dense_gflops([(len(take),width)],finish_dims)
                    p[at+np.asarray(rows)]=variance
                forward+=time.perf_counter()-tick
    return p,{"wall_seconds":time.perf_counter()-start,"forward_seconds":forward,
              "analytic_gflops":volume,"early_exits":stopped,"rows":len(texts)}


def public(output, tune_path, repeats):
    config=json.loads((tune_path/"policy.json").read_text())
    model,tokenizer=load_checkpoint(PORTABLE,"cuda")
    _,_,texts,y,ids,cal=dataset(public=True)
    tokenizer=DeferredOffsets(tokenizer)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    run_policy(model,tokenizer,texts[:1024],"cuda",BASE)
    run(model,tokenizer,texts[:1024],config,"adaptive")
    samples,predictions={kind:[] for kind in MODES},{}
    for round_ in range(repeats):
        for mode in MODES if round_%2==0 else MODES[::-1]:
            if mode=="baseline":
                prob,cost=run_policy(model,tokenizer,texts,"cuda",BASE)
                cost["early_exits"]=0
                threshold=THRESHOLDS
            else:
                prob,cost=run(model,tokenizer,texts,config,mode)
                threshold=(.5,.5)
            bits=prob>threshold
            if mode in predictions and not np.array_equal(bits,predictions[mode][0]>threshold):
                raise ValueError(f"Repeated decisions change for {mode}")
            predictions[mode]=(prob,threshold)
            samples[mode].append(cost)
            print(round_+1,mode,json.dumps(cost),flush=True)
    from score_runtime import competition_formula
    results={}
    current=predictions["baseline"][0]>THRESHOLDS
    source=HERE.parent/"round5/runs/public/skip_positive/predictions.csv"
    if not np.array_equal(predict_bits(source,ids,{"thresholds":THRESHOLDS}),current):
        raise ValueError("Benchmark baseline diverged from prior release")
    for mode,(prob,t) in predictions.items():
        cost={key:median(row[key] for row in samples[mode])
              for key in ("wall_seconds","forward_seconds","analytic_gflops","early_exits","rows")}
        save(output/mode,y,ids,cal,prob,cost,t)
        result=json.loads((output/mode/"metrics.json").read_text())
        f1=result["scores"]["mean"]["f1"]
        result["local_formula"]=competition_formula(f1,cost["forward_seconds"],cost["analytic_gflops"])
        result["changes_relative_to_baseline"]=changes(y,current,prob>t)
        result["delta_f1_interval"]=bootstrap_delta(y,current,prob>t,groups)
        results[mode]=result
    write_json(output/"analysis.json",dict(results=results,timing=samples,repetitions=repeats,
                                         apparatus=torch.cuda.get_device_name()))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","public"))
    ap.add_argument("--tune",type=Path,default=HERE/"frozen")
    ap.add_argument("--output",type=Path,default=HERE/"public")
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    if args.stage=="tune":
        tune(args.tune)
    else:
        public(args.output,args.tune,args.repeats)
