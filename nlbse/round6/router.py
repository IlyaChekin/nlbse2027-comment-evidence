"""Train and benchmark a budget-constrained routing head for one CodeT5+."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from common import (HERE, PORTABLE, SEED, SPLIT, THRESHOLDS, dataset, dimensions, forward_rows,
                    load_checkpoint, metric, prepare, save, sha256, write_json)
from compare import dense_gflops
from select_context import shapes
from windows import BLOCK, chunks, pack

OUT = HERE / "router"
CAP = 384
VIEWS = ("head", "cover64", "extend768")
PENALTIES = (0., .01, .03, .10, .30, 1., 3., 10., 30.)
PROJECT_DIM = 32
NAMES = ("stop", "comment", "wide")


def meta(inputs):
    return dict(length=inputs.lengths, missing=inputs.omitted, spans=inputs.groups,
                crossing=inputs.crossing, cue=inputs.cue)


def spans_and_sizes(inputs):
    return np.column_stack((np.zeros(len(inputs.lengths), dtype=np.int32),
                            np.array(list(map(len, inputs.views["cover64"])), dtype=np.int32),
                            np.array(list(map(len, inputs.views["extend768"])), dtype=np.int32)))


@torch.inference_mode()
def first_pass(model, inputs, device="cuda"):
    sequences = inputs.views["head"]
    n = len(sequences)
    head = np.empty((n,2), dtype=np.float32)
    pooled = np.empty((n,model.encoder.config.d_model), dtype=np.float16)
    stats = dict(forward_seconds=0., analytic_gflops=0.)
    dims = dimensions(model)
    start = time.perf_counter()
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for offset in range(0,n,BLOCK):
            for rows in chunks(sequences, range(offset,min(offset+BLOCK,n))):
                x,mask = pack(sequences,rows,inputs.pad,device)
                t = time.perf_counter()
                hidden = model.encoder(input_ids=x,attention_mask=mask).last_hidden_state
                aggregate = (hidden*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1,keepdim=True)
                head[rows] = model.head(aggregate).float().sigmoid().cpu().numpy()
                stats["forward_seconds"] += time.perf_counter()-t
                stats["analytic_gflops"] += dense_gflops([tuple(x.shape)],dims)
                pooled[rows] = aggregate.cpu().numpy()
    stats["elapsed_seconds"] = time.perf_counter()-start
    return head, pooled, stats


def rule(head, sizes):
    return ((sizes[:,1]>0) & (head[:,1]<=THRESHOLDS[1])).astype(np.int8)


def extra_cost(sizes, choices, dims):
    return sum(dense_gflops(shapes(sizes[:, action], choices==action),dims) for action in (1,2))


def prices(sizes, dims):
    widths = np.arange(int(sizes.max())+1)
    costs = np.asarray([dense_gflops([(1,int(width))],dims) for width in widths])
    return costs[sizes]


def available(head, sizes):
    return np.column_stack((np.ones(len(head),dtype=bool), rule(head, sizes).astype(bool), sizes[:,2]>0))


def features(head, pooled, metadata, projector):
    return np.column_stack((
        pooled.astype(np.float32)@projector, head,
        np.abs(head - THRESHOLDS), np.log1p(metadata["length"])/6.,
        np.log1p(metadata["missing"])/5., np.log1p(metadata["spans"]),
        metadata["crossing"], metadata["cue"]
    )).astype(np.float32)


def budgeted(gains, head, sizes, dims, penalty):
    allowed = available(head,sizes)
    cost = prices(sizes,dims)
    scores = np.where(allowed, gains-penalty*cost, -np.inf)
    scores[:,0] = 0.
    choice = scores.argmax(axis=1).astype(np.int8)
    base = rule(head,sizes)
    for start in range(0,len(choice),BLOCK):
        group = slice(start,start+BLOCK)
        selected = choice[group]
        sizes_block = sizes[group]
        limit = extra_cost(sizes_block,base[group],dims)
        if extra_cost(sizes_block,selected,dims) <= limit + 1e-9:
            continue
        proposed = np.flatnonzero(selected)
        quality_per_cost = scores[group][proposed, selected[proposed]] / cost[group][proposed,selected[proposed]]
        for idx in proposed[np.argsort(quality_per_cost,kind="stable")]:
            selected[idx] = 0
            if extra_cost(sizes_block,selected,dims) <= limit + 1e-9:
                break
        if extra_cost(sizes_block,selected,dims) > limit + 1e-9:
            raise AssertionError("Batch exceeded the rule's operation budget")
    return choice


def target_utility(y, p, denom, base_f1):
    base = (p[:,0]>THRESHOLDS).astype(np.float32)
    changes = (p>THRESHOLDS).astype(np.float32)-base[:,None,:]
    marginal = (2*y-base_f1)/denom
    return (changes*marginal[:,None,:]).sum(axis=2)*len(y)/2.


def cached_probabilities(p, action):
    return p[np.arange(len(action)),action]


def learn():
    OUT.mkdir(parents=True,exist_ok=True)
    _,_,texts,y,ids,_ = dataset(full=True)
    model, tokenizer = load_checkpoint(PORTABLE,"cuda")
    start = time.perf_counter()
    inputs = prepare(texts,tokenizer,VIEWS,head_limit=CAP)
    sizes = spans_and_sizes(inputs)
    head, pooled, _ = first_pass(model,inputs)
    p = np.repeat(head[:,None,:],3,axis=1)
    eligible = available(head,sizes)
    for k,view in ((1,"cover64"),(2,"extend768")):
        rows = np.flatnonzero(eligible[:,k])
        second = forward_rows(model,inputs.views[view],rows,inputs.pad,"cuda")
        if k == 1:
            p[rows,1,1] = np.maximum(head[rows,1],second[rows,1])
        else:
            p[rows,2] = second[rows]
    dims = dimensions(model)
    with SPLIT.open(newline="") as handle:
        hashes = {int(row["row"]):row["function_sha256"] for row in csv.DictReader(handle)}
    groups = np.array([hashes[i] for i in ids])
    folds = np.zeros(len(ids),dtype=np.int8)
    joint = 2*y[:,0]+y[:,1]
    for i,(_,subset) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(p,joint,groups)):
        folds[subset] = i
    fit, calibrate, review = folds<=2,folds==3,folds==4
    if any(set(groups[a]) & set(groups[b]) for a,b in ((fit,calibrate),(fit,review),(calibrate,review))):
        raise ValueError("Routing splits overlap by exact text")
    seed = np.random.default_rng(SEED)
    projector = seed.normal(size=(pooled.shape[1],PROJECT_DIM)).astype(np.float32)/np.sqrt(np.float32(pooled.shape[1]))
    x = features(head,pooled,meta(inputs),projector)
    use = fit & eligible[:,2]
    mean, stdev = x[use].mean(axis=0), np.maximum(x[use].std(axis=0),.025)
    x = ((x-mean)/stdev).clip(-8,8)
    prior = metric(y[fit],head[fit]>THRESHOLDS)
    base_f1 = np.asarray([prior[key]["f1"] for key in ("vulnerability","MAT")])
    denominator = np.maximum(1,y[fit].sum(axis=0)+(head[fit]>THRESHOLDS).sum(axis=0))
    # Rescale the linearized macro-F1 changes by the number of routing-fit rows.
    utility = target_utility(y,p,denominator,base_f1)*int(fit.sum())/len(y)
    coefficients = np.zeros((2,x.shape[1]),dtype=np.float32)
    bias = np.zeros(2,dtype=np.float32)
    for action in (1,2):
        mask = fit & eligible[:,action]
        estimator = Ridge(alpha=32.)
        estimator.fit(x[mask],utility[mask,action])
        coefficients[action-1] = estimator.coef_
        bias[action-1] = estimator.intercept_
    pred = np.column_stack((np.zeros(len(x)), x@coefficients.T+bias))
    calibration = []
    for penalty in PENALTIES:
        action = budgeted(pred[calibrate],head[calibrate],sizes[calibrate],dims,penalty)
        yhat = cached_probabilities(p[calibrate],action)
        scores = metric(y[calibrate],yhat>THRESHOLDS)
        cost = extra_cost(sizes[calibrate],action,dims)
        calibration.append(dict(penalty=penalty,scores=scores,extra_gflops=cost))
    chosen = max(calibration,key=lambda r:(r["scores"]["mean"]["f1"],-r["extra_gflops"]))
    config = dict(penalty=chosen["penalty"], dimensions=dims, head_limit=CAP, comparison="rule comment64",
                  checkpoint_sha256=sha256(PORTABLE/"model.safetensors"),
                  partition_rows=dict(fit=int(fit.sum()),budget_cal=int(calibrate.sum()),review=int(review.sum())),
                  class_positives=dict(fit=y[fit].sum(axis=0).tolist(),
                                       calibrated=y[calibrate].sum(axis=0).tolist(),review=y[review].sum(axis=0).tolist()),
                  calibration=calibration,head_features=x.shape[1],seconds=time.perf_counter()-start)
    np.savez_compressed(OUT/"weights.npz",projector=projector,mean=mean,stdev=stdev,
                        coefficients=coefficients,bias=bias)
    np.savez_compressed(OUT/"train_actions.npz",ids=ids,folds=folds,p=p,y=y,sizes=sizes,eligible=eligible,
                        gains_predicted=pred, utility=utility)
    write_json(OUT/"frozen.json",config)
    rids = np.asarray(ids)[review]
    rh, rs, ry = head[review], sizes[review], y[review]
    variants = dict(stop=np.zeros(len(rids),dtype=np.int8),rule=rule(rh,rs),
                    learned=budgeted(pred[review],rh,rs,dims,chosen["penalty"]),
                    oracle_diagnostic=budgeted(utility[review],rh,rs,dims,0.))
    for key,action in variants.items():
        save(OUT/"runs"/"holdout"/key,ry,rids,np.zeros(len(ry),dtype=bool),
             cached_probabilities(p[review],action),
             dict(extra_gflops=extra_cost(rs,action,dims),actions=np.bincount(action,minlength=3).tolist()),THRESHOLDS)
    print("Frozen router",config["partition_rows"],"penalty",chosen["penalty"],flush=True)


def learned_actions(head, pooled, inputs, sizes, weights, config):
    x = features(head,pooled,meta(inputs),weights["projector"])
    x = ((x-weights["mean"])/weights["stdev"]).clip(-8,8)
    gains = np.column_stack((np.zeros(len(x)),x@weights["coefficients"].T+weights["bias"]))
    return budgeted(gains,head,sizes,config["dimensions"],config["penalty"])


def run_windows(model, inputs, head, action, device="cuda"):
    output = head.copy()
    cost = dict(forward_seconds=0.,analytic_gflops=0.)
    for step,view in ((1,"cover64"), (2,"extend768")):
        rows = np.flatnonzero(action==step)
        if not len(rows):
            continue
        tail = forward_rows(model,inputs.views[view],rows,inputs.pad,device,cost)
        if step == 1:
            output[rows,1] = np.maximum(output[rows,1],tail[rows,1])
        else:
            output[rows] = tail[rows]
    return output,cost


def public():
    _,_,texts,y,ids,cal = dataset(public=True)
    config = json.loads((OUT/"frozen.json").read_text())
    weights = dict(np.load(OUT/"weights.npz"))
    model,tokenizer = load_checkpoint(PORTABLE,"cuda")
    start = time.perf_counter()
    inputs = prepare(texts,tokenizer,VIEWS,head_limit=CAP)
    pre_seconds = time.perf_counter()-start
    sizes = spans_and_sizes(inputs)
    results = {}
    for rep in range(3):
        head, pooled, first = first_pass(model,inputs)
        start = time.perf_counter()
        choices = {"stop":np.zeros(len(head),dtype=np.int8),"rule":rule(head,sizes)}
        basic_route = time.perf_counter()-start
        start = time.perf_counter()
        choices["learned"] = learned_actions(head,pooled,inputs,sizes,weights,config)
        learned_route = time.perf_counter()-start
        order = ("rule","learned") if rep % 2 == 0 else ("learned","rule")
        for name in ("stop",*order):
            start = time.perf_counter()
            p, extra = run_windows(model,inputs,head,choices[name])
            duration = time.perf_counter()-start
            route = learned_route if name == "learned" else basic_route
            route_ops = 2.*len(head)*(weights["projector"].size+weights["coefficients"].size)/1e9 if name=="learned" else 0.
            stat = dict(forward_seconds=first["forward_seconds"]+extra["forward_seconds"],
                        inference_seconds=pre_seconds+first["elapsed_seconds"]+route+duration,
                        route_seconds=route, analytic_gflops=first["analytic_gflops"]+extra["analytic_gflops"]+route_ops,
                        extra_gflops=extra["analytic_gflops"], head_gflops=first["analytic_gflops"],
                        routing_head_gflops=route_ops, actions=np.bincount(choices[name],minlength=3).tolist())
            if name in results and (not np.array_equal(results[name][0]>THRESHOLDS,p>THRESHOLDS)
                                    or not np.array_equal(choices[name],results[name][2])):
                raise ValueError("Router predictions or actions changed across repetitions")
            previous = results[name][1] if name in results else []
            results[name] = (p, [*previous,stat], choices[name])
            print("router timing",rep+1,name,stat,flush=True)
    for name,(p,stats,action) in results.items():
        cost = dict(stats[0],rows=len(y),shared_tokenization_seconds=pre_seconds,apparatus=torch.cuda.get_device_name())
        for key in ("forward_seconds","inference_seconds","route_seconds"):
            cost[key+"_repeats"] = [r[key] for r in stats]
            cost[key] = median(cost[key+"_repeats"])
        save(OUT/"runs"/"public"/name,y,ids,cal,p,cost,THRESHOLDS)
        np.save(OUT/"runs"/"public"/name/"actions.npy", action)
    if results["learned"][1][0]["extra_gflops"] > results["rule"][1][0]["extra_gflops"]+1e-8:
        raise ValueError("Measured routed passes surpassed matched rule budget")


def predict(source,output,device):
    frozen = json.loads((OUT/"frozen.json").read_text())
    weights = dict(np.load(OUT/"weights.npz"))
    model, tokenizer = load_checkpoint(PORTABLE,device)
    with source.open(newline="",encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function column")
        texts = [row["Function"] for row in reader]
    inputs = prepare(texts,tokenizer,VIEWS,head_limit=CAP)
    head,pooled,_ = first_pass(model,inputs,device)
    sizes = spans_and_sizes(inputs)
    action = learned_actions(head,pooled,inputs,sizes,weights,frozen)
    p,_ = run_windows(model,inputs,head,action,device)
    with output.open("w",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>THRESHOLDS).astype(np.int8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage",required=True)
    sub.add_parser("learn")
    sub.add_parser("public")
    infer = sub.add_parser("predict")
    infer.add_argument("--input",type=Path,required=True)
    infer.add_argument("--output",type=Path,required=True)
    infer.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args = parser.parse_args()
    if args.stage == "learn":
        learn()
    elif args.stage == "public":
        public()
    else:
        predict(args.input,args.output,args.device)
