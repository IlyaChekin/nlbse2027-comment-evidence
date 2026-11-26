"""Compare pooled, project-balanced and worst-project continuation on fixed rotations."""
import argparse
import csv
from functools import lru_cache
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
sys.path.insert(0, str(ROOT/"round17"))
from view_train import batch, batch_plan, checkpoint, checksum, fit_data, inputs, json_at, lexical_features
from view_train import load_original, parallel, select_indices, small, tokenize, write_json
from view_eval import benchmark, pinned, selected_views
from experiment import comparison, infer, strict_budget
from evaluate import counts, exact_budget, grouped_delta
from local_public import heldout, run
from pipeline import TRAIN, metric
from score_runtime import competition_formula
from students import group_codes


ARMS = ("pooled", "balanced", "robust")
ADVERSARY = "invariant"
ROTATIONS = ("test_a", "test_b")
SEEDS = (42, 101)
RESULTS = HERE
TEACHER = ROOT/"round13/backbones/unix"
GAIN = .70
TEMPERATURE = .15
EMA = .07
WEIGHT_CAP = 8.
DOMAIN_GRADIENT = .24


def verified_matrix(path, role_ids, label):
    digest = checksum(path)
    if digest != json_at(path.with_suffix(".json"))["sha256"]:
        raise ValueError(f"Teacher file changed: {path}")
    with np.load(path) as f:
        if not np.array_equal(f["ids"], role_ids) or not np.array_equal(f["y"], label[role_ids]):
            raise ValueError(f"Teacher rows changed: {path}")
        return f["probabilities"], digest


def partition(rotation):
    texts, label, project, files, old = inputs()
    if rotation == "official":
        x, y, q, digest = fit_data("official")
        ids, _ = select_indices(False)
        ids = np.asarray(ids)
        if not np.array_equal(y, label[ids]) or any(a != texts[i] for a, i in zip(x, ids, strict=True)):
            raise ValueError("Official fit rows do not match the training data")
        return x, y, q, project[ids], {"teacher": digest, "train": checksum(TRAIN)}
    receipt = json_at(HERE/"partition.json")
    entry = receipt["rotations"][rotation]
    path = HERE/entry["partition_file"]
    if checksum(path) != entry["partition_sha256"] or checksum(TRAIN) != receipt["train_sha256"]:
        raise ValueError("Rotation or data changed")
    with np.load(path) as f:
        fit = f["fit"]
    q = np.full((len(fit), 2), np.nan, dtype=np.float32)
    digests = {"rotation": entry["partition_sha256"], "train": receipt["train_sha256"]}
    for name in ("fit", "dev", "test"):
        overlap, left, right = np.intersect1d(fit, old[name], return_indices=True)
        if not len(overlap):
            continue
        source = TEACHER/("teacher.npz" if name == "fit" else f"{name}.npz")
        predictions, digest = verified_matrix(source, old[name], label)
        q[left] = predictions[right]
        digests[name+"_teacher"] = digest
    if not np.isfinite(q).all() or set(project[fit]) & set(entry["heldout_projects"]):
        raise ValueError("Missing teacher predictions or a project overlap")
    return [texts[i] for i in fit], label[fit], q, project[fit], digests


def row_weights(prior, fraction, running, codes, mode):
    # Mix pooled empirical risk with a tempered worst-project objective.
    emphasis = prior
    if mode == "robust":
        emphasis = torch.softmax(prior.log()+(running-running.mean())/TEMPERATURE, dim=0)
    weight = ((1.-GAIN) + GAIN*emphasis/fraction).clamp(max=WEIGHT_CAP)
    weight = weight / (weight*fraction).sum()
    return weight[codes]


@lru_cache(maxsize=1)
def prepared_training(rotation):
    text, label, teacher, project, digests = partition(rotation)
    tok = small.tokenizer()
    plain = tokenize(text, tok)
    normalized = tokenize(parallel(text), tok)
    return label, teacher, project, digests, tok, plain, normalized, lexical_features(text)


def fit(rotation, seed, modes):
    scope = "official" if rotation == "official" else "projects"
    out = RESULTS/rotation/str(seed)
    out.mkdir(parents=True, exist_ok=True)
    label, teacher, project, digests, tok, plain, normalized, lexical = prepared_training(rotation)
    rng = random.Random(seed)
    sequences = [b if rng.random() < .45 else a for a, b in zip(plain, normalized, strict=True)]
    del plain, normalized
    order = list(range(len(sequences)))
    rng.shuffle(order)
    steps = list(batch_plan(order, sequences, small.BATCH))
    rng.shuffle(steps)
    source = checkpoint(scope)
    side = torch.as_tensor(lexical, device="cuda")
    gold = torch.as_tensor(label, device="cuda", dtype=torch.float32)
    soft = (torch.as_tensor(teacher, device="cuda").clamp(1e-5, 1-1e-5).logit()/1.7).sigmoid()
    positive_weight = torch.as_tensor(np.clip((len(label)-label.sum(0))/label.sum(0), 1, 5),
                                      device="cuda", dtype=torch.float32)
    names, code, size = np.unique(project, return_inverse=True, return_counts=True)
    group = torch.as_tensor(code, device="cuda")
    fraction = torch.as_tensor(size/size.sum(), device="cuda", dtype=torch.float32)
    # Square root preserves representation for very small projects and limits their influence.
    prior = fraction.sqrt()
    prior /= prior.sum()
    domain_counts = np.zeros((2, len(names)), dtype=np.int64)
    np.add.at(domain_counts, (label[:,0], code), 1)
    domain_weights = np.zeros_like(domain_counts, dtype=np.float32)
    occupied = domain_counts > 0
    domain_weights[occupied] = 1/np.sqrt(domain_counts[occupied])
    domain_weights *= len(label)/np.sum(domain_weights*domain_counts)
    domain_weights = torch.as_tensor(domain_weights, device="cuda")
    config = dict(epochs=1, seed=seed, rows=len(label), steps=len(steps), warmup=.04, lr_emb=.00012,
                  lr_model=.00025, weight_decay=.025, normalize_probability=.45, temperature=TEMPERATURE,
                  project_mix=GAIN, smoothing=EMA, weight_cap=WEIGHT_CAP, distillation_temperature=1.7,
                  domain_gradient=DOMAIN_GRADIENT)
    record = {}
    for mode in modes:
        torch.manual_seed(seed)
        model = load_original(tok, "control", source)
        model.train()
        params = [dict(params=model.embed.parameters(), lr=config["lr_emb"]),
                  dict(params=[p for k,p in model.named_parameters() if not k.startswith("embed.")],
                       lr=config["lr_model"])]
        adversary = None
        if mode == ADVERSARY:
            adversary = torch.nn.Sequential(torch.nn.LayerNorm(model.width),
                                           torch.nn.Linear(model.width, 2*len(names))).cuda()
            params.append(dict(params=adversary.parameters(), lr=.0008))
        opt = torch.optim.AdamW(params, weight_decay=.025, fused=True)
        scheduler = get_linear_schedule_with_warmup(opt, int(.04*len(steps)), len(steps))
        running = torch.zeros_like(fraction)
        seen = torch.zeros_like(fraction, dtype=torch.bool)
        torch.manual_seed(seed)
        start = time.perf_counter()
        for i, rows in enumerate(steps, 1):
            token, mask = batch(rows, sequences, tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if adversary is None:
                    p = model(token, mask, side[rows])
                else:
                    states = model.token_states(token, mask)
                    p = model.readout(states, mask, side[rows])
                    feat = (states.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                    detached = feat.detach()
                    alpha = DOMAIN_GRADIENT * min(1., 4*i/len(steps))
                    reversed_view = detached - alpha*(feat-detached)
                    classes = gold[rows,0].long()
                    domain = adversary(reversed_view).view(-1,2,len(names))[torch.arange(len(rows),device="cuda"),classes]
                    g = group[rows]
                    project_loss = F.cross_entropy(domain,g,reduction="none")
                    domain_loss = (project_loss*domain_weights[classes,g]).mean()
                supervised = F.binary_cross_entropy_with_logits(p, gold[rows], pos_weight=positive_weight,
                                                                reduction="none")
                distill = F.binary_cross_entropy_with_logits(p/1.7, soft[rows], reduction="none")
                loss = .7*supervised+.3*1.7**2*distill
                if mode in ("pooled",ADVERSARY):
                    scalar = loss.mean() if adversary is None else loss.mean() + .30*domain_loss
                else:
                    ids = group[rows]
                    if mode == "robust":
                        with torch.no_grad():
                            totals = torch.zeros_like(fraction).scatter_add_(0, ids, supervised[:,0].detach())
                            sizes = torch.zeros_like(fraction).scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
                            present = sizes > 0
                            current = totals/sizes.clamp_min(1.)
                            initial = torch.where(seen, running, supervised[:,0].detach().mean())
                            # Shrunken batch means keep singleton observations from moving an entire group.
                            step = EMA*sizes/(sizes+8.)
                            running = torch.where(present, initial+step*(current-initial), initial)
                            seen |= present
                    w = row_weights(prior, fraction, running, ids, mode)
                    scalar = .5*((loss[:,0]*w).mean()+loss[:,1].mean())
            scalar.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            scheduler.step()
            if i % 400 == 0:
                print(rotation, seed, mode, i, len(steps), round(time.perf_counter()-start,1), flush=True)
        path = out/f"{mode}.pt"
        temporary = path.with_suffix(".part")
        torch.save(model.state_dict(), temporary)
        temporary.replace(path)
        final = row_weights(prior, fraction, running, torch.arange(len(size),device="cuda"), mode)
        weights = np.ones(len(size)) if mode in ("pooled",ADVERSARY) else final.cpu().numpy()
        record[mode] = dict(checkpoint=path.name, sha256=checksum(path), seconds=time.perf_counter()-start,
                            group_weights={name:float(w) for name,w in zip(names,weights,strict=True)})
        write_json(out/"fit.json", dict(config=config, source_sha256=source[1], teacher_and_data=digests,
                                       group_rows=dict(zip(names, map(int,size),strict=True)), modes=record))
        del model, opt, scheduler


def open_encoder(rotation, seed, name, tok):
    scope = "official" if rotation == "official" else "projects"
    if name in ("reference", "threshold"):
        source = checkpoint(scope)
    else:
        folder = RESULTS/rotation/str(seed)
        entry = json_at(folder/"fit.json")["modes"][name]
        source = folder/entry["checkpoint"], entry["sha256"]
    model = load_original(tok, "control", source)
    for block in model.local:
        block.use_shifted = True
    return SimpleNamespace(encoder=model.eval(), tok=tok, mode="control")


def project_scores(rotation, seed, modes):
    source = HERE/f"{rotation}.npz"
    if checksum(source) != json_at(HERE/"partition.json")["rotations"][rotation]["partition_sha256"]:
        raise ValueError("Project partition changed")
    texts, label, project, files, _ = inputs()
    folder = RESULTS/rotation/str(seed)
    original_cutoff = json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    route, _ = pinned()
    tok = small.tokenizer()
    with np.load(source) as parts:
        entries = {role:parts[role] for role in ("dev","test")}
    result, policy = {}, {}
    for role, ids in entries.items():
        subset = [texts[i] for i in ids]
        seq = tokenize(subset, tok)
        alternate = parallel(subset)
        canonical = tokenize(alternate, tok)
        y = label[ids,0]
        prediction = {}
        for name in ("reference", *modes):
            pipe = open_encoder(rotation, seed, name, tok)
            p, _ = infer(pipe, subset, sequences=seq)
            q, _ = infer(pipe, alternate, sequences=canonical)
            if not np.array_equal(p[:,1],q[:,1]):
                raise ValueError("Identifier normalization changed the MAT rule")
            s, active = selected_views(p[:,0],q[:,0],route)
            prediction[name] = (s, active, p[:,0], q[:,0], p[:,1])
        base, *_ = prediction["reference"]
        budget = counts(y,base>original_cutoff)["fp"]
        ref,_ = exact_budget(base,y,budget)
        grouping = group_codes(zip(project[ids],[files[i] for i in ids],strict=True))
        old = base>(policy["reference"]["threshold"] if policy else original_cutoff)
        result[role] = {}
        for name, (s, active, p, q, mat) in prediction.items():
            if not np.array_equal(mat,prediction["reference"][-1]):
                raise ValueError("MAT decisions changed during project evaluation")
            equal,_ = exact_budget(s,y,budget)
            if role == "dev":
                _, cut = strict_budget(s,y,budget)
                policy[name] = dict(threshold=cut, delta_tp=comparison(y,ref,equal)["delta_tp"])
            frozen = s>policy[name]["threshold"]
            by_project = {}
            project_equal = np.zeros(len(y),dtype=bool)
            project_base = np.zeros(len(y),dtype=bool)
            for family in sorted(set(project[ids])):
                group = project[ids]==family
                group_budget = counts(y[group],ref[group])["fp"]
                matched,_ = exact_budget(s[group],y[group],group_budget)
                base_match,_ = exact_budget(base[group],y[group],group_budget)
                project_equal[group] = matched
                project_base[group] = base_match
                by_project[family] = comparison(y[group],base_match,matched)
            result[role][name] = dict(equal=comparison(y,ref,equal),
                frozen=comparison(y,old,frozen), by_project=by_project,
                matched_each_project=comparison(y,project_base,project_equal),
                paired_interval=grouped_delta(y,old,frozen,grouping) if role=="test" else None,
                routed=int(active.sum()), rows=len(ids), budget=budget)
            with (folder/f"{role}_{name}.csv").open("w",newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("row","project","label","mat","probability","raw","canonical","routed",
                                 "baseline_equal","equal","baseline_by_project","by_project","baseline_frozen","frozen"))
                writer.writerows(zip(ids,project[ids],y,mat.astype(int),s,p,q,active.astype(int),ref.astype(int),
                                   equal.astype(int),project_base.astype(int),project_equal.astype(int),old.astype(int),
                                   frozen.astype(int),strict=True))
            print(rotation, seed, role, name, budget, result[role][name]["equal"]["delta_tp"],
                  result[role][name]["matched_each_project"]["delta_tp"], flush=True)
        if role == "dev":
            chosen = max(policy,key=lambda n:(policy[n]["delta_tp"],n=="reference"))
            write_json(folder/"policy.json",dict(chosen=chosen, modes=policy, route=route, budget=budget))
    write_json(folder/"results.json",dict(selected=chosen, **result))


def transfer(modes):
    evidence = [json_at(RESULTS/r/str(s)/"results.json") for r in ROTATIONS for s in SEEDS]
    dev = {name:[v["dev"][name]["equal"]["delta_tp"] for v in evidence] for name in modes}
    test = {name:[v["test"][name]["matched_each_project"]["delta_tp"] for v in evidence] for name in modes}
    selected = max(modes,key=lambda name:sum(dev[name]))
    report = dict(selected=selected, dev=dev, heldout=test,
                  pairs=[(r,s) for r in ROTATIONS for s in SEEDS])
    path = RESULTS/"official"
    path.mkdir(exist_ok=True)
    write_json(path/"transfer.json",report)
    print(report,flush=True)
    fit("official",42,tuple(dict.fromkeys(("pooled",selected))))


def tune():
    folder = RESULTS/"official"
    text, y, ids, cal, review = heldout(include_test=False)
    route, threshold = pinned()
    chosen = json_at(folder/"transfer.json")["selected"]
    tok = small.tokenizer()
    base, base_cost = run(open_encoder("official",42,"reference",tok),text,route,tokenize)
    old = base[:,0]>threshold
    budget = counts(y[cal,0],old[cal])["fp"]
    ref,_ = exact_budget(base[cal,0],y[cal,0],budget)
    rows = {}
    for n in dict.fromkeys(("reference","threshold","pooled",chosen)):
        p,c = (base,base_cost) if n in ("reference","threshold") else run(open_encoder("official",42,n,tok),text,route,tokenize)
        if not np.array_equal(p[:,1],base[:,1]):
            raise ValueError("Official MAT decisions changed")
        _, cut = strict_budget(p[cal,0],y[cal,0],budget)
        if n=="reference":
            cut = threshold
        b = p[cal]>(cut,.5)
        score = competition_formula(metric(y[cal],b)["mean"]["f1"],0,c["analytic_gflops"]*30655/len(y))
        rows[n] = dict(threshold=cut, projected_score=score, calibration=comparison(y[cal,0],ref,b[:,0]),
                       review=comparison(y[review,0],old[review],p[review,0]>cut), cost=c)
        print(n,rows[n],flush=True)
    selected = max(rows,key=lambda n:(rows[n]["projected_score"],n=="reference"))
    write_json(folder/"policy.json",dict(selected=selected, modes=rows, route=route, fp=budget,
                  cal_rows=ids[cal].tolist(),review_rows=ids[review].tolist()))


def public():
    folder = RESULTS/"official"
    rule = json_at(folder/"policy.json")
    names = list(dict.fromkeys(("reference",rule["selected"])))
    tok = small.tokenizer()
    pipes = {n:open_encoder("official",42,n,tok) for n in names}
    benchmark(folder,names,rule,lambda n,text:run(pipes[n],text,rule["route"],tokenize))


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","score","transfer","tune","public"))
    parser.add_argument("--rotation",choices=ROTATIONS,default="test_a")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--modes",nargs="+",choices=(*ARMS,ADVERSARY),default=ARMS)
    parser.add_argument("--output-root",type=Path,default=HERE)
    args = parser.parse_args()
    RESULTS = args.output_root.resolve()
    if args.stage in ("fit","score"):
        (fit if args.stage=="fit" else project_scores)(args.rotation,args.seed,args.modes)
    elif args.stage=="transfer":
        transfer(args.modes)
    else:
        {"tune":tune,"public":public}[args.stage]()


if __name__ == "__main__":
    main()
