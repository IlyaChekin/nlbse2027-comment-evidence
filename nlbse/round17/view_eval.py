"""Freeze operating points on development data and compare full-context training."""

import argparse
import csv
from statistics import median

import numpy as np

from view_train import HERE, ROOT, MODES, REPR_MODES, SEEDS, checksum, inputs, json_at, open_view, small, tokenize
from experiment import comparison, infer, strict_budget
from evaluate import counts, exact_budget, grouped_delta, raw
from local_public import heldout, run, write_public
from normalize_code import parallel
from pipeline import PUBLIC, load, metric, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from students import group_codes


SOURCE = ROOT/"round16/context_public"
OTHER = HERE/"official"/"evaluation"


def pinned():
    cfg = json_at(SOURCE/"policy.json")
    return cfg["route"], cfg["modes"]["full"]["cutoff"]


def selected_views(p, canonical, route):
    eligible = (p > route["lower"]) & (p < route["upper"])
    return np.where(eligible, .5*(p+canonical), p), eligible


def offline(scope, seed, models, texts, experiment=None):
    tok = small.tokenizer()
    clean = parallel(texts)
    a, b = tokenize(texts, tok), tokenize(clean, tok)
    output = {}
    for name in models:
        pipe = open_view(scope, seed, name, tok, experiment)
        p, _ = infer(pipe, texts, sequences=a)
        q, _ = infer(pipe, clean, sequences=b)
        if not np.array_equal(p[:,1],q[:,1]):
            raise ValueError("Identifier normalization changed the MAT decision")
        output[name] = [p[:,0],q[:,0]]
    return output,np.asarray([len(v) for v in a])


def variance(p,q,base):
    active = (base>.35)&(base<.85)
    if not active.any():
        return dict(eligible=0, abs_gap=None)
    return dict(eligible=int(active.sum()),abs_gap=float(np.mean(np.abs(p[active]-q[active]))))


def project(seed, representation=False, root=None, variants=None):
    texts,labels,projects,files,roles = inputs()
    route,_ = pinned()
    frozen = json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    modes = variants or (REPR_MODES if representation else MODES)
    models = ("reference", *modes)
    source = root or HERE/("representation" if representation else "projects")
    folder = source/str(seed)
    i = roles["dev"]
    probabilities,_ = offline("projects",seed,models,[texts[j] for j in i],source)
    y = labels[i,0]
    fp = counts(y,raw("dev")["compact"]>frozen)["fp"]
    base,_ = selected_views(*probabilities["reference"],route)
    ref,_ = exact_budget(base,y,fp)
    rules = {}
    for name,(p,q) in probabilities.items():
        mixed,routed = selected_views(p,q,route)
        bits,cut = strict_budget(mixed,y,fp)
        rules[name] = dict(threshold=cut,calibration=comparison(y,ref,bits),n_routed=int(routed.sum()),gap=variance(p,q,probabilities["reference"][0]))
    winner = max(models,key=lambda k:(rules[k]["calibration"]["delta_tp"],k=="reference"))
    write_json(folder/"policy.json",dict(seed=seed,selected=winner,budget=fp,modes=rules,route=route))

    i = roles["test"]
    y = labels[i,0]
    probabilities,length = offline("projects",seed,models,[texts[j] for j in i],source)
    base,_ = selected_views(*probabilities["reference"],route)
    old = base>rules["reference"]["threshold"]
    budget = counts(y,raw("test")["compact"]>frozen)["fp"]
    b,_ = exact_budget(base,y,budget)
    clusters = group_codes(zip(projects[i],[files[j] for j in i],strict=True))
    report = {}
    for name,(p,q) in probabilities.items():
        s,routed = selected_views(p,q,route)
        bits,_ = exact_budget(s,y,budget)
        fix = s>rules[name]["threshold"]
        raw_bits,_ = exact_budget(p,y,budget)
        report[name] = dict(equal=comparison(y,b,bits),frozen=comparison(y,old,fix),
            raw_only=comparison(y,b,raw_bits),paired_interval=grouped_delta(y,old,fix,clusters),
            long=comparison(y[length>512],b[length>512],bits[length>512]),
            short=comparison(y[length<=512],b[length<=512],bits[length<=512]),
            n_routed=int(routed.sum()),gap=variance(p,q,probabilities["reference"][0]))
        with (folder/f"{name}_test.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","label","length","raw","canonical","score","routed","before_equal","after_equal","before_frozen","after_frozen"))
            w.writerows(zip(i,projects[i],y,length,p,q,s,routed.astype(int),b.astype(int),bits.astype(int),old.astype(int),fix.astype(int),strict=True))
        print(seed,name,report[name]["equal"],report[name]["gap"],flush=True)
    write_json(folder/"results.json",dict(rows=len(y),selected=winner,budget=budget,modes=report))


def plan_official():
    records = [json_at(HERE/"projects"/str(seed)/"policy.json")["modes"] for seed in SEEDS]
    gains = {k:[r[k]["calibration"]["delta_tp"] for r in records] for k in MODES}
    # Always compare the crop continuation, the full-context continuation and the stronger paired arm.
    paired = max(("paired","consistent"),key=lambda k:sum(gains[k]))
    OTHER.mkdir(parents=True,exist_ok=True)
    write_json(OTHER/"plan.json",dict(modes=["crop","full",paired],selection=gains,source_seeds=SEEDS))
    print("crop full " + paired,flush=True)


def tune():
    texts,y,ids,cal,review = heldout(include_test=False)
    route,threshold = pinned()
    plan = json_at(OTHER/"plan.json")
    tok = small.tokenizer()
    p,base_cost = run(open_view("official",42,"reference",tok),texts,route,tokenize)
    original = p[:,0]>threshold
    budget = counts(y[cal,0],original[cal])["fp"]
    ref,_ = exact_budget(p[cal,0],y[cal,0],budget)
    rules = {}
    for name in ("reference","threshold",*plan["modes"]):
        q,cost = (p,base_cost) if name in {"reference","threshold"} else run(open_view("official",42,name,tok),texts,route,tokenize)
        if not np.array_equal(q[:,1],p[:,1]):
            raise ValueError("Continued training changed the MAT decision")
        _,cut = strict_budget(q[cal,0],y[cal,0],budget)
        cut = threshold if name=="reference" else cut
        bits = q[cal]>(cut,.5)
        projection = competition_formula(metric(y[cal],bits)["mean"]["f1"],0.,cost["analytic_gflops"]*30655/len(y))
        rules[name] = dict(threshold=cut,projected_score=projection,calibration=comparison(y[cal,0],ref,bits[:,0]),
                           review=comparison(y[review,0],original[review],q[review,0]>cut),cost=cost)
        print(name,rules[name],flush=True)
    selected = max(rules,key=lambda n:(rules[n]["projected_score"],n=="reference"))
    write_json(OTHER/"policy.json",dict(selected=selected,cal_rows=ids[cal].tolist(),review_rows=ids[review].tolist(),
                                      fp=budget,modes=rules,route=route))


def benchmark(folder, models, rule, runner, reference=SOURCE/"full_public.csv"):
    texts,y,meta = load(PUBLIC,audit=True)
    ids = list(range(len(y)))
    with reference.open(newline="") as stream:
        archive = list(csv.DictReader(stream))
    if [int(r["row"]) for r in archive]!=ids:
        raise ValueError("The reference does not cover the full public set")
    old = np.asarray([[int(r[k]) for k in ("vulnerability","mat")] for r in archive],dtype=bool)
    groups = group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    timings = {n:[] for n in models}
    outputs = {}
    for name in models:
        runner(name,texts[:2048])
    for t in range(3):
        for name in models if t%2==0 else reversed(models):
            p,cost = runner(name,texts)
            bits = p>(rule["modes"][name]["threshold"],.5)
            if name=="reference" and not np.array_equal(bits,old):
                raise ValueError("Archived reference decisions changed")
            if name in outputs and not np.array_equal(bits,outputs[name][1]):
                raise ValueError("Repeated predictions changed")
            timings[name].append(cost)
            outputs[name] = p,bits
            print(name,cost,flush=True)
    fp = counts(y[:,0],old[:,0])["fp"]
    ref,_ = exact_budget(outputs["reference"][0][:,0],y[:,0],fp)
    result = {}
    for name,times in timings.items():
        p,bits = outputs[name]
        if not np.array_equal(bits[:,1],old[:,1]):
            raise ValueError("Measured MAT predictions changed")
        equal,_ = exact_budget(p[:,0],y[:,0],fp)
        cost = {k:median(v[k] for v in times) for k in times[0]}
        quality = metric(y,bits)
        result[name] = dict(score=competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
            quality=quality,cost=cost,trials=times,matched=comparison(y[:,0],ref,equal),frozen=comparison(y[:,0],old[:,0],bits[:,0]),
            interval=bootstrap_delta(y,old,bits,groups))
        write_public(folder/f"{name}_public.csv",ids,y,p,bits,equal,ref)
    write_json(folder/"results.json",result)
    print({k:(r["score"],r["matched"]["delta_tp"]) for k,r in result.items()},flush=True)


def public():
    rule = json_at(OTHER/"policy.json")
    models = list(dict.fromkeys(("reference",rule["selected"])))
    tok = small.tokenizer()
    pipes = {n:open_view("official",42,"reference" if n=="threshold" else n,tok) for n in models}
    benchmark(OTHER, models, rule, lambda name,texts:run(pipes[name],texts,rule["route"],tokenize))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("project","plan","tune","public"))
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--representation",action="store_true")
    args=ap.parse_args()
    if args.stage=="project":
        project(args.seed,args.representation)
    else:
        {"plan":plan_official,"tune":tune,"public":public}[args.stage]()
