"""Freeze an internal rule combiner and time the complete public pipeline."""
import argparse
import os
import time

import joblib
import numpy as np
from replay import HERE, checksum, comparison, counts, exact_budget, fit_combiner
from replay import strict_budget, suite, write_json
from transfer import ROOT, benchmark, json_at, open_encoder, pinned, run, small, tokenize
from pipeline import metric
from score_runtime import competition_formula
from students import dataset
from runtime import GATES, get_features, refine, scanner


OUT = HERE/"official"


def internal():
    _, _, texts, y, ids, _ = dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as folds:
        if not np.array_equal(ids,folds["rows"]):
            raise ValueError("Internal calibration rows have moved")
        membership = folds["folds"]
    return texts,y,np.asarray(ids),membership


def project_selection():
    pairs = {role:json_at(HERE/role/"results.json") for role in ("test_a","test_b")}
    support = {}
    for key in pairs["test_a"]["modes"]:
        if key == "rule_priority":
            continue
        gain = [r["modes"][key]["each_project"]["delta_tp"] for r in pairs.values()]
        if min(gain) > 0:
            support[key] = gain
    if not support:
        raise ValueError("No rule policy improved both held-out groups at matched per-project FP")
    return support


def fit():
    allowed = project_selection()
    OUT.mkdir(exist_ok=True)
    texts, y, ids, group = internal()
    route, threshold = pinned()
    tok = small.tokenizer()
    pipe = open_encoder("official",42,"reference",tok)
    p,cost = run(pipe,texts,route,tokenize)
    at = time.perf_counter()
    names,raw = scanner(texts)
    all_scan_time = time.perf_counter()-at
    selected = group<3
    calibration = group==3
    review = group==4
    _, _, x = get_features(p[:,0],raw)
    models = fit_combiner(x[selected],y[selected,0])
    catalog = OUT/"models.joblib"
    joblib.dump(models,catalog,compress=3)
    old = p[:,0]>threshold
    local = y[:,0]
    budget = counts(local[calibration],old[calibration])["fp"]
    base,_ = exact_budget(p[calibration,0],local[calibration],budget)
    result = {}
    for name, score in suite(p[:,0],raw,models).items():
        if name!="reference" and name not in allowed:
            continue
        diag, cut = strict_budget(score[calibration],local[calibration],budget)
        if name=="reference":
            cut = threshold
            fraction = 0.
        else:
            gate = name.rsplit("_",1)[-1] if not name.startswith("rule_") else "all"
            lo,hi = GATES[gate]
            fraction = 1. if gate=="all" else float(((p[:,0]>lo)&(p[:,0]<hi)).mean())
        b = np.column_stack((score[calibration]>cut,p[calibration,1]>.5))
        f1 = metric(y[calibration],b)["mean"]["f1"]
        scaled = 30655/len(y)
        time_proxy = (cost["forward_seconds"]+fraction*all_scan_time)*scaled
        grade = competition_formula(f1,time_proxy,cost["analytic_gflops"]*scaled)
        result[name] = dict(threshold=cut, score_proxy=grade, scan_fraction=fraction,
                            calibration=comparison(local[calibration],base,diag),
                            review=comparison(local[review],old[review],score[review]>cut))
        print(name,result[name],flush=True)
    choice = max(result,key=lambda n:(result[n]["score_proxy"],n=="reference"))
    write_json(OUT/"policy.json",dict(selected=choice, modes=result, route=route, allowed=allowed,
        rule_columns=names, model_sha256=checksum(catalog), rule_source_sha256=checksum(HERE/"rules_manifest.json"),
        scanner_sha256=checksum(HERE/"scan.mjs"), worker_limit=int(os.environ.get("NLBSE_SCANNER_WORKERS","6")),
        calibration_rows=ids[calibration].tolist(), review_rows=ids[review].tolist(),
        scan_training_seconds=all_scan_time, restore_exact_public_reference=True))


def public():
    rule = json_at(OUT/"policy.json")
    catalog = OUT/"models.joblib"
    if (checksum(catalog)!=rule["model_sha256"] or checksum(HERE/"rules_manifest.json")!=rule["rule_source_sha256"]
            or checksum(HERE/"scan.mjs")!=rule["scanner_sha256"]
            or int(os.environ.get("NLBSE_SCANNER_WORKERS","6"))!=rule["worker_limit"]):
        raise ValueError("Rule policy dependencies changed")
    heads = joblib.load(catalog)
    choice = rule["selected"]
    pipe = open_encoder("official",42,"reference",small.tokenizer())

    def inference(mode,texts):
        p,cost = run(pipe,texts,rule["route"],tokenize)
        if mode=="reference":
            return p,cost
        at = time.perf_counter()
        p[:,0] = refine(texts,p[:,0],rule,heads)
        added = time.perf_counter()-at
        cost["forward_seconds"] += added
        cost["wall_seconds"] += added
        return p,cost

    benchmark(OUT,list(dict.fromkeys(("reference",choice))),rule,inference)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fit","public"))
    a=p.parse_args()
    {"fit":fit,"public":public}[a.stage]()
