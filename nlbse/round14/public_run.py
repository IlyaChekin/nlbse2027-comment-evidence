"""Benchmark a frozen MAT residual alongside the released vulnerability inference."""

import argparse
import csv
import json
from statistics import median
import sys
import time

import numpy as np

from audit import HERE, ROOT, dataset, write_json
from mat import cut_at_budget, evaluate, load_policy, score_one

sys.path[:0] = [str(ROOT/"round12"), str(ROOT/"round10"), str(ROOT/"round11")]
from efficient_aug import OUT as RELEASE, run, startup
from pipeline import metric
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from students import group_codes, predict_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("chosen", "marker_geometry"), default="chosen")
    parser.add_argument("--training", choices=("project", "official"), default="official")
    args = parser.parse_args()
    recipe, weights = load_policy(args.training)
    selected = recipe["chosen"] if args.mode == "chosen" else recipe["by_mode"][args.mode]
    name = selected["mode"]
    head = weights.get(selected["key"])
    heldout = json.loads((HERE / "project_results.json").read_text())
    result_dir = HERE / "public" / args.training / name
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RELEASE / "policy.json").read_text())["selected"]["selective"]
    texts, y, projects, paths = dataset(True)
    archived = predict_bits(RELEASE/"selective/predictions.csv", list(range(len(y))), {"thresholds": (cfg["threshold"], .5)})
    labels = y[:,1].astype(bool)
    pipe = startup()
    trials, scores, predictions = {}, {}, {}
    for mode in ("reference", "candidate"):
        run(pipe, texts[:2048], cfg)
        if mode == "candidate":
            score_one(texts[:2048], name, head)
        trials[mode] = []
    for index in range(3):
        for mode in (("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")):
            begin = time.perf_counter()
            p, cost = run(pipe, texts, cfg)
            base = p > (cfg["threshold"], .5)
            if not np.array_equal(base, archived):
                raise ValueError("Released reference changed")
            start = time.perf_counter()
            score = score_one(texts, name, head) if mode == "candidate" else p[:,1]
            added = time.perf_counter()-start if mode == "candidate" else 0.
            if mode == "candidate" and not np.array_equal(score == 2, archived[:,1]):
                raise ValueError("The candidate changed the tag scanner itself")
            elapsed = time.perf_counter()-begin
            cost.update(wall_seconds=elapsed, mat_seconds=added, forward_seconds=cost["forward_seconds"]+added)
            bits = base.copy()
            if mode == "candidate":
                bits[:,1] = score > selected["threshold"]
            if mode in predictions and not np.array_equal(bits, predictions[mode]):
                raise ValueError("Different predictions between benchmark repeats")
            trials[mode].append(cost)
            predictions[mode] = bits
            scores[mode] = score
            print(mode,cost,flush=True)
    budget = int((archived[:,1] & ~labels).sum())
    exact_cut = cut_at_budget(scores["candidate"], labels, budget)
    exact = scores["candidate"] > exact_cut
    groups = group_codes(zip(projects, paths, strict=True))
    result = {"config": selected, "matched_fp": evaluate(labels,archived[:,1],exact),
              "matched_cutoff": exact_cut, "heldout": {k: r[name]["exact"] for k,r in heldout.items()}}
    for mode, raw in trials.items():
        costs = {k: median(r[k] for r in raw) for k in raw[0]}
        quality = metric(y, predictions[mode])
        result[mode] = dict(metrics=quality, mat=evaluate(labels,archived[:,1],predictions[mode][:,1]),
                            score=competition_formula(quality["mean"]["f1"],costs["forward_seconds"],costs["analytic_gflops"]),
                            wall_charged=competition_formula(quality["mean"]["f1"],costs["wall_seconds"],costs["analytic_gflops"]),
                            interval=bootstrap_delta(y,archived,predictions[mode],groups), cost=costs, trials=raw)
        with (result_dir/f"{mode}.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","label_vulnerability","label_MAT","pred_vulnerability","pred_MAT","score_MAT","MAT_matched_budget"))
            ex = exact if mode == "candidate" else archived[:,1]
            w.writerows(zip(range(len(y)),y[:,0],y[:,1],predictions[mode][:,0].astype(int),
                            predictions[mode][:,1].astype(int),scores[mode],ex.astype(int),strict=True))
    write_json(result_dir/"analysis.json", result)
    print(name, {k: result[k]["score"] for k in trials}, result["matched_fp"], flush=True)


if __name__ == "__main__":
    main()
