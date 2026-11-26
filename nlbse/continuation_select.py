"""Freeze thresholds and compare NLBSE candidates using only official train rows."""
import argparse
import csv
import json

import numpy as np

from pipeline import ROOT, SPLIT, TRAIN, load, metric, write_json
from score_runtime import competition_formula

RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
HIDDEN_ROWS = 61311


def read_json(path):
    return json.loads(path.read_text())


def split():
    with SPLIT.open(newline="") as file:
        tune = [r for r in csv.DictReader(file) if r["partition"] == "tune"]
    calibration_families = {r["function_sha256"] for r in tune if r["pilot"] == "1"}
    calibration = np.array([r["function_sha256"] in calibration_families for r in tune])
    return [int(r["row"]) for r in tune], calibration


def read_predictions(run, ids):
    with (RUNS / run / "predictions.csv").open(newline="") as file:
        predictions = {int(r["row"]): (float(r["p_vulnerability"]), float(r["p_MAT"]))
                       for r in csv.DictReader(file)}
    if set(predictions) != set(ids):
        raise ValueError(f"{run}: mismatched holdout coverage")
    return np.array([predictions[i] for i in ids])


def best_thresholds(gold, probabilities):
    thresholds = np.linspace(0.04, 0.96, 47)
    chosen = []
    for j in range(2):
        f1 = []
        for value in thresholds:
            pred = probabilities[:, j] > value
            tp = np.sum(pred & (gold[:, j] == 1))
            f1.append(2 * tp / max(1, np.sum(pred) + np.sum(gold[:, j])))
        idx = max(range(len(f1)), key=lambda i: (f1[i], -abs(thresholds[i] - 0.5)))
        chosen.append(round(float(thresholds[idx]), 2))
    return chosen


def pick_compact_pilot():
    options = ("compact6_profile_tune", "compact6_kd_profile_tune")
    results = {name: read_json(RUNS / name / "metrics.json")["scores"] for name in options}
    winner = max(options, key=lambda name: results[name]["mean"]["f1"])
    write_json(REPORTS / "compact_pilot_selection.json",
               dict(criterion="mean F1 on common pilot tune rows", results=results, winner=winner))
    print("kd" if winner == options[1] else "bce")


def select():
    _, labels, _ = load(TRAIN)
    ids, calibration = split()
    labels = labels[ids]
    options = {
        "own12": ("own12_packed_tune", "runs/final_fit/checkpoint"),
        "own6": ("own6_packed_tune", "runs/compact6_full/checkpoint"),
    }
    results = {}
    for name, (run, checkpoint) in options.items():
        probabilities = read_predictions(run, ids)
        thresholds = best_thresholds(labels[calibration], probabilities[calibration])
        metric_result = metric(labels[~calibration], probabilities[~calibration] > thresholds)
        cost = read_json(RUNS / run / "metrics.json")["cost"]
        ratio = HIDDEN_ROWS / cost["rows"]
        forward_proxy = cost["forward_seconds"] * ratio
        gflops_proxy = cost["profiler_counted_gflops"] * ratio
        results[name] = dict(
            checkpoint=checkpoint, thresholds=thresholds, selection_metrics=metric_result,
            measured_tune_cost=cost,
            projected_rank_at_61311=competition_formula(metric_result["mean"]["f1"], forward_proxy, gflops_proxy),
            projected_forward_seconds=forward_proxy, projected_gflops=gflops_proxy,
        )
        write_json(REPORTS / f"{name}_frozen_thresholds.json", thresholds)
    winner = max(results, key=lambda name: results[name]["projected_rank_at_61311"])
    document = dict(
        winner=winner, calibration_rows=int(calibration.sum()), selection_rows=int((~calibration).sum()),
        cost_scope="single local profiled pass over full internal holdout, linearly projected; organizer score pending",
        threshold_grid="0.04–0.96 in steps of 0.02, ties favor proximity to 0.50",
        criterion="competition formula on internal F1 and projected local cost",
        candidates=results,
    )
    write_json(REPORTS / "round2_selection.json", document)
    print(winner)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "freeze"))
    (pick_compact_pilot if parser.parse_args().mode == "pilot" else select)()
