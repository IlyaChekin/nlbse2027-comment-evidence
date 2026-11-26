"""Freeze CodeT5+ context policies using the official training holdout only."""
import csv
from dataclasses import asdict
import json

import numpy as np

from cache_context import HERE
from context_policy import THRESHOLDS, combine, first_rows, jobs, policies
from continuation_select import best_thresholds
from pipeline import metric, write_json
from score_runtime import competition_formula
from windows import BATCH, BLOCK
from compare import dense_gflops

PUBLIC_ROWS = 30655
REFERENCE_PUBLIC_FORWARD = 13.427988981595263


def shapes(lengths, mask):
    for start in range(0, len(lengths), BLOCK):
        use = lengths[start:start+BLOCK][mask[start:start+BLOCK]]
        ordered = np.sort(use)
        for idx in range(0, len(ordered), BATCH):
            block = ordered[idx:idx+BATCH]
            yield len(block), int(block[-1])


def apply_cached(policy, data, dims):
    head = data["head"].copy()
    lengths = data["size_head"].copy()
    if policy.first:
        mask = first_rows(policy, data)
        head[mask] = data[policy.first][mask]
        lengths[mask] = data["size_"+policy.first][mask]
    compute = dense_gflops(shapes(lengths, np.ones(len(head), dtype=bool)), dims)
    result = head.copy()
    count = 0
    for name, role, mask in jobs(policy, head, data):
        validate = data[name][mask]
        if not np.isfinite(validate).all():
            raise ValueError(f"Invalid {name} scores for {policy.name}")
        count += int(mask.sum())
        compute += dense_gflops(shapes(data["size_"+name], mask), dims)
        combine(policy, head, data[name], mask, role, result)
    return result, compute, count


def main():
    with np.load(HERE / "holdout_scores.npz") as stored:
        data = dict(stored)
    dims = json.loads((HERE / "cache_config.json").read_text())["dimensions"]
    y, cal = data["gold"], data["calibration"]
    variants = {policy.name: policy for policy in policies()}
    baseline_gflops = apply_cached(variants["reference"], data, dims)[1]
    receipts = {}
    decisions = {}
    for name, policy in variants.items():
        p, compute, windows = apply_cached(policy, data, dims)
        bits = p > THRESHOLDS
        quality = metric(y[~cal], bits[~cal])
        local = competition_formula(quality["mean"]["f1"],
                                    REFERENCE_PUBLIC_FORWARD * compute/baseline_gflops,
                                    compute * PUBLIC_ROWS/len(y))
        threshold = best_thresholds(y[cal], p[cal])
        receipts[name] = dict(config=asdict(policy), scores=quality, extra_windows=windows,
                              analytic_gflops=compute, projected_public_cost_formula=local,
                              calibrated_thresholds_diagnostic=threshold,
                              calibrated_scores_diagnostic=metric(y[~cal], p[~cal] > threshold))
        decisions[name] = bits
    if not np.array_equal(decisions["reference"], decisions["skip_positive"]):
        raise ValueError("The theoretically redundant passes changed cached decisions")
    diagnostic = ("seen_control", "equal_length_tail", "all_long_tail")
    standard = ("head", "reference", "skip_positive")
    candidates = [k for k in receipts if k not in (*standard, *diagnostic)]
    by_score = sorted(candidates, key=lambda k: -receipts[k]["projected_public_cost_formula"])
    by_f1 = sorted(candidates, key=lambda k: (-receipts[k]["scores"]["mean"]["f1"],
                                             receipts[k]["analytic_gflops"]))
    shortlisted = list(dict.fromkeys((*standard, by_score[0], by_f1[0])))
    write_json(HERE / "selection.json", dict(thresholds=THRESHOLDS, calibration_rows=int(cal.sum()),
                                            selection_rows=int((~cal).sum()), shortlisted=shortlisted,
                                            diagnostic=diagnostic, candidates=receipts))
    with (HERE / "holdout_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("policy", "mean_F1", "vulnerability_F1", "MAT_F1", "extra_windows",
                         "estimated_GFLOPs", "projected_formula"))
        for k, r in sorted(receipts.items(), key=lambda row: -row[1]["projected_public_cost_formula"]):
            writer.writerow((k, *(r["scores"][label]["f1"] for label in ("mean", "vulnerability", "MAT")),
                             r["extra_windows"], r["analytic_gflops"], r["projected_public_cost_formula"]))
    print("Frozen shortlist", ", ".join(shortlisted), flush=True)
    for name in shortlisted:
        row = receipts[name]
        print(name, row["scores"]["mean"]["f1"], row["projected_public_cost_formula"], row["extra_windows"], flush=True)


if __name__ == "__main__":
    main()
