"""Verify the archived decisions and aggregate the project and public results."""

import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((HERE / name).read_text())


def read_csv(name):
    with (HERE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def counts(gold, pred):
    pairs = Counter(zip(gold, pred, strict=True))
    return {key: pairs[pair] for key, pair in zip(
        ("tp", "fp", "fn", "tn"), ((1, 1), (0, 1), (1, 0), (0, 0)), strict=True
    )}


def verify(rows, target, report, original, candidate):
    label = "label" if "label" in rows[0] else "y"
    gold = [int(row[label]) for row in rows]
    before = [int(row[original]) for row in rows]
    after = [int(row[candidate]) for row in rows]
    if counts(gold, before) != report["baseline"] or counts(gold, after) != report["candidate"]:
        raise ValueError(f"Confusion counts disagree with archived predictions for {candidate}")
    recovered = sum(y and not b and a and t for y, b, a, t in zip(gold, before, after, target, strict=True))
    if recovered != report["repaired_in_target"]:
        raise ValueError(f"Target repairs disagree with archived predictions for {candidate}")


def main():
    study = read_json("study/analysis.json")
    redis = read_json("redistribute/analysis.json")
    activation = read_json("activation/analysis.json")
    reweight = read_json("reweight/analysis.json")
    cohort = read_csv("study/compact_low_bindings.csv")
    ids = [row["row"] for row in cohort]
    target = [int(row["joint_low"]) for row in cohort]
    positives = [row for row in cohort if int(row["joint_low"]) and int(row["label"])]
    if len(ids) != study["rows"] or sum(target) != study["target_rows"] or len(positives) != study["target_misses"]:
        raise ValueError("Target cohort differs from the recorded test split")

    records = []

    def add(name, path, exact, frozen, short, before_exact="base_exact_fp", after_exact="pred_exact_fp",
            after_frozen="pred_dev_cut"):
        rows = read_csv(path)
        if [row["row"] for row in rows] != ids:
            raise ValueError(f"Out-of-order test rows in {path}")
        verify(rows, target, exact, before_exact, after_exact)
        verify(rows, target, frozen, "base_pred", after_frozen)
        if exact["delta_fp"]:
            raise ValueError(f"Unequal false-positive budget in {path}")
        records.append({"method": name, "equal_fp": exact["candidate"]["fp"],
                        "reference_tp": exact["baseline"]["tp"], "candidate_tp": exact["candidate"]["tp"],
                        "delta_tp": exact["delta_tp"], "target_recovered": exact["repaired_in_target"],
                        "equal_fp_f1": exact["candidate_f1"],
                        "short_equal_fp_delta_tp": short["delta_tp"] if short else "",
                        "frozen_delta_tp": frozen["delta_tp"], "frozen_delta_fp": frozen["delta_fp"],
                        "frozen_target_recovered": frozen["repaired_in_target"]})

    for name, result in study["results"].items():
        add(name, f"study/{name.replace('/', '_')}.csv", result["test_fp_budget"],
            result["frozen_dev"], result["equal_budget_short"])
    for name, result in redis["results"].items():
        add(f"redistribute/{name}", f"redistribute/{name}.csv", result["exact_fp"],
            result["frozen"], result["exact_fp_short"])
    add("activation", "activation/predictions.csv", activation["equal"], activation["frozen"], None,
        after_exact="exact_fp", after_frozen="frozen")
    for name, result in reweight.items():
        add(f"reweight/{name}", f"reweight/{name}/predictions.csv", result["equal_fp"],
            result["frozen"], result["short_equal_fp"])

    with (HERE / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)

    public = []
    for ablation in ("official", "fusion"):
        for name, result in read_json(f"{ablation}/analysis.json").items():
            tally = result["comparison"]["candidate"] if ablation == "official" else result["counts"]
            public.append({"ablation": ablation, "method": name, "tp": tally["tp"], "fp": tally["fp"],
                           "mean_f1": result["metrics"]["mean"]["f1"], "score": result["score"],
                           "wall_seconds": result["cost"]["wall_seconds"],
                           "analytics_gflops": result["cost"]["analytic_gflops"]})
    with (HERE / "public_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=public[0])
        writer.writeheader()
        writer.writerows(public)

    print(f"Verified {len(records)} comparisons. Target positives by project:",
          dict(Counter(row["project"] for row in positives)))


if __name__ == "__main__":
    main()
