"""Verify the public prediction tables and summarize the frozen operating points."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pipeline import PUBLIC, TRAIN, load, metric, write_json
from round2_report import bootstrap_delta


def read_json(path):
    return json.loads(path.read_text())


def read_predictions(path, size):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows] != list(range(size)):
        raise ValueError(f"Invalid row order in {path}")
    return np.asarray([[int(r["pred_vulnerability"]), int(r["pred_MAT"])] for r in rows])


def main():
    with (ROOT / "round9/results.csv").open(newline="") as stream:
        previous = {r["mode"]: r for r in csv.DictReader(stream)}
    results, decisions = {}, {}
    texts, labels, fields = load(PUBLIC, audit=True)
    for mode, name in (("balanced", "combined/balanced_length"), ("quality", "fused_quality/fused")):
        label = f"round9_{mode}"
        path = ROOT / "round9" / name / "predictions.csv"
        decisions[label] = read_predictions(path, len(labels))
        results[label] = {k: float(v) for k, v in previous[mode].items() if k != "mode"}
    for mode in ("t5_small", "unixcoder9", "depth_score", "depth_quality", "fixed_depth",
                 "student_fused", "student_context", "student_lexical", "compact_exit",
                 "unix_context", "unix_lexical", "compact_lexical", "compact_lexical_deferred"):
        if mode.startswith("depth") or mode == "fixed_depth":
            report = HERE / "depth_public"
            record = read_json(report / "analysis.json")["results"][mode]
            path = report / mode / "predictions.csv"
        elif mode.startswith(("student_", "unix_")):
            report = HERE / ("student_context" if mode.startswith("student_") else "unix_context")
            record = read_json(report / "analysis.json")[mode]
            path = report / mode / "predictions.csv"
        else:
            if mode.startswith("compact_"):
                folder = {"compact_exit": "public", "compact_lexical": "lexical",
                          "compact_lexical_deferred": "lexical_deferred"}[mode]
                report = HERE / "compact_exit" / folder
            else:
                report = HERE / "runs" / mode / "public"
            record = read_json(report / "analysis.json")
            path = report / "predictions.csv"
        decisions[mode] = read_predictions(path, len(labels))
        scores, cost = record["scores"], record["cost"]
        results[mode] = dict(mean_f1=scores["mean"]["f1"], vulnerability_f1=scores["vulnerability"]["f1"],
                             mat_f1=scores["MAT"]["f1"], end_to_end_seconds=cost["wall_seconds"],
                             forward_seconds=cost["forward_seconds"], estimated_gflops=cost["analytic_gflops"],
                             local_formula=record["local_formula"])
    for mode, predictions in decisions.items():
        value = metric(labels, predictions)["mean"]["f1"]
        if abs(value - results[mode]["mean_f1"]) > 1e-12:
            raise ValueError(f"Metrics and predictions disagree for {mode}")
    with (HERE / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", *results["round9_balanced"]])
        writer.writeheader()
        writer.writerows(dict(mode=mode, **values) for mode, values in results.items())
    train_texts, _, _ = load(TRAIN)
    train_digests = {hashlib.sha256(t.encode()).digest() for t in train_texts}
    clean = np.asarray([hashlib.sha256(t.encode()).digest() not in train_digests for t in texts])
    groups = {}
    group_ids = [groups.setdefault(k, len(groups)) for k in zip(fields["Projectname"], fields["Filepath"], strict=True)]
    output = dict(public_rows=len(texts), distinct_project_file_groups=len(groups), exact_train_overlap=int((~clean).sum()),
                  clean_metric={k: metric(labels[clean], p[clean]) for k, p in decisions.items()},
                  score_winner=max(results, key=lambda k: results[k]["local_formula"]),
                  f1_winner=max(results, key=lambda k: results[k]["mean_f1"]))
    winner = output["f1_winner"]
    if winner != "round9_quality":
        output["f1_winner_interval_vs_previous_quality"] = bootstrap_delta(
            labels, decisions["round9_quality"], decisions[winner], group_ids)
    write_json(HERE / "comparison.json", output)
    print(f"Verified {len(texts):,} rows per operating point. Score {output['score_winner']}. F1 {winner}.")


if __name__ == "__main__":
    main()
