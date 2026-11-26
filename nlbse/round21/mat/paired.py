"""Run a paired full public measurement including actual MAT recovery CPU time."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R20 = ROOT / "round20"
sys.path.insert(0, str(R20))
import factor as f
from marker import recover
from pipeline import PUBLIC, load


def predictions(path, columns):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["row"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"Unexpected row order in {path}")
    return np.asarray([[int(row[c]) for c in columns] for row in rows], dtype=bool)


def main():
    archive = R20 / "factor_numeric/official"
    info = f.json_at(archive / "policy.json")
    name = info["selected"]
    recipe = info["modes"][name]
    policy = f.Policy(**recipe["policy"])
    frozen = f.json_at(HERE / "annotation_policy.json")
    if frozen["selected"] != "union_lexical" or frozen["thresholds"]["union_lexical"] != .5:
        raise ValueError("Release path expects the frozen lexical union")
    program = R20 / "scan_extended.mjs"
    if f.base.checksum(program) != info["program_sha256"] or not info["numeric"]:
        raise ValueError("Frozen round 20 scanner changed")
    old = predictions(archive / f"{name}_public.csv", ("vulnerability", "mat"))
    check = predictions(HERE / "annotation_public.csv", ("base", "union_lexical_frozen"))
    if not np.array_equal(check[:, 0], old[:, 1]):
        raise ValueError("New study baseline differs from clean archived predictions")
    texts, gold, _ = load(PUBLIC)
    observed = f.json_at(f.SOURCE / "policy.json")["rule_columns"] + ["rule_195"]
    tok = f.small.tokenizer()
    model = f.pipe("official", tok)
    heads = joblib.load(archive / "heads.joblib")
    route, _ = f.pinned()
    groups = ("reference", "marker")
    trials = {g: [] for g in groups}

    def run(text, variant):
        p, auxiliary, cost = f.infer_student(model, text, route, columns=22, return_all=True)
        clock = time.perf_counter()
        values = np.column_stack((p[:, 0], auxiliary[:, 0], auxiliary[:, -2:]))
        score, no, yes, active = f.choose(values, f.estimate(heads, values), policy, recipe["gate"])
        if active.any():
            cols, raw = f.scanner([t for t, on in zip(text, active, strict=True) if on], program=program)
            if cols != observed:
                raise ValueError("Scanner returned different columns")
            score[active] = np.where(f.valid(raw), np.where(f.weak(raw), yes[active], no[active]), score[active])
        p[:, 0] = score
        marker_at = time.perf_counter()
        if variant == "marker":
            p[:, 1] = recover(text, p[:, 1] > .5)
        end = time.perf_counter()
        cost["forward_seconds"] += end-clock
        cost["wall_seconds"] += end-clock
        cost.update(scanned=int(active.sum()), marker_seconds=end-marker_at)
        return p > (recipe["threshold"], .5), cost

    for variant in groups:
        run(texts[:2048], variant)
    result = {}
    for turn in range(5):
        for variant in groups if turn % 2 == 0 else reversed(groups):
            bits, cost = run(texts, variant)
            expected = old if variant == "reference" else np.column_stack((old[:, 0], check[:, 1]))
            if not np.array_equal(bits, expected):
                raise ValueError(f"Public decisions changed unexpectedly for {variant}")
            result[variant] = bits
            trials[variant].append(cost)
            print(variant, cost, flush=True)
    summary = {}
    for variant, bits in result.items():
        costs = trials[variant]
        med = {k: median(x[k] for x in costs) for k in costs[0]}
        quality = f.metric(gold, bits)
        score = f.competition_formula(quality["mean"]["f1"], med["forward_seconds"], med["analytic_gflops"])
        path = HERE / f"profile_{variant}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("row", "label_vulnerability", "label_mat", "pred_vulnerability", "pred_MAT"))
            writer.writerows((i, *y, *b) for i, (y, b) in enumerate(zip(gold, bits.astype(int), strict=True)))
        summary[variant] = dict(score=score, quality=quality, cost=med, trials=costs,
                                predictions_sha256=f.base.checksum(path))
    summary["provenance"] = dict(rows=len(texts), public_sha256=f.base.checksum(PUBLIC),
                                 marker_sha256=f.base.checksum(HERE / "marker.py"),
                                 audit_policy_sha256=f.base.checksum(HERE / "annotation_policy.json"),
                                 baseline_predictions_sha256=f.base.checksum(archive / f"{name}_public.csv"))
    f.write_json(HERE / "profile.json", summary)
    print({k: summary[k]["score"] for k in groups}, flush=True)


if __name__ == "__main__":
    main()
