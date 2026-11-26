"""Compare two prediction runs on exactly the same functions and row order."""
import argparse
import csv
import json
from pathlib import Path


def read(run):
    with (run / "predictions.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    metrics = json.loads((run / "metrics.json").read_text())
    return rows, metrics


def compare(control, candidate):
    a, ma = read(control)
    b, mb = read(candidate)
    if [r["row"] for r in a] != [r["row"] for r in b] or not a:
        raise ValueError("Prediction inputs differ or are empty.")
    before, after = ma["cost"], mb["cost"]
    fields = ("wall_seconds", "forward_seconds", "profiler_counted_gflops", "padded_tokens",
              "peak_allocated_mib", "parameters")
    return {
        "rows": len(a),
        "labels_changed": {label: sum(x[label] != y[label] for x, y in zip(a, b, strict=True))
                           for label in ("vulnerability", "MAT")},
        "maximum_absolute_probability_change": {
            label: max(abs(float(x[f"p_{label}"]) - float(y[f"p_{label}"]))
                       for x, y in zip(a, b, strict=True)) for label in ("vulnerability", "MAT")
        },
        "mean_F1": [ma["scores"]["mean"]["f1"], mb["scores"]["mean"]["f1"]],
        "cost": {k: [before[k], after[k]] for k in fields},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.candidate), indent=2))
