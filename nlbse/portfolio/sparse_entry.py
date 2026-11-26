"""Run the compact role-aware linear NLBSE model on CPU."""
import argparse
import csv
from itertools import islice
import json
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np

from classical import predict

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "runs/full/sparse_roles"
BATCH = 1024


def predict_batches(rows, model):
    while batch := list(islice(rows, BATCH)):
        yield predict("sparse_roles", model, batch)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ARTIFACT/"model.joblib")
    parser.add_argument("--config", type=Path, default=ARTIFACT/"metrics.json")
    modes = parser.add_subparsers(dest="mode", required=True)
    infer = modes.add_parser("predict")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    check = modes.add_parser("verify")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    thresholds = json.loads(args.config.read_text())["thresholds"]
    if args.mode == "predict":
        with args.input.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if "Function" not in (reader.fieldnames or ()):
                parser.error("Expected a Function column")
            with args.output.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(("vulnerability", "MAT"))
                for prob in predict_batches((row["Function"] for row in reader), model):
                    writer.writerows((prob > thresholds).astype(np.int8))
    else:
        import sys
        sys.path.insert(0, str(HERE.parent))
        from pipeline import PUBLIC, load, metric, write_json
        texts, y, _ = load(PUBLIC)
        reference = ARTIFACT/"public/predictions.csv"
        with reference.open(newline="") as file:
            py = np.asarray([[float(row["p_vulnerability"]), float(row["p_MAT"])]
                             for row in csv.DictReader(file)])
        times = []
        for _ in range(3):
            start = time.perf_counter()
            prob = np.concatenate(list(predict_batches(iter(texts), model)))
            times.append(time.perf_counter()-start)
        bits = prob > thresholds
        if not np.array_equal(bits, py > thresholds):
            raise ValueError("The streaming entry predictions differ from the full-matrix run")
        report = dict(rows=len(y), max_probability_difference=float(np.max(np.abs(prob-py))),
                      scores=metric(y, bits), streaming_seconds=times, median_seconds=median(times),
                      batch_size=BATCH, weights_bytes=args.model.stat().st_size)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
