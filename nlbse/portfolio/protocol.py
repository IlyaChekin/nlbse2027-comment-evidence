"""Shared grouped pilot, calibration and evaluation for the NLBSE portfolio."""
import csv
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from continuation_select import best_thresholds, split
from pipeline import PUBLIC, TRAIN, load, metric, select_indices, write_json


def dataset(full=False, public=False):
    if public:
        eval_texts, eval_labels, _ = load(PUBLIC)
        ids, calibration = list(range(len(eval_texts))), np.zeros(len(eval_texts), dtype=bool)
        return [], np.empty((0, 2), dtype=np.int8), eval_texts, eval_labels, ids, calibration
    texts, labels, _ = load(TRAIN)
    fit, _ = select_indices(pilot=not full)
    ids, calibration = split()
    return [texts[i] for i in fit], labels[fit], [texts[i] for i in ids], labels[ids], ids, calibration


def save(out, labels, ids, calibration, probabilities, cost, thresholds=None):
    out.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite joint probabilities for every evaluation row")
    if thresholds is None:
        if not calibration.any():
            raise ValueError("A frozen threshold is required for public evaluation")
        thresholds = best_thresholds(labels[calibration], probabilities[calibration])
    predictions = (probabilities > thresholds).astype(np.int8)
    scores = metric(labels[~calibration], predictions[~calibration])
    payload = dict(thresholds=thresholds, calibration_rows=int(calibration.sum()),
                   evaluation_rows=int((~calibration).sum()), scores=scores, cost=cost)
    pending = out / "predictions.csv.part"
    with pending.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("row", "calibration", "p_vulnerability", "p_MAT", "pred_vulnerability", "pred_MAT"))
        writer.writerows((i, int(cal), *prob, *pred)
                         for i, cal, prob, pred in zip(ids, calibration, probabilities, predictions))
    pending.replace(out / "predictions.csv")
    write_json(out / "metrics.json", payload)
    print(out.name, scores, cost, flush=True)
