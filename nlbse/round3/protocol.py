"""Use the frozen NLBSE fit/calibration/selection partitions for round 3."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from continuation_select import best_thresholds, split
from pipeline import TRAIN, load, metric, select_indices, write_json

OUT = Path(__file__).resolve().parent / "runs"
SIX = ROOT / "runs/compact6_full/checkpoint"
TWELVE = ROOT / "runs/final_fit/checkpoint"
LABELS = ("vulnerability", "MAT")


def dataset():
    texts, labels, _ = load(TRAIN)
    fit = select_indices(True)[0]
    tune, is_cal = split()
    ids = np.array(fit+tune)
    return [texts[i] for i in ids], labels[ids], ids, len(fit), is_cal


def threshold(y, prob):
    return np.array(best_thresholds(y, prob))


def length_thresholds(y, prob, lengths):
    global_ = threshold(y, prob)
    edges = [128, 256]
    groups = np.digitize(lengths, edges, right=True)
    local = []
    for group in range(3):
        mask = groups == group
        per_label = (0.5 * global_ + 0.5 * threshold(y[mask], prob[mask])) if mask.any() else global_.copy()
        support = y[mask].sum(axis=0)
        per_label[support < 10] = global_[support < 10]
        local.append(per_label)
    return np.array(local), edges


def save(name, ids, labels, probabilities, bits, cost, config):
    run = OUT / name
    run.mkdir(parents=True, exist_ok=True)
    if not (len(ids) == len(labels) == len(probabilities) == len(bits)):
        raise ValueError(f"Prediction coverage mismatch in {name}")
    with (run / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("row", *LABELS, "p_vulnerability", "p_MAT"))
        writer.writerows((int(i), *p.astype(int), *v) for i,p,v in zip(ids, bits, probabilities, strict=True))
    output = dict(scores=metric(labels, bits), cost=cost, config=config)
    write_json(run / "metrics.json", output)
    print(name, json.dumps(output), flush=True)
    return output
