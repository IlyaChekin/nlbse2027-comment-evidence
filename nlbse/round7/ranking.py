"""Grouped bootstrap for ranking differences and a second-score-only decision."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from external_satd import CAPS, HERE, SEED, measures, paired, write_json

COMPARISONS = (
    ("mat_plus_seen", "mat_plus_compact"),
    ("probe_plus_seen", "probe_plus_compact"),
    ("probe_plus_compact", "probe_replace_if_hidden"),
    ("probe_plus_compact", "probe_compact_only"),
    ("lexical_compact_only", "probe_compact_only"),
)


def score_ap(groups, gold, p, cluster_draws):
    order = np.argsort(-p, kind="stable")
    boundaries = np.flatnonzero(np.r_[p[order][1:] != p[order][:-1], True])
    weights = cluster_draws[:, groups[order]]
    true_positive = (weights * gold[order]).cumsum(axis=1)[:, boundaries]
    predicted = weights.cumsum(axis=1)[:, boundaries]
    positives = true_positive[:, -1:]
    precision = true_positive / np.maximum(1, predicted)
    gains = np.diff(true_positive, axis=1, prepend=0)
    return (precision * (gains / np.maximum(1, positives))).sum(axis=1)


def report(path, cap):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    y = np.asarray([int(row["gold_human_satd"]) for row in rows], dtype=bool)
    if not y.any() or y.all():
        return {"rows": len(y), "positives": int(y.sum())}
    clusters = [row["cluster"] for row in rows]
    _, groups = np.unique(clusters, return_inverse=True)
    n_groups = int(groups.max()+1)
    fields = sorted({name for pair in COMPARISONS for name in pair} - {"probe_replace_if_hidden"})
    scores = {key: np.asarray([float(x[key]) for x in rows]) for key in fields}
    fully_omitted = np.asarray([int(row["target_first_token"]) >= cap - 2 for row in rows])
    scores["probe_replace_if_hidden"] = np.where(fully_omitted, scores["probe_compact_only"],
                                                 scores["probe_plus_compact"])
    explicit = {key: float(average_precision_score(y, p)) for key, p in scores.items()}
    for key, p in scores.items():
        expected = score_ap(groups, y, p, np.ones((1, n_groups), dtype=np.int32))[0]
        if not np.isclose(expected, explicit[key], rtol=1e-8):
            raise ValueError(f"Weighted AP did not replay {key}")
    deltas = {pair: [] for pair in COMPARISONS}
    rng = np.random.default_rng(SEED)
    for _ in range(40):
        ids = rng.integers(0, n_groups, size=(50, n_groups))
        counts = np.array([np.bincount(index, minlength=n_groups) for index in ids])
        aps = {key: score_ap(groups, y, p, counts) for key, p in scores.items()}
        for a, b in COMPARISONS:
            deltas[(a, b)].extend(aps[b]-aps[a])
    return dict(
        rows=len(y), positives=int(y.sum()), clusters=n_groups, fully_omitted=int(fully_omitted.sum()),
        scores={key: measures(y, p) for key, p in scores.items()},
        pairs={f"{b}_minus_{a}": dict(
            delta_ap=explicit[b]-explicit[a],
            interval_ap=np.quantile(deltas[(a, b)], [.025, .975]).tolist(),
            f1=paired(y, scores[a], scores[b], clusters)) for a, b in COMPARISONS})


def main(folder):
    pairs = {name: {} for name in ("cppsatd", "tesoro")}
    for name, results in pairs.items():
        for cap in CAPS:
            path = folder/f"{name}_{cap}_predictions.csv"
            if path.exists():
                results[str(cap)] = report(path, cap)
    write_json(folder/"ranking.json", pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE/"external")
    main(parser.parse_args().input)
