"""Test whether held-out SATD performance is limited by the MAT decision threshold."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve

from external_satd import CAPS, HERE, measures, paired

sys.path.insert(0, str(HERE.parent))
from pipeline import write_json

VARIANTS = ("mat_head", "mat_plus_seen", "mat_plus_full", "mat_plus_compact")


def load(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.array([int(row["gold_human_satd"]) for row in rows], dtype=bool),
        np.array([row["group"] for row in rows]),
        [row["cluster"] for row in rows],
        np.array([[float(row[variant]) for variant in VARIANTS] for row in rows]),
    )


def threshold(y, probabilities):
    p, r, t = precision_recall_curve(y, probabilities)
    f1 = 2*p[:-1]*r[:-1] / np.maximum(p[:-1]+r[:-1], np.finfo(float).tiny)
    return float(t[np.flatnonzero(f1 == max(f1))[-1]])


def evaluate(folder, source, cap):
    cpp = folder / f"cppsatd_{cap}_predictions.csv"
    target = folder / f"{source}_{cap}_predictions.csv"
    if not cpp.exists() or not target.exists():
        return None
    cpp_y, cpp_groups, _, cpp_p = load(cpp)
    y, groups, clusters, p = load(target)
    binary = np.empty_like(p, dtype=bool)
    cutoffs = {}
    for group in np.unique(groups) if source == "cppsatd" else ["all"]:
        train = cpp_groups != group if source == "cppsatd" else np.ones(len(cpp_y), dtype=bool)
        test = groups == group if source == "cppsatd" else np.ones(len(y), dtype=bool)
        cutoffs[group] = [threshold(cpp_y[train], cpp_p[train, column]) for column in range(len(VARIANTS))]
        binary[test] = p[test] >= cutoffs[group]
    return dict(thresholds=cutoffs,
                scores={k: measures(y, binary[:, col]) for col,k in enumerate(VARIANTS)},
                compact_minus_seen=paired(y, binary[:, 1], binary[:, 3], clusters),
                full_minus_head=paired(y, binary[:, 0], binary[:, 2], clusters))


def main(folder):
    report = dict(
        protocol="Threshold per variant selected for F1 on other CppSATD projects, or all CppSATD for Tesoro",
        results={name: {str(cap): evaluate(folder, name, cap) for cap in CAPS}
                 for name in ("cppsatd", "tesoro")})
    for results in report["results"].values():
        for row in results.values():
            if row:
                for measures_by_variant in row["scores"].values():
                    measures_by_variant.pop("average_precision", None)
    write_json(folder/"calibration.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE/"external")
    main(parser.parse_args().input)
