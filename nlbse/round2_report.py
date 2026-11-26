"""Summarize the frozen full public comparison and paired uncertainty."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import RobertaTokenizerFast

from pipeline import ROOT, TRAIN, PUBLIC, load, metric, write_json
from score_runtime import competition_formula

RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
CONTROLS = {
    "organizer_original": "official_public",
    "organizer_packed": "official_packed_public",
    "own12_original": "final_public",
    "own12_packed": "own12_packed_public",
    "own6_packed": "own6_packed_public",
}
LABELS = ("vulnerability", "MAT")


def read_pred(path, size):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if [int(r["row"]) for r in rows] != list(range(size)):
        raise ValueError(f"Unexpected row coverage or order in {path}")
    return np.array([[int(r[k]) for k in LABELS] for r in rows], dtype=np.int8)


def f1_from_counts(counts):
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    return 2 * tp / np.maximum(1, 2 * tp + fp + fn)


def bootstrap_delta(y, a, b, group_ids):
    n_groups = max(group_ids) + 1
    outcomes = np.array([[(p == 1) & (y == 1), (p == 1) & (y == 0), (p == 0) & (y == 1)]
                         for p in (a, b)]).transpose(2, 0, 3, 1)
    values = np.zeros((n_groups, 2, 2, 3), dtype=np.int32)
    np.add.at(values, group_ids, outcomes)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(40):
        pick = rng.integers(0, n_groups, size=(50, n_groups), dtype=np.int32)
        f1 = f1_from_counts(values[pick].sum(axis=1))
        delta = f1[:, 1, :] - f1[:, 0, :]
        draws.append(np.column_stack((delta, delta.mean(axis=1))))
    bounds = np.quantile(np.vstack(draws), [0.025, 0.975], axis=0).T.tolist()
    return dict(zip((*LABELS, "mean"), bounds))


def main():
    texts, gold, fields = load(PUBLIC, audit=True)
    estimates, predictions = {}, {}
    for label, run in CONTROLS.items():
        record = json.loads((RUNS / run / "metrics.json").read_text())
        predictions[label] = read_pred(RUNS / run / "predictions.csv", len(gold))
        scores = metric(gold, predictions[label])
        for key in scores:
            if abs(scores[key]["f1"] - record["scores"][key]["f1"]) > 1e-10:
                raise ValueError(f"Stored score differs from predictions for {label}")
        c = record["cost"]
        estimates[label] = dict(name=label, **{f"{label}_{m}": value for label, measures in scores.items()
                                             for m, value in measures.items()},
                                **c, public_local_formula=competition_formula(
                                    scores["mean"]["f1"], c["forward_seconds"], c["profiler_counted_gflops"]))
    with (REPORTS / "round2_public.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(next(iter(estimates.values()))))
        writer.writeheader()
        writer.writerows(estimates.values())

    groups = {}
    group_ids = [groups.setdefault(key, len(groups)) for key in zip(fields["Projectname"], fields["Filepath"])]
    choice = json.loads((REPORTS / "round2_selection.json").read_text())["winner"] + "_packed"
    comparisons = (("organizer_packed", choice), ("own12_packed", "own6_packed"))
    stats = {f"{b}_minus_{a}": bootstrap_delta(gold, predictions[a], predictions[b], group_ids)
             for a, b in dict.fromkeys(comparisons)}
    write_json(REPORTS / "round2_paired_intervals.json",
               dict(draws=2000, clusters=len(groups), unit="(Projectname, Filepath)",
                    percentile_intervals_95=stats))

    train_texts, _, _ = load(TRAIN)
    overlap = {hashlib.sha256(t.encode()).digest() for t in train_texts}
    clean = np.array([hashlib.sha256(t.encode()).digest() not in overlap for t in texts])
    tokenizer = RobertaTokenizerFast.from_pretrained(ROOT / "models/official", local_files_only=True)
    lengths = []
    for start in range(0, len(texts), 1024):
        input_ids = tokenizer(texts[start:start+1024], truncation=True, max_length=512,
                              return_attention_mask=False)["input_ids"]
        lengths.extend(map(len, input_ids))
    lengths = np.array(lengths)
    strata = {
        "no_exact_train_function": clean,
        "up_to_128_tokens": lengths <= 128,
        "129_to_256_tokens": (lengths > 128) & (lengths <= 256),
        "257_to_384_tokens": (lengths > 256) & (lengths <= 384),
        "385_to_512_tokens": lengths > 384,
    }
    write_json(REPORTS / "round2_public_sensitivity.json",
               {k: dict(rows=int(mask.sum()), positive=gold[mask].sum(axis=0).tolist(),
                        scores={label: metric(gold[mask], p[mask]) for label, p in predictions.items()})
                for k, mask in strata.items() if mask.any()})
    print(f"Verified {len(gold)} public predictions for {len(CONTROLS)} runs.")


if __name__ == "__main__":
    main()
