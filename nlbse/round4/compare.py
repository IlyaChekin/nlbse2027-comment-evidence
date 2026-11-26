"""Select compact context on grouped train rows and confirm frozen variants."""
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from statistics import median
import time

import numpy as np
import torch

from windows import (BATCH, BLOCK, ROOT, SPLICES, TAILS, cache_probabilities, chunks,
                     encode, forward, load_model, pack, tokenizer_at)
from continuation_select import best_thresholds
from pipeline import PUBLIC, load, metric, write_json
from score_runtime import competition_formula

HERE = ROOT / "round4"
REFERENCE_TIME = 118.084
REFERENCE_GFLOPS = 625868.988


@dataclass(frozen=True)
class Variant:
    view: str = "head"
    gate: str = "long"
    blend: tuple = (0.5, 0.5)
    pool: str = "mean"

    @property
    def key(self):
        if self.view == "head" or self.view in SPLICES:
            return self.view
        return f"{self.view}--{self.gate}--{self.pool}--" + "-".join(map(str, self.blend))


def routed(lengths, missing, head, v, center):
    long = lengths > 512
    if v.gate == "long":
        return long
    if v.gate == "missing":
        return long & missing
    margin = np.abs(head - center)
    if v.gate == "ambiguous":
        return long & (margin <= [0.20, 0.20]).any(axis=1)
    if v.gate == "missing_or_ambiguous":
        return long & (missing | (margin <= [0.20, 0.20]).any(axis=1))
    raise ValueError(v.gate)


def merge(head, rest, mask, v):
    probability = head.copy()
    if v.pool == "mean":
        other = rest[mask]
    else:
        other = np.maximum(head[mask], rest[mask])
    probability[mask] = head[mask] + np.asarray(v.blend) * (other - head[mask])
    return probability


def planned_shapes(lengths, mask):
    # Sorting within a bounded block gives the same padded shapes as live inference.
    for offset in range(0, len(lengths), BLOCK):
        small = lengths[offset:offset+BLOCK]
        order = np.sort(small[mask[offset:offset+BLOCK]])
        for start in range(0, len(order), BATCH):
            batch = order[start:start+BATCH]
            yield len(batch), int(batch[-1])


def dense_gflops(shapes, dimensions):
    d, f, layers = (dimensions[k] for k in ("hidden_size", "intermediate_size", "layers"))
    classifier = 2*d*(dimensions.get("head_inner", d) + dimensions.get("labels", 2))
    return sum(n * (layers * (2*l*(4*d*d+2*d*f)+4*l*l*d)+classifier)
               for n, l in shapes) / 1e9


def variants():
    yield Variant()
    yield from (Variant(view) for view in SPLICES)
    for view, gate in product(TAILS, ("long", "missing", "ambiguous", "missing_or_ambiguous")):
        yield Variant(view, gate)
        yield Variant(view, gate, (0.25, 0.75))
        yield Variant(view, gate, (0.0, 1.0), "max")


def select():
    a = np.load(HERE / "train_holdout_scores.npz")
    dims = json.loads((HERE / "dimensions.json").read_text())
    lengths, missing, y = a["lengths"], a["missing_comment"], a["labels"]
    cal = a["calibration"]
    head = a["head"]
    center = best_thresholds(y[cal], head[cal])
    limit = np.ones(len(y), dtype=bool)
    base = dense_gflops(planned_shapes(a["length_head"], limit), dims)
    results, len_by_view = {}, {v: a[f"length_{v}"] for v in ("head", *TAILS, *SPLICES)}
    predictions = {v: a[v] for v in len_by_view}
    for v in variants():
        mask = routed(lengths, missing, head, v, center)
        additional = 0
        if v.view == "head" or v.view in SPLICES:
            p = predictions[v.view]
            cost = dense_gflops(planned_shapes(len_by_view[v.view], limit), dims)
        else:
            p = merge(head, predictions[v.view], mask, v)
            additional = int(mask.sum())
            cost = base + dense_gflops(planned_shapes(len_by_view[v.view], mask), dims)
        threshold = best_thresholds(y[cal], p[cal])
        score = metric(y[~cal], p[~cal] > threshold)
        ratio = cost/base
        proxy = competition_formula(score["mean"]["f1"], REFERENCE_TIME*ratio, REFERENCE_GFLOPS*ratio)
        results[v.key] = dict(config=asdict(v), thresholds=threshold, scores=score, cost_ratio=ratio,
                              projected_local_formula_proxy=proxy, extra_windows=additional)
    control = results["head"]
    full = results["tail512--long--mean--0.5-0.5"]
    best_score = max(results.values(), key=lambda r: r["projected_local_formula_proxy"])
    best_quality = max(results.values(), key=lambda r: r["scores"]["mean"]["f1"])
    with (HERE / "holdout_table.csv").open("w", newline="") as file:
        out = csv.writer(file)
        out.writerow(("variant", "mean_F1", "vulnerability_F1", "MAT_F1", "ratio_ops", "proxy", "extra_windows"))
        for name, r in sorted(results.items(), key=lambda pair: -pair[1]["projected_local_formula_proxy"]):
            out.writerow((name, *(r["scores"][label]["f1"] for label in ("mean", "vulnerability", "MAT")),
                          r["cost_ratio"], r["projected_local_formula_proxy"], r["extra_windows"]))
    shortlist = {}
    for name, receipt in (("control", control), ("full_tail", full), ("score_candidate", best_score),
                          ("quality_candidate", best_quality)):
        v = Variant(**receipt["config"])
        if v.key not in {Variant(**r["config"]).key for r in shortlist.values()}:
            shortlist[name] = receipt
    write_json(HERE / "selection.json", {"gate_thresholds": center, "original_thresholds": [0.72, 0.50],
                                       "calibration": int(cal.sum()), "selection": int((~cal).sum()),
                                       "shortlist": shortlist, "results": results})
    for name, r in shortlist.items():
        print(name, Variant(**r["config"]).key, f"F1={r['scores']['mean']['f1']:.5f}",
              f"cost={r['cost_ratio']:.4f}", f"proxy={r['projected_local_formula_proxy']:.5f}")


def make_batches(sequence, mask, pad):
    for start in range(0, len(sequence), BLOCK):
        ids = np.flatnonzero(mask[start:start+BLOCK]) + start
        for group in chunks(sequence, ids):
            yield group, pack(sequence, group, pad, "cpu")


@torch.inference_mode()
def run(model, batches, n):
    prob = np.full((n, 2), np.nan, dtype=np.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        for group, inputs in batches:
            prob[group] = forward(model, [x.to("cuda") for x in inputs]).cpu().numpy()
    return prob


def public(repetitions):
    config = json.loads((HERE / "selection.json").read_text())
    dims = json.loads((HERE / "dimensions.json").read_text())
    shortlist = {key: Variant(**row["config"]) for key, row in config["shortlist"].items()}
    views = {"head", *(v.view for v in shortlist.values())}
    if any("missing" in v.gate for v in shortlist.values()):
        views.add("comment128")
    texts, y, _ = load(PUBLIC)
    n = len(y)
    t = time.perf_counter()
    data = encode(texts, tokenizer_at(), sorted(views))
    tokenizing_seconds = time.perf_counter()-t
    model = load_model()
    all_rows = np.ones(n, dtype=bool)
    head = cache_probabilities(model, data, "head", "cuda")
    report = {}
    payload = dict(y=y, lengths=data.lengths, missing_comment=data.missing_comment)
    for name, v in shortlist.items():
        first = v.view if v.view in SPLICES else "head"
        second = v.view if v.view in TAILS else None
        batches = list(make_batches(data.view[first], all_rows, data.pad))
        pick = routed(data.lengths, data.missing_comment, head, v, config["gate_thresholds"])
        tails = list(make_batches(data.view[second], pick, data.pad)) if second else []
        shaped = [(t[0].shape[0], t[0].shape[1]) for _, t in [*batches, *tails]]
        gflops = dense_gflops(shaped, dims)
        seconds = []
        for _ in range(repetitions):
            torch.cuda.synchronize()
            started = time.perf_counter()
            p = run(model, batches, n)
            if tails:
                # Verify the frozen routing also agrees with probabilities from this pass.
                assert np.array_equal(pick, routed(data.lengths, data.missing_comment, p, v,
                                                   config["gate_thresholds"]))
                p = merge(p, run(model, tails, n), pick, v)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter()-started)
        assert np.isfinite(p).all()
        threshold = config["shortlist"][name]["thresholds"]
        scores = metric(y, p > threshold)
        result = dict(config=asdict(v), thresholds=threshold, scores=scores,
                      forward_seconds=seconds, forward_seconds_median=median(seconds),
                      analytic_gflops=gflops, extra_windows=int(pick.sum()) if second else 0,
                      formula_using_local_time_and_analytic_gflops=competition_formula(
                          scores["mean"]["f1"], median(seconds), gflops))
        if name == "control":
            result["scores_original_thresholds"] = metric(y, p > config["original_thresholds"])
        payload[name] = p
        report[name] = result
        write_json(HERE / "public_results.json",
                   dict(rows=n, apparatus=torch.cuda.get_device_name(), nominal_repetitions=repetitions,
                        shared_tokenizing_seconds=tokenizing_seconds, variants=report))
        np.savez_compressed(HERE / "public_predictions.npz", **payload)
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "public"))
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10:
        parser.error("Choose 1–10 local timing repetitions")
    select() if args.stage == "select" else public(args.repetitions)
