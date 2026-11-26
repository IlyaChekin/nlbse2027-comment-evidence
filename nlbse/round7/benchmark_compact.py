"""Time the original and compact second windows through the same public pipeline."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round6"))
from common import (PORTABLE, THRESHOLDS, dataset, forward_rows, load_checkpoint,
                    prepare, save, sha256, write_json)
from context_views import COMPACT_VIEW
from windows import BLOCK

MODES = {"full": "cover128", "compact": COMPACT_VIEW}


def stage_cost():
    return dict(forward_seconds=0., analytic_gflops=0.)


def run(model, tokenizer, texts, view):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    probabilities = np.empty((len(texts), 2), dtype=np.float32)
    first_cost, second_cost = stage_cost(), stage_cost()
    second_wall, preparation, routed = 0., 0., 0
    used_tokens = []
    for start in range(0, len(texts), BLOCK):
        begin_prep = time.perf_counter()
        inputs = prepare(texts[start:start + BLOCK], tokenizer, ("head", view))
        preparation += time.perf_counter() - begin_prep
        first = forward_rows(model, inputs.views["head"], np.arange(len(inputs.lengths)),
                             inputs.pad, "cuda", first_cost)
        selected = np.flatnonzero(inputs.missing & (first[:, 1] <= THRESHOLDS[1]))
        routed += len(selected)
        if len(selected):
            extra = inputs.views[view]
            used_tokens.extend(len(extra[i]) for i in selected)
            begin_extra = time.perf_counter()
            other = forward_rows(model, extra, selected, inputs.pad, "cuda", second_cost)
            second_wall += time.perf_counter() - begin_extra
            first[selected, 1] = np.maximum(first[selected, 1], other[selected, 1])
        probabilities[start:start + len(first)] = first
    torch.cuda.synchronize()
    return probabilities, dict(total_wall=time.perf_counter()-begin, preparations=preparation,
                               head_forward=first_cost["forward_seconds"],
                               second_forward=second_cost["forward_seconds"],
                               second_wall=second_wall, second_gflops=second_cost["analytic_gflops"],
                               total_gflops=first_cost["analytic_gflops"]+second_cost["analytic_gflops"],
                               extra_windows=routed, mean_second_tokens=float(np.mean(used_tokens)) if routed else 0.)


def benchmark(output, repeats):
    _, _, texts, gold, ids, cal = dataset(public=True)
    model, tokenizer = load_checkpoint(PORTABLE, "cuda")
    for view in MODES.values():
        run(model, tokenizer, texts[:2048], view)
    scores = {}
    trials = {name: [] for name in MODES}
    predictions = {}
    names = tuple(MODES)
    for cycle in range(repeats):
        order = names if cycle % 2 == 0 else names[::-1]
        for name in order:
            p, cost = run(model, tokenizer, texts, MODES[name])
            if name in predictions and not np.array_equal(p > THRESHOLDS, predictions[name] > THRESHOLDS):
                raise ValueError(f"Unstable binary decisions in {name}")
            predictions[name] = p
            trials[name].append(cost)
            print(cycle+1, name, json.dumps(cost), flush=True)
    for name, measurements in trials.items():
        summary = {key: median(run[key] for run in measurements) for key in measurements[0]}
        save(output / name, gold, ids, cal, predictions[name], summary, THRESHOLDS)
        scores[name] = json.loads((output / name / "metrics.json").read_text())
    differences = [(a["total_wall"]-b["total_wall"]) for a, b in zip(trials["full"], trials["compact"], strict=True)]
    original = HERE.parent / "round5/runs/public/skip_positive/predictions.csv"
    from analyze import predict_bits
    reference = predict_bits(original, ids, {"thresholds": THRESHOLDS})
    changed = {name: int(np.count_nonzero((p > THRESHOLDS) != reference)) for name, p in predictions.items()}
    if changed["full"]:
        raise ValueError("Timing baseline differs from the original release")
    write_json(output / "analysis.json", dict(
        apparatus=torch.cuda.get_device_name(), checkpoint_sha256=sha256(PORTABLE/"model.safetensors"),
        protocol="warm up both variants, alternating order per pair, same process and weights",
        decisions_changed=changed, results=scores, trials=trials, speed_gain_wall_pairs=differences))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--output", type=Path, default=HERE/"timing")
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("Use at least two timing pairs")
    benchmark(args.output, args.repeat)
