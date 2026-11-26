"""Benchmark deferred materialization of fast-tokenizer character offsets."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round6"))
from common import PORTABLE, THRESHOLDS, dataset, load_checkpoint, run_policy, save, write_json
from context_policy import Policy


class Offsets:
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        return self.encodings[index].offsets


class DeferredOffsets:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def __call__(self, *args, return_offsets_mapping=False, **kwargs):
        result = self.tokenizer(*args, **kwargs)
        if return_offsets_mapping:
            result["offset_mapping"] = Offsets(result.encodings)
        return result


def benchmark(output, repeats):
    _, _, texts, y, ids, cal = dataset(public=True)
    model, ordinary = load_checkpoint(PORTABLE, "cuda")
    variants = {"ordinary": ordinary, "deferred": DeferredOffsets(ordinary)}
    policy = Policy("skip_positive", "cover128", gate="skip")
    trials, predictions = {key: [] for key in variants}, {}
    for t in variants.values():
        run_policy(model, t, texts[:2048], "cuda", policy)
    order = list(variants)
    for pair in range(repeats):
        for name in order if pair % 2 == 0 else order[::-1]:
            p, cost = run_policy(model, variants[name], texts, "cuda", policy)
            if name in predictions and not np.array_equal(p, predictions[name]):
                raise ValueError(f"Nonreproducible scores for {name}")
            predictions[name] = p
            trials[name].append(cost)
            print(pair+1, name, cost["wall_seconds"], cost["forward_seconds"], flush=True)
    if not np.array_equal(predictions["ordinary"], predictions["deferred"]):
        raise ValueError("Deferred offsets changed a model probability")
    report = dict(device=torch.cuda.get_device_name(), trials=trials, rows=len(texts), exact_probability_match=True)
    for name, p in predictions.items():
        avg = {key: median(x[key] for x in trials[name])
               for key in ("wall_seconds", "forward_seconds", "analytic_gflops", "extra_windows")}
        save(output/name, y, ids, cal, p, avg, THRESHOLDS)
        report[name] = avg
    write_json(output/"analysis.json", report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--output", type=Path, default=HERE/"tokenizer")
    args = ap.parse_args()
    benchmark(args.output, args.repeats)
