"""Measure CPU preparation overlapped with GPU scoring, preserving batch shapes."""
import argparse
from pathlib import Path
from statistics import median
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round6"))
from common import PORTABLE, THRESHOLDS, dataset, load_checkpoint, run_policy, save, sha256, write_json
from context_policy import Policy
from tokenizer_speed import DeferredOffsets

BASE = Policy("skip_positive", "cover128", gate="skip")


def benchmark(output, repeats):
    _, _, texts, y, ids, cal = dataset(public=True)
    model, tokenizer = load_checkpoint(PORTABLE,"cuda")
    candidates = {"serial": (tokenizer, False), "overlap": (DeferredOffsets(tokenizer), True)}
    for tok, flag in candidates.values():
        run_policy(model, tok, texts[:4096], "cuda", BASE, prefetch=flag)
    measurements, predictions = {k: [] for k in candidates}, {}
    order = tuple(candidates)
    for i in range(repeats):
        for mode in order if i % 2 == 0 else order[::-1]:
            tok, flag = candidates[mode]
            torch.cuda.reset_peak_memory_stats()
            p, cost = run_policy(model, tok, texts, "cuda", BASE, prefetch=flag)
            cost["peak_cuda_mib"] = torch.cuda.max_memory_allocated()/2**20
            if mode in predictions and not np.array_equal(p, predictions[mode]):
                raise ValueError(f"Repeat changed output for {mode}")
            predictions[mode] = p
            measurements[mode].append(cost)
            print(i+1,mode,cost["wall_seconds"],cost["forward_seconds"],flush=True)
    if not np.array_equal(predictions["serial"], predictions["overlap"]):
        raise ValueError("Concurrency changed output")
    scores = {}
    for mode, p in predictions.items():
        trials = measurements[mode]
        score = {key: median(row[key] for row in trials)
                 for key in ("wall_seconds","forward_seconds","analytic_gflops","peak_cuda_mib","extra_windows")}
        scores[mode]=score
        save(output/mode,y,ids,cal,p,score,THRESHOLDS)
    paired=[a["wall_seconds"]-b["wall_seconds"]
            for a,b in zip(measurements["serial"],measurements["overlap"],strict=True)]
    write_json(output/"analysis.json",dict(device=torch.cuda.get_device_name(),
                                          checkpoint_sha256=sha256(PORTABLE/"model.safetensors"),
                                          exact_probabilities=True, trials=measurements, medians=scores,
                                          paired_wall_gain=paired))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat",type=int,default=3)
    ap.add_argument("--output",type=Path,default=HERE/"overlap")
    args=ap.parse_args()
    benchmark(args.output,args.repeat)
