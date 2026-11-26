"""Compare continuation-training arms on identical inference policies."""
import argparse
import csv
import json
import time

import numpy as np

from common import (HERE, PORTABLE, THRESHOLDS, VARIANTS, competition_formula, dataset, forward_rows,
                    load_checkpoint, metric, prepare, save, sha256, write_json)
from context_policy import combine, jobs
from train_context import ARMS, OUT as TRAINED
from select_context import PUBLIC_ROWS

OUT = HERE / "training"
MODELS = {"frozen": PORTABLE, **{arm: TRAINED/arm for arm in ARMS}}


def receipt():
    return dict(forward_seconds=0., analytic_gflops=0., extra_windows=0)


def evaluate(stage):
    public = stage == "public"
    _, _, texts, y, ids, cal = dataset(full=True, public=public)
    if public:
        frozen = json.loads((OUT/"frozen.json").read_text())
        expected = {key: sha256(path/"model.safetensors") for key,path in MODELS.items()}
        if frozen["weights_sha256"] != expected:
            raise ValueError("Public checkpoint hashes differ from selection")
    _, tokenizer = load_checkpoint(PORTABLE, "cpu")
    start = time.perf_counter()
    data = {cap: prepare(texts, tokenizer, ("head", view), head_limit=cap)
            for cap,view in ((384,"cover64"), (512,"cover128"))}
    prep = time.perf_counter()-start
    results = {}
    for name, path in MODELS.items():
        model, _ = load_checkpoint(path, "cuda")
        head, costs = {}, {}
        for cap, inputs in data.items():
            costs[cap] = receipt()
            head[cap] = forward_rows(model, inputs.views["head"], np.arange(len(y)), inputs.pad,
                                     "cuda", costs[cap])
        for p in VARIANTS:
            inputs = data[p.head_limit]
            first = head[p.head_limit]
            out = first.copy()
            cost = dict(costs[p.head_limit], shared_preparation_seconds=prep, mode=p.name, rows=len(y))
            metadata = dict(lengths=inputs.lengths, omitted=inputs.omitted, spans=inputs.groups,
                            crossing=inputs.crossing, cue=inputs.cue)
            for view, role, mask in jobs(p, first, metadata):
                rows = np.flatnonzero(mask)
                tail = forward_rows(model, inputs.views[view], rows, inputs.pad, "cuda", cost)
                cost["extra_windows"] = len(rows)
                combine(p, first, tail, mask, role, out)
            target = OUT/"runs"/stage/name/p.name
            save(target, y, ids, cal, out, cost, THRESHOLDS)
            quality = metric(y[~cal], out[~cal]>THRESHOLDS)["mean"]["f1"]
            scale = PUBLIC_ROWS / len(y)
            score = competition_formula(quality, cost["forward_seconds"]*scale, cost["analytic_gflops"]*scale)
            results[f"{name}/{p.name}"] = dict(mean_F1=quality, cost=cost, local_formula=score)
        del model
    write_json(OUT/f"{stage}_summary.json", results)
    with (OUT/f"{stage}_table.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("variant","mean_F1","forward_seconds","extra_windows","analytic_GFLOPs","local_formula"))
        for name, r in results.items():
            cost = r["cost"]
            writer.writerow((name, r["mean_F1"], cost["forward_seconds"], cost["extra_windows"],
                             cost["analytic_gflops"], r["local_formula"]))
    if not public:
        names = [key for key in results if key.endswith("/comment384")]
        best = max(names, key=lambda k: results[k]["mean_F1"])
        write_json(OUT/"frozen.json", dict(thresholds=THRESHOLDS, preferred_384=best,
                                         weights_sha256={key:sha256(path/"model.safetensors") for key,path in MODELS.items()},
                                         criterion="mean F1 on grouped train holdout, same inference view"))
        print("Frozen 384-token candidate", best, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("holdout","public"))
    evaluate(parser.parse_args().split)
