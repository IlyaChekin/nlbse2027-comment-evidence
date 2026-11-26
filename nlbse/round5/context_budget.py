"""Freeze shorter first-window budgets using only labeled training holdout rows."""
from dataclasses import asdict
import json
import time

import numpy as np

from cache_context import HERE, PORTABLE, dataset, dimensions, forward_rows, load, write_json
from context_policy import THRESHOLDS, Policy
from context_views import prepare
from pipeline import metric
from score_runtime import competition_formula
from select_context import PUBLIC_ROWS, REFERENCE_PUBLIC_FORWARD, apply_cached

CAPS = (256, 384)
VIEWS = ("head", "cover64", "cover128")


def main():
    _, _, texts, y, ids, cal = dataset(full=True)
    model, tokenizer = load(PORTABLE, "cuda")
    dims = dimensions(model)
    baseline = json.loads((HERE/"selection.json").read_text())["candidates"]["reference"]["analytic_gflops"]
    receipts = {}
    for cap in CAPS:
        t = time.perf_counter()
        inputs = prepare(texts, tokenizer, VIEWS, head_limit=cap)
        data = dict(ids=ids, gold=y, calibration=cal, lengths=inputs.lengths, omitted=inputs.omitted,
                    spans=inputs.groups, cue=inputs.cue, crossing=inputs.crossing)
        for view, sequences in inputs.views.items():
            size = np.asarray(list(map(len, sequences)), dtype=np.int32)
            rows = np.flatnonzero(size)
            data["size_"+view] = size
            data[view] = forward_rows(model, sequences, rows, inputs.pad, "cuda")
        pending = HERE/f"budget{cap}_holdout.part.npz"
        np.savez_compressed(pending, **data)
        pending.replace(HERE/f"budget{cap}_holdout.npz")
        for view in VIEWS:
            policy = Policy(f"cap{cap}_{view}", comment="" if view == "head" else view,
                            gate="skip", head_limit=cap)
            p, gflops, count = apply_cached(policy, data, dims)
            scores = metric(y[~cal], p[~cal] > THRESHOLDS)
            local = competition_formula(scores["mean"]["f1"], REFERENCE_PUBLIC_FORWARD*gflops/baseline,
                                        gflops*PUBLIC_ROWS/len(y))
            receipts[policy.name] = dict(config=asdict(policy), scores=scores, analytic_gflops=gflops,
                                         extra_windows=count, projected_public_cost_formula=local,
                                         long_rows=int((inputs.lengths > cap).sum()),
                                         missing_rows=int(inputs.missing.sum()))
            print(policy.name, scores["mean"]["f1"], "extra", count, "proxy", local, flush=True)
        print("Budget", cap, "seconds", round(time.perf_counter()-t, 2), flush=True)
    candidate = max((k for k in receipts if not k.endswith("_head")),
                    key=lambda k: receipts[k]["projected_public_cost_formula"])
    matched = f"cap{receipts[candidate]['config']['head_limit']}_head"
    write_json(HERE/"budget_selection.json", dict(thresholds=THRESHOLDS, calibration_rows=int(cal.sum()),
                                                 selection_rows=int((~cal).sum()),
                                                 candidates=receipts, shortlisted=[matched, candidate],
                                                 diagnostic=[]))
    print("Frozen length-budget policies", matched, candidate, flush=True)


if __name__ == "__main__":
    main()
