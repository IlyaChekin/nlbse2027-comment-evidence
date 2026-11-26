"""Benchmark frozen CodeT5+ context policies or predict from a CSV."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch

from cache_context import HERE, PORTABLE, dataset, forward_rows, load, write_json
from context_policy import THRESHOLDS, Policy, combine, first_rows, jobs, view_names
from context_views import prepare
from protocol import save
from score_runtime import competition_formula
from windows import BLOCK


def ordered_batches(starts, prepare_at, prefetch):
    starts = iter(starts)
    if not prefetch:
        for start in starts:
            yield start, prepare_at(start)
        return

    start = next(starts, None)
    if start is None:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(prepare_at, start)
        for next_start in starts:
            ready = pending.result()
            pending = executor.submit(prepare_at, next_start)
            yield start, ready
            start = next_start
        yield start, pending.result()


def prepared_batches(texts, tokenizer, views, metadata, head_limit, prefetch):
    def prepare_at(start):
        return prepare(texts[start:start+BLOCK], tokenizer, views,
                       metadata=metadata, head_limit=head_limit)

    return ordered_batches(range(0, len(texts), BLOCK), prepare_at, prefetch)


def run_policy(model, tokenizer, texts, device, policy, profile_reps=0, *, prefetch=False):
    started = time.perf_counter()
    predictions = np.empty((len(texts), 2), dtype=np.float32)
    stat = dict(forward_seconds=0., analytic_gflops=0.,
                profiler_counted_gflops=0. if profile_reps else None,
                rows=len(texts), extra_windows=0, replaced_first_windows=0, selection_by_view={})
    views = view_names(policy)
    metadata = bool(policy.comment or policy.code or (policy.first and policy.first_scope == "missing"))
    for start, prepared in prepared_batches(texts, tokenizer, views, metadata, policy.head_limit, prefetch):
        n = len(prepared.lengths)
        context = dict(lengths=prepared.lengths, omitted=prepared.omitted, spans=prepared.groups,
                       crossing=prepared.crossing, cue=prepared.cue)
        first = prepared.views["head"]
        if policy.first:
            mask = first_rows(policy, context)
            first = [prepared.views[policy.first][i] if mask[i] else row for i, row in enumerate(first)]
            if any(not sequence for sequence in first):
                raise ValueError(f"Policy chose an empty first window in {policy.name}")
            stat["replaced_first_windows"] += int(mask.sum())
        base = forward_rows(model, first, np.arange(n), prepared.pad, device, stat, profile_reps)
        out = base.copy()
        for name, role, mask in jobs(policy, base, context):
            rows = np.flatnonzero(mask)
            if len(rows):
                sequences = prepared.views[name]
                if any(not sequences[i] for i in rows):
                    raise ValueError(f"Policy chose an empty {name} in {policy.name}")
                stat["extra_windows"] += len(rows)
                signature = f"{role}/{name}"
                stat["selection_by_view"][signature] = stat["selection_by_view"].get(signature, 0) + len(rows)
                second = forward_rows(model, sequences, rows, prepared.pad, device, stat, profile_reps)
                combine(policy, base, second, mask, role, out)
        predictions[start:start+n] = out
    stat["wall_seconds"] = time.perf_counter()-started
    return predictions, stat


def benchmark(out, freeze, selected_names, repeat, profile_reps, limit):
    _, _, texts, y, ids, calibration = dataset(public=True)
    if limit is not None:
        texts, y, ids, calibration = texts[:limit], y[:limit], ids[:limit], calibration[:limit]
    thresholds = freeze["thresholds"]
    modes = {name: Policy(**freeze["candidates"][name]["config"]) for name in selected_names}
    model, tokenizer = load(PORTABLE, "cuda")
    names = list(modes)
    samples = {name: [] for name in names}
    probabilities = {}
    if not profile_reps:
        run_policy(model, tokenizer, texts[:256], "cuda", modes[names[0]])
    for cycle in range(repeat):
        order = names[cycle % len(names):] + names[:cycle % len(names)]
        for name in order:
            p, timing = run_policy(model, tokenizer, texts, "cuda", modes[name], profile_reps)
            if not np.isfinite(p).all():
                raise ValueError("Prediction coverage or finiteness failed")
            if name in probabilities and not np.array_equal(probabilities[name] > thresholds, p > thresholds):
                raise ValueError(f"Repeat changed binary decisions for {name}")
            probabilities[name] = p
            samples[name].append(timing)
            print("timing", cycle+1, name, round(timing["forward_seconds"], 3), round(timing["wall_seconds"], 3),
                  "windows", timing["extra_windows"], flush=True)
    for name in modes:
        rows = samples[name]
        cost = dict(rows[0])
        for key in ("forward_seconds", "wall_seconds"):
            cost[key+"_repeats"] = [result[key] for result in rows]
            cost[key] = median(cost[key+"_repeats"])
        if any(r["extra_windows"] != cost["extra_windows"] or r["analytic_gflops"] != cost["analytic_gflops"] for r in rows):
            raise ValueError("Repeated execution changed routing or batch shapes")
        cost.update(apparatus=torch.cuda.get_device_name(), profile_repetitions=profile_reps,
                    repetitions=repeat, mode=name)
        save(out / name, y, ids, calibration, probabilities[name], cost, thresholds)
        receipt = json.loads((out / name / "metrics.json").read_text())
        compute = cost["profiler_counted_gflops"] if profile_reps else cost["analytic_gflops"]
        if limit is None:
            receipt["local_formula"] = competition_formula(receipt["scores"]["mean"]["f1"],
                                                           cost["forward_seconds"], compute)
            write_json(out / name / "metrics.json", receipt)


def inference(mode, config, model_path, device, source, target):
    freeze = json.loads(config.read_text())
    if "config" in freeze:
        if mode not in (None, freeze["config"]["name"]):
            raise ValueError("The exported configuration describes a different mode")
        policy = Policy(**freeze["config"])
    else:
        if mode is None:
            raise ValueError("Supply --mode with the research selection file")
        policy = Policy(**freeze["candidates"][mode]["config"])
    threshold = freeze["thresholds"]
    if tuple(threshold) != THRESHOLDS:
        raise ValueError("The exported configuration has different thresholds from its routing policy")
    model, tokenizer = load(model_path, device)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function column")
        texts = [row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Expected function text in every row")
    p, _ = run_policy(model, tokenizer, texts, device, policy)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("vulnerability", "MAT"))
        writer.writerows((p > threshold).astype(np.int8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="stage", required=True)
    evaluate = modes.add_parser("public")
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--selection", type=Path, default=HERE/"selection.json")
    evaluate.add_argument("--repetitions", type=int, default=3)
    evaluate.add_argument("--profile-reps", type=int, default=0)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--mode", nargs="+")
    evaluate.add_argument("--diagnostic", action="store_true")
    cli = modes.add_parser("predict")
    cli.add_argument("--mode")
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cli.add_argument("--checkpoint", type=Path, default=PORTABLE)
    cli.add_argument("--config", type=Path, default=HERE/"selected_inference.json")
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "public":
        freeze = json.loads(args.selection.read_text())
        names = args.mode or freeze["diagnostic" if args.diagnostic else "shortlisted"]
        if args.repetitions < 1 or args.profile_reps < 0 or (args.limit is not None and args.limit < 1):
            parser.error("Invalid repetition count or row limit")
        if set(names) - set(freeze["candidates"]):
            parser.error("A requested policy is unavailable")
        benchmark(args.out, freeze, names, args.repetitions, args.profile_reps, args.limit)
    else:
        inference(args.mode, args.config, args.checkpoint, args.device, args.input, args.output)


if __name__ == "__main__":
    main()
