"""Predict both NLBSE labels with one six-block model and an optional comment window."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from windows import BLOCK, ROOT, SIX, chunks, encode, forward, load_model, pack, tokenizer_at

CONFIG = ROOT / "round4/entry_config.json"


def read_config(path):
    config = json.loads(path.read_text())
    thresholds = np.asarray(config["thresholds"], dtype=float)
    if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or not ((0 <= thresholds) & (thresholds <= 1)).all():
        raise ValueError("Expected two thresholds in [0, 1]")
    return thresholds


@torch.inference_mode()
def predict(texts, model, tokenizer, device, *, extra=True, profile_reps=0, profile_cuda=False):
    output = np.empty((len(texts), 2), dtype=np.float32)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda" and profile_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    windows = ("head", "comment128") if extra else ("head",)
    stats = dict(rows=len(texts), extra_windows=0, forward_seconds=0., padded_tokens=0,
                 profiler_counted_gflops=0. if profile_reps else None, profile_repetitions=profile_reps)
    started = time.perf_counter()
    for begin in range(0, len(texts), BLOCK):
        data = encode(texts[begin:begin+BLOCK], tokenizer, windows)
        current = output[begin:begin+len(data.lengths)]
        for window in windows:
            is_comment = window == "comment128"
            rows = np.flatnonzero(data.missing_comment) if is_comment else range(len(data.lengths))
            stats["extra_windows"] += int(len(rows)) if is_comment else 0
            sequences = data.view[window]
            for group in chunks(sequences, rows):
                ids, mask = pack(sequences, group, data.pad, device)
                stats["padded_tokens"] += ids.numel()
                if device == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    if profile_reps:
                        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
                            for _ in range(profile_reps):
                                logits = model(input_ids=ids, attention_mask=mask).logits
                            if device == "cuda":
                                torch.cuda.synchronize()
                        stats["profiler_counted_gflops"] += (
                            sum(event.flops for event in profiler.key_averages()) / 1e9 / profile_reps)
                        values = logits.float().sigmoid().cpu().numpy()
                    else:
                        values = forward(model, (ids, mask)).cpu().numpy()
                stats["forward_seconds"] += (time.perf_counter()-now) / max(1, profile_reps)
                if is_comment:
                    current[group, 1] = np.maximum(current[group, 1], values[:, 1])
                else:
                    current[group] = values
    stats["wall_seconds"] = time.perf_counter()-started
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SIX)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    modes = parser.add_subparsers(dest="mode", required=True)
    infer = modes.add_parser("predict", help="Predict labels from a CSV with a Function column.")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    evaluate = modes.add_parser("evaluate", help="Evaluate on the official public split.")
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--head-only", action="store_true")
    evaluate.add_argument("--profile-reps", type=int, default=0)
    evaluate.add_argument("--profile-cuda", action="store_true")
    evaluate.add_argument("--limit", type=int)
    args = parser.parse_args()
    threshold = read_config(args.config)
    if args.mode == "predict":
        with args.input.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if "Function" not in (reader.fieldnames or ()):
                parser.error("Expected a Function column")
            texts = [row["Function"] for row in reader]
        if not texts or any(text is None for text in texts):
            parser.error("Expected one or more function strings")
    else:
        from pipeline import PUBLIC, load
        if args.profile_reps < 0 or (args.limit is not None and args.limit < 1):
            parser.error("Expected a nonnegative repetition count and a positive limit")
        texts, labels, _ = load(PUBLIC)
        texts, labels = texts[:args.limit], labels[:args.limit]
    tokenizer = tokenizer_at(args.checkpoint)
    model = load_model(args.checkpoint, args.device)
    if args.mode == "predict":
        p, _ = predict(texts, model, tokenizer, args.device)
        with args.output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("vulnerability", "MAT"))
            writer.writerows((p > threshold).astype(np.int8))
    else:
        from pipeline import save_scores
        from score_runtime import competition_formula
        p, cost = predict(texts, model, tokenizer, args.device, extra=not args.head_only,
                          profile_reps=args.profile_reps, profile_cuda=args.profile_cuda)
        cost["apparatus"] = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
        args.out.mkdir(parents=True, exist_ok=True)
        save_scores(args.out, labels, range(len(labels)), p, cost, threshold.tolist())
        if args.profile_reps:
            receipt = json.loads((args.out / "metrics.json").read_text())
            formula = competition_formula(receipt["scores"]["mean"]["f1"], cost["forward_seconds"],
                                          cost["profiler_counted_gflops"])
            print("Formula on these local rows and measurements", formula)


if __name__ == "__main__":
    main()
