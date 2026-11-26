"""Export, compare and run the six-block CodeT5+ classifier with frozen thresholds."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_model as load_safetensors, save_model as save_safetensors
from transformers import AutoTokenizer, T5Config, T5EncoderModel

from neural import T5Joint
from protocol import HERE, ROOT, dataset, save, write_json
from score_runtime import competition_formula

sys.path.insert(0, str(ROOT / "round4"))
from compare import dense_gflops
from windows import BLOCK, chunks, encode, pack

FULL = HERE / "runs/full/codet5p"
PORTABLE = FULL / "portable"


def model_from_config(path):
    return T5Joint(T5EncoderModel(T5Config.from_pretrained(path, local_files_only=True, num_layers=6)))


def export(path):
    base = ROOT / "models/codet5p"
    model = model_from_config(base)
    state = torch.load(FULL / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(path)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(path)
    pending = path / "model.part.safetensors"
    save_safetensors(model.half(), pending)
    pending.replace(path / "model.safetensors")


def load(path, device):
    model = model_from_config(path)
    load_safetensors(model, path / "model.safetensors")
    return model.eval().to(device), AutoTokenizer.from_pretrained(path, local_files_only=True)


@torch.inference_mode()
def probabilities(model, tokenizer, texts, device, mode, *, repetitions=1, profile_reps=0):
    views = ("head", "comment128") if mode == "comment" else ("head",)
    predictions = np.empty((len(texts), 2), dtype=np.float32)
    config = model.encoder.config
    dims = dict(hidden_size=config.d_model, intermediate_size=config.d_ff, layers=config.num_layers, head_inner=0)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    seconds, forward = [], []
    for _ in range(repetitions):
        gflops, profiled_gflops, elapsed, calls = 0., 0., 0., 0
        t = time.perf_counter()
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            for start in range(0, len(texts), BLOCK):
                inputs = encode(texts[start:start+BLOCK], tokenizer, views)
                p = predictions[start:start+len(inputs.lengths)]
                for view in views:
                    is_comment = view == "comment128"
                    rows = np.flatnonzero(inputs.missing_comment) if is_comment else range(len(inputs.lengths))
                    if is_comment:
                        calls += len(rows)
                    for group in chunks(inputs.view[view], rows):
                        values = pack(inputs.view[view], group, inputs.pad, device)
                        n, length = values[0].shape
                        gflops += dense_gflops([(n, length)], dims)
                        begin = time.perf_counter()
                        if profile_reps:
                            with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
                                for _ in range(profile_reps):
                                    logits = model(*values)
                                if device == "cuda":
                                    torch.cuda.synchronize()
                            profiled_gflops += sum(event.flops for event in profiler.key_averages()) / 1e9 / profile_reps
                        else:
                            logits = model(*values)
                        current = logits.float().sigmoid().cpu().numpy()
                        elapsed += (time.perf_counter()-begin)/max(1, profile_reps)
                        if is_comment:
                            p[group, 1] = np.maximum(p[group, 1], current[:, 1])
                        else:
                            p[group] = current
        seconds.append(time.perf_counter()-t)
        forward.append(elapsed)
    if not np.isfinite(predictions).all():
        raise ValueError("Encountered non-finite CodeT5+ probabilities")
    return predictions, dict(rows=len(texts), runtime_seconds=seconds, forward_seconds=forward,
                             forward_seconds_median=median(forward), analytic_dense_gflops=gflops,
                             extra_windows=calls, profiler_counted_gflops=profiled_gflops if profile_reps else None,
                             profile_repetitions=profile_reps,
                             platform=torch.cuda.get_device_name() if device == "cuda" else "CPU")


def compare(path, device, repetitions, public, profile_reps, root, limit):
    model, tokenizer = load(path, device)
    _, _, texts, labels, ids, calibration = dataset(full=True, public=public)
    modes = ("head", "comment")
    if public:
        frozen = json.loads((path / "frozen.json").read_text())
    else:
        frozen = {}
    if limit is not None:
        texts, labels, ids, calibration = texts[:limit], labels[:limit], ids[:limit], calibration[:limit]
    scores = {}
    for mode in modes:
        p, cost = probabilities(model, tokenizer, texts, device, mode, repetitions=repetitions,
                                profile_reps=profile_reps)
        output = root / mode / ("public" if public else "holdout")
        save(output, labels, ids, calibration, p, cost, frozen.get("thresholds", {}).get(mode))
        receipt = json.loads((output / "metrics.json").read_text())
        ratio = 30655 / len(texts)
        if profile_reps:
            runtime, compute = median(cost["forward_seconds"]), cost["profiler_counted_gflops"]
        else:
            runtime, compute = median(cost["runtime_seconds"]), cost["analytic_dense_gflops"]
        scores[mode] = competition_formula(receipt["scores"]["mean"]["f1"], runtime*ratio, compute*ratio)
    if limit is not None:
        return
    if public:
        write_json(root / "public_local_formula.json", dict(scores=scores, profiled=bool(profile_reps)))
    else:
        threshold = {mode: json.loads((root / mode / "holdout/metrics.json").read_text())["thresholds"]
                     for mode in modes}
        selection = dict(mode=max(scores, key=scores.get), thresholds=threshold,
                         projected_public_local_formula=scores)
        write_json(root / "selected.json", selection)
        write_json(path / "frozen.json", selection)
    print("CodeT5+ local formula", scores, flush=True)


def predict(path, device, input_path, output_path):
    model, tokenizer = load(path, device)
    frozen = json.loads((path / "frozen.json").read_text())
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function column")
        texts = [row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Expected one or more function strings")
    p, _ = probabilities(model, tokenizer, texts, device, frozen["mode"])
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("vulnerability", "MAT"))
        writer.writerows((p > frozen["thresholds"][frozen["mode"]]).astype(np.int8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PORTABLE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("export")
    evaluate = modes.add_parser("compare")
    evaluate.add_argument("--public", action="store_true")
    evaluate.add_argument("--repetitions", type=int, default=1)
    evaluate.add_argument("--profile-reps", type=int, default=0)
    evaluate.add_argument("--out", type=Path, default=FULL/"portable_results")
    evaluate.add_argument("--limit", type=int, help="Use a fixed public prefix for a technical smoke check.")
    infer = modes.add_parser("predict")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "export":
        export(args.checkpoint)
    elif args.mode == "compare":
        if args.repetitions < 1 or args.profile_reps < 0:
            parser.error("Repetitions must be positive and profile-reps nonnegative")
        if args.limit is not None and (not args.public or args.limit < 1 or args.out == FULL/"portable_results"):
            parser.error("A smoke prefix requires --public, a positive limit and a separate --out")
        compare(args.checkpoint, args.device, args.repetitions, args.public, args.profile_reps, args.out, args.limit)
    else:
        predict(args.checkpoint, args.device, args.input, args.output)


if __name__ == "__main__":
    main()
