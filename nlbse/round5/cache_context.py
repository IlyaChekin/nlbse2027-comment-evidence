"""Score candidate windows with the frozen NLBSE CodeT5+ checkpoint."""
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from context_views import ROOT, VIEWS, prepare
from windows import BLOCK, chunks, encode, pack

sys.path.insert(0, str(ROOT / "portfolio"))
from protocol import dataset
from t5_entry import PORTABLE, load
from pipeline import write_json
from compare import dense_gflops

HERE = Path(__file__).resolve().parent


def dimensions(model):
    config = model.encoder.config
    return dict(hidden_size=config.d_model, intermediate_size=config.d_ff,
                layers=config.num_layers, head_inner=0)


@torch.inference_mode()
def forward_rows(model, sequences, rows, pad, device, stats=None, profile_reps=0):
    n = len(sequences)
    p = np.full((n, 2), np.nan, dtype=np.float32)
    chosen = np.zeros(n, dtype=bool)
    chosen[rows] = True
    dims = dimensions(model) if stats is not None else None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for start in range(0, n, BLOCK):
            ids = np.flatnonzero(chosen[start:start+BLOCK])+start
            for group in chunks(sequences, ids):
                inputs = pack(sequences, group, pad, device)
                t = time.perf_counter()
                if profile_reps:
                    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
                        for _ in range(profile_reps):
                            logits = model(*inputs)
                        if device == "cuda":
                            torch.cuda.synchronize()
                    stats["profiler_counted_gflops"] += sum(e.flops for e in profiler.key_averages()) / 1e9 / profile_reps
                else:
                    logits = model(*inputs)
                p[group] = logits.float().sigmoid().cpu().numpy()
                if stats is not None:
                    stats["forward_seconds"] += (time.perf_counter()-t)/max(1, profile_reps)
                    stats["analytic_gflops"] += dense_gflops([tuple(inputs[0].shape)], dims)
    if not np.isfinite(p[rows]).all():
        raise ValueError("Non-finite predictions for scored rows")
    return p


def cache():
    _, _, texts, labels, ids, cal = dataset(full=True)
    model, tokenizer = load(PORTABLE, "cuda")
    started = time.perf_counter()
    prepared = prepare(texts, tokenizer)
    reference = encode(texts, tokenizer, views=("head", "comment128"))
    if prepared.views["head"] != reference.view["head"] or not np.array_equal(prepared.missing, reference.missing_comment):
        raise ValueError("The frozen head or comment-gate implementation differs")
    for i in np.flatnonzero(prepared.missing):
        if prepared.views["cover128"][i] != reference.view["comment128"][i]:
            raise ValueError(f"The frozen comment window differs at holdout position {i}")
    del reference
    print("Encoded", len(ids), "holdout rows", int(prepared.missing.sum()), "long with omissions",
          int((prepared.lengths > 512).sum()), "long total", round(time.perf_counter()-started, 2), flush=True)
    result = dict(ids=np.array(ids), gold=labels, calibration=cal, lengths=prepared.lengths,
                  omitted=prepared.omitted, spans=prepared.groups, crossing=prepared.crossing, cue=prepared.cue)
    for view in VIEWS:
        seqs = prepared.views[view]
        lengths = np.asarray(list(map(len, seqs)), dtype=np.int32)
        rows = np.flatnonzero(lengths)
        result[view] = forward_rows(model, seqs, rows, prepared.pad, "cuda")
        result["size_"+view] = lengths
        print(view, len(rows), round(time.perf_counter()-started, 2), flush=True)
    pending = HERE / "holdout_scores.part.npz"
    np.savez_compressed(pending, **result)
    pending.replace(HERE / "holdout_scores.npz")
    with (PORTABLE / "model.safetensors").open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    write_json(HERE / "cache_config.json", dict(checkpoint=digest, dimensions=dimensions(model),
                                                rows=len(ids), calibration=int(cal.sum()),
                                                device=torch.cuda.get_device_name(),
                                                seconds=time.perf_counter()-started))


if __name__ == "__main__":
    cache()
