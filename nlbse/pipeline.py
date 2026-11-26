"""Train or evaluate one shared CodeBERT classifier for both NLBSE labels."""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw"
TRAIN = DATA / "train-00000-of-00001.parquet"
PUBLIC = DATA / "test-00000-of-00001.parquet"
SPLIT = ROOT / "data/split_manifest.csv"
MODEL = ROOT / "models/codebert"
OFFICIAL = ROOT / "models/official"


def write_json(path, data):
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def load(path, audit=False):
    fields = ["Function", "Label"] + (["Projectname", "Filepath"] if audit else [])
    columns = pq.read_table(path, columns=fields).to_pydict()
    texts = columns["Function"]
    labels = np.asarray(columns["Label"])
    if labels.shape != (len(texts), 2) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Invalid labels in {path}")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"Invalid function in {path}")
    return texts, labels.astype(np.int8), columns


def summary(labels):
    return {str(key): value for key, value in sorted(Counter(map(tuple, labels.tolist())).items())}


def prepare():
    train, y, tm = load(TRAIN, audit=True)
    public, v, vm = load(PUBLIC, audit=True)
    print(f"train={len(train)} public={len(public)}", flush=True)
    hashes = [hashlib.sha256(s.encode()).hexdigest() for s in train]
    dev_hash = {hashlib.sha256(s.encode()).hexdigest() for s in public}
    groups = defaultdict(list)
    for i, value in enumerate(hashes):
        groups[value].append(i)
    keys = sorted(groups)
    # Stratify groups by the set of joint labels seen for the same exact function.
    signatures = ["|".join(map(str, sorted(set(map(tuple, y[groups[key]].tolist()))))) for key in keys]
    counts = Counter(signatures)
    singleton = {index for index, label in enumerate(signatures) if counts[label] == 1}
    eligible = [index for index in range(len(keys)) if index not in singleton]
    training, tuned = train_test_split(eligible, test_size=0.1, random_state=42,
                                       stratify=[signatures[index] for index in eligible])
    fit = sorted(i for index in [*training, *singleton] for i in groups[keys[index]])
    tune = sorted(i for index in tuned for i in groups[keys[index]])
    label_codes = y[:, 0] * 2 + y[:, 1]
    pilot, _ = train_test_split(fit, train_size=20000, random_state=42, stratify=label_codes[fit])
    pilot_tune, _ = train_test_split(tune, train_size=3000, random_state=42, stratify=label_codes[tune])
    pilot, pilot_tune, tuning = set(pilot), set(pilot_tune), set(tune)
    with SPLIT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("row", "function_sha256", "partition", "pilot"))
        writer.writerows((i, value, "tune" if i in tuning else "fit", int(i in pilot or i in pilot_tune))
                         for i, value in enumerate(hashes))
    audit = {
        "train": summary(y), "public_validation": summary(v), "internal_fit": summary(y[fit]),
        "internal_tune": summary(y[tune]), "pilot_fit": summary(y[sorted(pilot)]),
        "pilot_tune": summary(y[sorted(pilot_tune)]),
        "duplicate_train_rows": len(train) - len(groups),
        "duplicate_public_rows": len(public) - len(dev_hash),
        "shared_function_hashes": len(set(groups) & dev_hash),
        "shared_file_keys": len(set(zip(tm["Projectname"], tm["Filepath"])) &
                                set(zip(vm["Projectname"], vm["Filepath"]))),
    }
    write_json(ROOT / "data/data_audit.json", audit)
    print(json.dumps(audit, indent=2))


def select_indices(pilot):
    with SPLIT.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return [[int(row["row"]) for row in rows if row["partition"] == part and (not pilot or row["pilot"] == "1")]
            for part in ("fit", "tune")]


def metric(gold, pred):
    tp = ((gold == 1) & (pred == 1)).sum(axis=0)
    fp = ((gold == 0) & (pred == 1)).sum(axis=0)
    fn = ((gold == 1) & (pred == 0)).sum(axis=0)
    values = np.array([tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1),
                       (2 * tp) / np.maximum(2 * tp + fp + fn, 1)])
    return {name: dict(zip(("precision", "recall", "f1"), column.tolist()))
            for name, column in zip(("vulnerability", "MAT", "mean"), [*values.T, values.mean(axis=1)])}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_inputs(tokenizer, texts, ids, length, fixed, device="cuda"):
    return tokenizer([texts[i] for i in ids], padding="max_length" if fixed else True,
                     truncation=True, max_length=length, return_tensors="pt").to(device)


def inference_batches(tokenizer, texts, ids, cfg, device):
    batch_size = cfg["batch_size"]
    window = cfg.get("bucket_window", 0)
    if not window:
        for start in range(0, len(ids), batch_size):
            positions = range(start, min(start + batch_size, len(ids)))
            yield positions, batch_inputs(tokenizer, texts, ids[start:start + batch_size],
                                          cfg["length"], cfg["fixed"], device)
        return

    for start in range(0, len(ids), window):
        chunk = ids[start:start + window]
        tokens = tokenizer([texts[i] for i in chunk], truncation=True, max_length=cfg["length"],
                           return_attention_mask=False)["input_ids"]
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        for offset in range(0, len(tokens), batch_size):
            positions = order[offset:offset + batch_size]
            batch = tokenizer.pad({"input_ids": [tokens[i] for i in positions]},
                                  return_tensors="pt").to(device)
            yield [start + i for i in positions], batch


@torch.inference_mode()
def predict(model, tokenizer, texts, ids, cfg):
    model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and cfg.get("profile_cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    probabilities = np.empty((len(ids), 2), dtype=np.float32)
    forward_s, gflops, total_tokens, completed = 0, 0, 0, 0
    started = time.perf_counter()
    for batch_number, (positions, batch) in enumerate(inference_batches(tokenizer, texts, ids, cfg, device)):
        total_tokens += batch["input_ids"].numel()
        if use_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda):
            if cfg["profile_reps"]:
                with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                    for _ in range(cfg["profile_reps"]):
                        output = model(**batch).logits
                    if use_cuda:
                        torch.cuda.synchronize()
                gflops += sum(event.flops for event in prof.key_averages()) / 1e9 / cfg["profile_reps"]
            else:
                output = model(**batch).logits
                if use_cuda:
                    torch.cuda.synchronize()
        forward_s += (time.perf_counter() - t) / max(cfg["profile_reps"], 1)
        probabilities[positions, :] = (output.float() / cfg.get("probability_temperature", 1.0)).sigmoid().cpu().numpy()
        completed += len(positions)
        if batch_number % 100 == 0:
            print(f"inference {completed}/{len(ids)}", flush=True)
    return probabilities, dict(
        wall_seconds=time.perf_counter() - started, forward_seconds=forward_s,
        profiler_counted_gflops=gflops if cfg["profile_reps"] else None,
        padded_tokens=total_tokens, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
        parameters=sum(p.numel() for p in model.parameters()), rows=len(ids),
    )


def save_scores(out, gold, ids, prob, timing, thresholds):
    pred = (prob > np.asarray(thresholds)).astype(int)
    scores = metric(gold, pred)
    with (out / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("row", "vulnerability", "MAT", "p_vulnerability", "p_MAT"))
        writer.writerows((i, *bits, *p) for i, bits, p in zip(ids, pred, prob))
    write_json(out / "metrics.json", dict(scores=scores, cost=timing, thresholds=thresholds))
    print(json.dumps(dict(scores=scores, cost=timing), indent=2), flush=True)


def tune_threshold(gold, prob):
    result = []
    for i in range(2):
        def rank(threshold):
            pred = prob[:, i] > threshold
            truth = gold[:, i]
            return (2 * np.sum(pred * truth) / max(np.sum(pred) + np.sum(truth), 1),
                    -abs(threshold - 0.5))
        result.append(max((value / 100 for value in range(20, 81, 5)), key=rank))
    return result


def train(cfg, out):
    fit, tune = select_indices(cfg["pilot"])
    texts, labels, _ = load(TRAIN)
    source = cfg["init_from"]
    tokenizer = RobertaTokenizerFast.from_pretrained(source, local_files_only=True)
    teacher_targets = None
    if cfg["teacher_from"]:
        teacher_source = cfg["teacher_from"]
        teacher_tokenizer = RobertaTokenizerFast.from_pretrained(teacher_source, local_files_only=True)
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("Teacher and student tokenizers must use the same vocabulary.")
        teacher = RobertaForSequenceClassification.from_pretrained(
            teacher_source, local_files_only=True, attn_implementation=cfg["train_attention"]
        ).cuda()
        target, cost = predict(teacher, teacher_tokenizer, texts, fit, {
            **cfg, "profile_reps": 0, "probability_temperature": cfg["temperature"],
        })
        teacher_targets = np.empty((len(texts), 2), dtype=np.float32)
        teacher_targets[fit] = target
        write_json(out / "teacher_inference.json", cost)
        del teacher
        torch.cuda.empty_cache()
    model = RobertaForSequenceClassification.from_pretrained(
        source, num_labels=2, local_files_only=True, attn_implementation=cfg["train_attention"])
    layers = cfg["retain_layers"]
    if layers:
        available = model.roberta.encoder.layer
        if layers != sorted(set(layers)) or not (0 <= layers[0] <= layers[-1] < len(available)):
            raise ValueError("Retained layer indices must be unique, increasing and within the source model.")
        model.roberta.encoder.layer = torch.nn.ModuleList(available[i] for i in layers)
        model.config.num_hidden_layers = len(layers)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01, fused=True)
    total_steps = cfg["epochs"] * ((len(fit) + cfg["batch_size"] - 1) // cfg["batch_size"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    positives = labels[fit].sum(axis=0)
    weight = np.clip((len(fit) - positives) / positives, 1, 5) if cfg["weighted"] else [1., 1.]
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device="cuda"))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    rng = random.Random(cfg["seed"])
    started = time.perf_counter()
    step, loss_sum = 0, 0.
    for epoch in range(cfg["epochs"]):
        rng.shuffle(fit)
        for start in range(0, len(fit), cfg["batch_size"]):
            ids = fit[start:start + cfg["batch_size"]]
            tokens = batch_inputs(tokenizer, texts, ids, cfg["length"], cfg["fixed"])
            gold = torch.tensor(labels[ids], dtype=torch.float32, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**tokens).logits
                loss = bce(logits, gold)
                if teacher_targets is not None:
                    t = cfg["temperature"]
                    target = torch.as_tensor(teacher_targets[ids], device="cuda")
                    soft = torch.nn.functional.binary_cross_entropy_with_logits(logits.float() / t, target)
                    amount = cfg["distill_weight"]
                    loss = (1 - amount) * loss + amount * t * t * soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += loss.item()
            if step % 100 == 0 or step == total_steps:
                print(f"epoch={epoch + 1} step={step}/{total_steps} loss={loss_sum / step:.5f} "
                      f"wall_s={time.perf_counter() - started:.2f} peak_MiB={torch.cuda.max_memory_allocated()/2**20:.0f}",
                      flush=True)
    train_timing = dict(wall_seconds=time.perf_counter() - started,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        rows=len(fit), steps=total_steps, positive_weights=np.asarray(weight).tolist())
    write_json(out / "training.json", train_timing)
    prob, cost = predict(model, tokenizer, texts, tune, {**cfg, "profile_reps": 0})
    threshold = tune_threshold(labels[tune], prob)
    write_json(out / "fixed_threshold_scores.json", metric(labels[tune], prob > 0.5))
    save_scores(out, labels[tune], tune, prob, cost, threshold)
    write_json(out / "thresholds.json", threshold)
    if not cfg["pilot"] or cfg["save_checkpoint"]:
        model.save_pretrained(out / "checkpoint")
        tokenizer.save_pretrained(out / "checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--run")
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--size", type=int, default=0, help="Deterministic smoke subset; 0 evaluates all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--profile-reps", type=int, default=0)
    parser.add_argument("--profile-cuda", action="store_true",
                        help="Record CUDA profiler activity alongside the CPU operator events.")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--internal", action="store_true", help="Evaluate only the internal train holdout.")
    parser.add_argument("--evaluation-file", type=Path, help="Use a labeled parquet supplied by the evaluator.")
    parser.add_argument("--bucket-window", type=int, default=0)
    parser.add_argument("--retain-layers", type=int, nargs="+")
    parser.add_argument("--init-from", type=Path, default=MODEL)
    parser.add_argument("--train-attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--teacher-from", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    cfg = vars(parser.parse_args())
    if cfg["mode"] == "prepare":
        prepare()
        return
    if not cfg["run"]:
        parser.error("--run is required")
    if cfg["batch_size"] < 1 or not 1 <= cfg["length"] <= 512 or cfg["profile_reps"] < 0:
        parser.error("Use a positive batch size, 1–512 tokens and a nonnegative number of profiling repetitions.")
    if cfg["bucket_window"] and (cfg["fixed"] or cfg["bucket_window"] < cfg["batch_size"]
                                 or cfg["bucket_window"] % cfg["batch_size"]):
        parser.error("Use dynamic padding and a bucket window that is a positive multiple of batch size.")
    if not 0 < cfg["distill_weight"] <= 1 or not cfg["temperature"] > 0:
        parser.error("The distillation weight must be in (0, 1] and the temperature must be positive.")
    if cfg["evaluation_file"] and (cfg["mode"] != "evaluate" or cfg["internal"]):
        parser.error("Use an external evaluation file only in evaluate mode with no internal split.")
    out = ROOT / "runs" / cfg["run"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {**cfg, "checkpoint": str(cfg["checkpoint"]),
                                    "init_from": str(cfg["init_from"]),
                                    "teacher_from": str(cfg["teacher_from"]) if cfg["teacher_from"] else None,
                                    "evaluation_file": str(cfg["evaluation_file"]) if cfg["evaluation_file"] else None,
                                    "thresholds": str(cfg["thresholds"]) if cfg["thresholds"] else None,
                                    "command": [sys.executable, *sys.argv], "torch": torch.__version__,
                                    "gpu": torch.cuda.get_device_name(0)})
    seed_all(cfg["seed"])
    if cfg["mode"] == "train":
        train(cfg, out)
        return
    texts, labels, _ = load(TRAIN if cfg["internal"] else cfg["evaluation_file"] or PUBLIC)
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["checkpoint"], local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained(
        cfg["checkpoint"], local_files_only=True, attn_implementation="eager").cuda()
    ids = select_indices(cfg["pilot"])[1] if cfg["internal"] else list(range(len(texts)))
    if cfg["size"]:
        ids = sorted(random.Random(42).sample(ids, cfg["size"]))
    thresholds = json.loads(cfg["thresholds"].read_text()) if cfg["thresholds"] else [0.5, 0.5]
    prob, cost = predict(model, tokenizer, texts, ids, cfg)
    save_scores(out, labels[ids], ids, prob, cost, thresholds)


if __name__ == "__main__":
    main()
