"""Train a joint NLBSE classifier with learned, max-pooled sparse features."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit

from simple_bench import OUT as SIMPLE, batches, benchmark, counts, features, predict as bm25_predict

HERE = Path(__file__).resolve().parent
OUT = HERE / "learned_sparse"
BASE = SIMPLE / "runs/bm25_comments_bigram"
ARMS = ("linear", "expansion", "distilled")
LINEAR_DISTILLED = "linear_distilled"
QUANTIZED = "distilled_int8"
EXPANDED = ("expansion", "distilled")
WITH_TEACHER = ("distilled", LINEAR_DISTILLED)
WIDTH = 64
KEEP = 12
EPOCHS = 4
TRAIN_BATCH = 256
SEED = 42
SPARSITY = 0.004


def bm25(texts, stats):
    return features(counts(texts, True, bigrams=True), "bm25", stats)


def load_base():
    with (BASE / "model.joblib").open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    artifact = joblib.load(BASE / "model.joblib")
    if artifact["weight"] != "bm25" or len(artifact["stats"]) != 2 or not artifact["bigrams"]:
        raise ValueError("Expected the frozen BM25 bigram-and-comment control")
    return artifact, checksum


def marginal(ovr_probabilities):
    joint = ovr_probabilities / ovr_probabilities.sum(axis=1, keepdims=True)
    return np.column_stack((joint[:, 2] + joint[:, 3], joint[:, 1] + joint[:, 3]))


def predict_matrix(array, x):
    logits = x @ array["lexical"] + array["bias"]
    nonzero = int(x.nnz)
    if "expansion" in array:
        table = array["expansion"]
        pooled = np.zeros((x.shape[0], table.shape[1]), dtype=np.float32)
        present = np.diff(x.indptr) > 0
        if present.any():
            value = table[x.indices] * np.sqrt(x.data[:, None])
            pooled[present] = np.maximum.reduceat(value, x.indptr[:-1][present], axis=0)
        if "expansion_scale" in array:
            pooled *= array["expansion_scale"]
        np.maximum(pooled, 0, out=pooled)
        np.log1p(pooled, out=pooled)
        keep = int(array["keep"])
        selected = np.argpartition(pooled, -keep, axis=1)[:, -keep:]
        values = np.take_along_axis(pooled, selected, axis=1)
        logits += (values[:, :, None] * array["head"][selected]).sum(axis=1)
        nonzero += int(np.count_nonzero(values))
    return marginal(expit(logits)), nonzero


def predict(array, stats, texts):
    return predict_matrix(array, bm25(texts, stats))


def load_export(path):
    with np.load(path, allow_pickle=False) as contents:
        array = dict(contents)
    stats = list(zip((array.pop("idf0"), array.pop("idf1")), array.pop("average_lengths"), strict=True))
    return array, stats


def train(arm, x, labels, base, teacher):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Classifier(nn.Module):
        def __init__(self, expand):
            super().__init__()
            weight = torch.from_numpy(np.ascontiguousarray(base["model"].coef_.T)).float()
            self.lexical = nn.EmbeddingBag.from_pretrained(
                weight, mode="sum", freeze=False, include_last_offset=True)
            self.bias = nn.Parameter(torch.from_numpy(base["model"].intercept_.copy()).float())
            self.expansion = nn.Embedding(weight.shape[0], WIDTH) if expand else None
            if expand:
                nn.init.normal_(self.expansion.weight, mean=0.035, std=0.08)
                self.head = nn.Linear(WIDTH, 4, bias=False)
                nn.init.normal_(self.head.weight, mean=0., std=0.025)

        def forward(self, ids, offsets, values, rows, size):
            logits = self.lexical(ids, offsets, per_sample_weights=values) + self.bias
            if self.expansion is None:
                return logits, logits.new_zeros(())
            raw = self.expansion(ids) * values.sqrt()[:, None]
            pooled = raw.new_zeros((size, WIDTH))
            pooled.scatter_reduce_(0, rows[:, None].expand_as(raw), raw, reduce="amax", include_self=True)
            pooled = torch.log1p(pooled)
            top, idx = pooled.topk(KEEP, dim=1, sorted=False)
            sparse = torch.zeros_like(pooled).scatter(1, idx, top)
            return logits + self.head(sparse), pooled.mean()

        def export(self):
            def cpu(p):
                return p.detach().cpu().numpy()
            result = dict(lexical=cpu(self.lexical.weight), bias=cpu(self.bias))
            if self.expansion is not None:
                result.update(expansion=cpu(self.expansion.weight), head=np.ascontiguousarray(cpu(self.head.weight).T),
                              keep=np.array(KEEP))
            return result

    def tensors(matrix):
        lengths = np.diff(matrix.indptr)
        document = np.repeat(np.arange(len(lengths)), lengths)
        return (torch.as_tensor(matrix.indices, device="cuda", dtype=torch.long),
                torch.as_tensor(matrix.indptr, device="cuda", dtype=torch.long),
                torch.as_tensor(matrix.data, device="cuda", dtype=torch.float32),
                torch.as_tensor(document, device="cuda", dtype=torch.long), len(lengths))

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Classifier(arm in EXPANDED).cuda()
    options = [{"params": [model.lexical.weight, model.bias], "lr": 0.002}]
    if model.expansion is not None:
        options.append({"params": [model.expansion.weight, model.head.weight], "lr": 0.005})
    optimizer = torch.optim.AdamW(options, weight_decay=0.01, fused=True)
    steps = math.ceil(len(labels)/TRAIN_BATCH)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps)
    joint = 2*labels[:,0] + labels[:,1]
    targets = torch.as_tensor(joint, device="cuda", dtype=torch.long)
    class_weight = torch.tensor([1., 8., 2., 10.], device="cuda")
    soft_targets = torch.as_tensor(teacher, device="cuda") if arm in WITH_TEACHER else None
    t = time.perf_counter()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(labels))
        total = torch.zeros((), device="cuda")
        for start in range(0, len(labels), TRAIN_BATCH):
            selected = order[start:start+TRAIN_BATCH]
            ids = torch.as_tensor(selected, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            logits, density = model(*tensors(x[selected]))
            log_prob = F.logsigmoid(logits)
            log_prob = log_prob - log_prob.logsumexp(dim=1, keepdim=True)
            loss = F.nll_loss(log_prob, targets[ids], weight=class_weight)
            if soft_targets is not None:
                p = log_prob.exp()
                marginals = torch.stack((p[:, 2]+p[:, 3], p[:, 1]+p[:, 3]), dim=1)
                distill = F.binary_cross_entropy(marginals.clamp(1e-6, 1-1e-6), soft_targets[ids])
                loss = 0.65*loss + 0.35*distill
            loss = loss + SPARSITY*density
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.detach()
        print(arm, "epoch", epoch+1, "loss", round(total.item()/steps, 5),
              "seconds", round(time.perf_counter()-t, 2), flush=True)
    seconds = time.perf_counter()-t
    array = model.export()
    probe = x[:512]
    with torch.inference_mode():
        logits, _ = model(*tensors(probe))
        expected = marginal(logits.sigmoid().cpu().numpy())
    observed, _ = predict_matrix(array, probe)
    max_error = float(np.max(np.abs(expected-observed)))
    if max_error > 2e-4:
        raise ValueError(f"CPU export differs from GPU forward: {max_error}")
    return array, seconds, max_error


def check_initialization(base, stats, texts):
    lexical = np.ascontiguousarray(base["model"].coef_.T, dtype=np.float32)
    bias = base["model"].intercept_.astype(np.float32)
    reproduced, _ = predict(dict(lexical=lexical, bias=bias), stats, texts)
    reference, _ = bm25_predict(base, texts)
    if not np.allclose(reproduced, reference, rtol=1e-5, atol=1e-6):
        raise ValueError("The linear initialization differs from sklearn")


def fit(arms):
    from protocol import dataset, save
    from pipeline import select_indices, write_json

    texts, labels, holdout, gold, ids, calibration = dataset(full=True)
    base, checksum = load_base()
    stats = base["stats"]
    check_initialization(base, stats, holdout[:1024])
    with np.load(OUT / "teacher.npz", allow_pickle=False) as contents:
        fit_rows, _ = select_indices(pilot=False)
        if not np.array_equal(contents["rows"], fit_rows):
            raise ValueError("Teacher probabilities must match the fixed training rows")
        teacher = contents["probabilities"]
    x = bm25(texts, stats)
    del texts
    if teacher.shape != labels.shape or not np.isfinite(teacher).all() or teacher.min() < 0 or teacher.max() > 1:
        raise ValueError("Teacher probabilities are invalid")
    for arm in arms:
        directory = OUT / "runs" / arm
        directory.mkdir(parents=True, exist_ok=True)
        array, seconds, export_error = train(arm, x, labels, base, teacher)
        if not all(np.isfinite(a).all() for a in array.values()):
            raise ValueError(f"Non-finite weights for {arm}")
        path = directory / "model.npz"
        np.savez_compressed(directory / "model.part.npz", **array, idf0=stats[0][0], idf1=stats[1][0],
                            average_lengths=np.array([avg for _, avg in stats], dtype=np.float64))
        (directory / "model.part.npz").replace(path)
        write_json(directory / "config.json", dict(arm=arm, base_sha256=checksum, fit_rows=len(labels), epochs=EPOCHS,
                                                  batch_size=TRAIN_BATCH, seed=SEED, expansion_width=WIDTH if arm in EXPANDED else 0,
                                                  retained=KEEP if arm in EXPANDED else 0, l1_coefficient=SPARSITY,
                                                  teacher_soft_loss_fraction=0.35 if arm in WITH_TEACHER else 0.,
                                                  training_seconds=seconds, cpu_export_max_probability_error=export_error))
        p, timing = benchmark(lambda rows: predict(array, stats, rows), holdout, [0.5, 0.5], path, 1)
        save(directory / "holdout", gold, ids, calibration, p, timing)


def public(arms):
    from protocol import dataset, save

    _, _, texts, labels, ids, calibration = dataset(public=True)
    base, _ = load_base()
    for arm in arms:
        if arm == "bm25":
            thresholds = json.loads((BASE / "holdout/metrics.json").read_text())["thresholds"]
            path = BASE / "model.joblib"
            run = lambda rows: bm25_predict(base, rows)
        else:
            directory = OUT / "runs" / arm
            thresholds = json.loads((directory / "holdout/metrics.json").read_text())["thresholds"]
            path = directory / "model.npz"
            array, stats = load_export(path)
            run = lambda rows: predict(array, stats, rows)
        probabilities, cost = benchmark(run, texts, thresholds, path, 3)
        save(OUT / "runs" / arm / "public", labels, ids, calibration, probabilities, cost, thresholds)


def quantize():
    from protocol import dataset, save, write_json

    original = OUT / "runs/distilled"
    directory = OUT / "runs" / QUANTIZED
    directory.mkdir(parents=True, exist_ok=True)
    array, stats = load_export(original / "model.npz")
    thresholds = json.loads((original / "holdout/metrics.json").read_text())["thresholds"]
    quantized = dict(array)
    table = array["expansion"]
    scale = np.maximum(np.max(np.abs(table), axis=0)/127., np.finfo(np.float32).tiny)
    quantized["expansion"] = np.rint(table/scale).clip(-127, 127).astype(np.int8)
    quantized["expansion_scale"] = scale
    path = directory / "model.npz"
    np.savez_compressed(directory / "model.part.npz", **quantized, idf0=stats[0][0], idf1=stats[1][0],
                        average_lengths=np.array([avg for _, avg in stats], dtype=np.float64))
    (directory / "model.part.npz").replace(path)
    _, _, holdout, gold, ids, calibration = dataset(full=True)
    probe = bm25(holdout[:1024], stats)
    full, _ = predict_matrix(array, probe)
    converted, _ = predict_matrix(quantized, probe)
    write_json(directory / "config.json", dict(source="distilled", expansion_dtype="int8",
                                               quantization="symmetric per expansion dimension",
                                               thresholds=thresholds,
                                               max_probability_delta_probe=float(np.max(np.abs(full-converted)))))
    p, cost = benchmark(lambda texts: predict(quantized, stats, texts), holdout, thresholds, path, 1)
    save(directory / "holdout", gold, ids, calibration, p, cost, thresholds)


def infer(arm, source, destination):
    directory = OUT / "runs" / arm
    array, stats = load_export(directory / "model.npz")
    threshold = json.loads((directory / "holdout/metrics.json").read_text())["thresholds"]
    with source.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function column")
        with destination.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(("vulnerability", "MAT"))
            for rows in batches(row["Function"] for row in reader):
                p, _ = predict(array, stats, rows)
                writer.writerows((p > threshold).astype(np.int8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="stage", required=True)
    modes.add_parser("fit").add_argument("--linear-distilled", action="store_true")
    public_modes = modes.add_parser("public").add_mutually_exclusive_group()
    public_modes.add_argument("--linear-distilled", action="store_const", dest="variant", const=LINEAR_DISTILLED)
    public_modes.add_argument("--int8", action="store_const", dest="variant", const=QUANTIZED)
    modes.add_parser("quantize")
    cli = modes.add_parser("predict")
    cli.add_argument("arm", choices=(*ARMS, LINEAR_DISTILLED, QUANTIZED))
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        fit((LINEAR_DISTILLED,) if args.linear_distilled else ARMS)
    elif args.stage == "public":
        public((args.variant,) if args.variant else ("bm25", *ARMS))
    elif args.stage == "quantize":
        quantize()
    else:
        infer(args.arm, args.input, args.output)


if __name__ == "__main__":
    main()
