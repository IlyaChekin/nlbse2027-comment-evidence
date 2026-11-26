"""Compare lightweight term weighting for one joint NLBSE logistic classifier."""
import argparse
import csv
from itertools import islice
import json
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

from classical import TOKEN, comments, predict as predict_previous

HERE = Path(__file__).resolve().parent
OUT = HERE / "simple"
PREVIOUS = HERE / "runs/full/sparse_roles"
BATCH = 1024
REPEATS = 3
MODES = ("count", "tfidf", "bm25", "tfidf_comments", "bm25_comments")
FOLLOWUP = ("logtfidf", "logtfidf_comments", "bm25_comments_bigram")
ALL_MODES = (*MODES, *FOLLOWUP)
SETTINGS = dict(alpha=1e-5, max_iter=35, random_state=42, n_jobs=2, loss="log_loss")
K1, B = 1.2, 0.75


def counts(texts, use_comments, bigrams=False):
    def vector(bits, ngrams):
        return HashingVectorizer(n_features=2**bits, token_pattern=TOKEN, ngram_range=ngrams,
                                 dtype=np.float32, norm=None, alternate_sign=False)
    result = [vector(17, (1, 2) if bigrams else (1, 1)).transform(texts)]
    if use_comments:
        result.append(vector(15, (1, 2)).transform(map(comments, texts)))
    return result


def fit_stats(matrix, weight):
    n = matrix.shape[0]
    df = np.bincount(matrix.indices, minlength=matrix.shape[1]).astype(np.float32)
    if weight == "bm25":
        idf = np.log1p((n-df+0.5)/(df+0.5))
    else:
        idf = 1 + np.log((1+n)/(1+df))
    return idf.astype(np.float32), max(1.0, float(np.asarray(matrix.sum(axis=1)).mean()))


def features(matrices, weight, stats):
    blocks = []
    for channel, (raw, (idf, average_length)) in enumerate(zip(matrices, stats, strict=True)):
        block = raw.copy()
        if weight == "logtfidf":
            np.log(block.data, out=block.data)
            block.data += 1
        if weight in ("tfidf", "logtfidf"):
            block.data *= idf[block.indices]
        elif weight == "bm25":
            length = np.asarray(block.sum(axis=1)).ravel()
            norm = K1 * (1-B+B*(length/average_length))
            denominator = block.data + np.repeat(norm, np.diff(block.indptr))
            block.data *= (K1+1) * idf[block.indices] / denominator
        normalize(block, copy=False)
        if channel:
            block.data *= 1.5
        blocks.append(block)
    return hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]


def fit(mode, matrices, labels):
    weight, _, role = mode.partition("_")
    stats = [fit_stats(m, weight) for m in matrices[:2 if role else 1]]
    x = features(matrices[:len(stats)], weight, stats)
    joint = 2*labels[:,0] + labels[:,1]
    model = SGDClassifier(**SETTINGS)
    model.fit(x, joint, sample_weight=np.asarray([1., 8., 2., 10.])[joint])
    if model.classes_.tolist() != [0, 1, 2, 3]:
        raise ValueError("All four label combinations must be represented in training")
    return dict(weight=weight, stats=stats, model=model, bigrams=mode.endswith("_bigram"))


def predict(artifact, texts):
    raw = counts(texts, len(artifact["stats"]) == 2, artifact.get("bigrams", False))
    x = features(raw, artifact["weight"], artifact["stats"])
    joint = artifact["model"].predict_proba(x)
    return np.column_stack((joint[:, 2:].sum(axis=1), joint[:, [1, 3]].sum(axis=1))), x.nnz


def batches(rows):
    rows = iter(rows)
    while chunk := list(islice(rows, BATCH)):
        yield chunk


def pass_cpu(run, texts, thresholds):
    scores = []
    features_seen = 0
    for batch in batches(texts):
        p, nnz = run(batch)
        scores.append(p)
        features_seen += nnz
    p = np.concatenate(scores)
    bits = (p > thresholds).astype(np.int8)
    return p, bits, int(features_seen)


def benchmark(run, texts, thresholds, path, repeat):
    run(texts[:BATCH])
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        p, bits, nnz = pass_cpu(run, texts, thresholds)
        times.append(time.perf_counter()-start)
        if len(times) > 1 and not np.array_equal(bits, reference):
            raise ValueError("Repeated inference changed predicted labels")
        reference = bits
    return p, dict(inference_seconds=times, median_seconds=median(times), weights_bytes=path.stat().st_size,
                   rows=len(texts), batch_size=BATCH, nnz=nnz, platform="Banana CPU")


def run_previous(model, texts):
    p, info = predict_previous("sparse_roles", model, texts)
    return p, info["nonzero_features"]


def holdout(modes):
    from protocol import dataset, save, write_json
    train, y, validation, gold, ids, calibration = dataset(full=True)
    start = time.perf_counter()
    matrices = {False: counts(train, True)}
    if any(mode.endswith("_bigram") for mode in modes):
        matrices[True] = counts(train, True, bigrams=True)
    print("Encoded", len(train), "fit rows in", round(time.perf_counter()-start, 2), "seconds", flush=True)
    for mode in modes:
        directory = OUT / "runs" / mode
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.joblib"
        start = time.perf_counter()
        artifact = fit(mode, matrices[mode.endswith("_bigram")], y)
        pending = path.with_suffix(".part")
        joblib.dump(artifact, pending, compress=3)
        pending.replace(path)
        write_json(directory / "config.json", dict(mode=mode, fit_rows=len(train), trainer=SETTINGS,
                                                  bm25=dict(k1=K1, b=B), fit_seconds=time.perf_counter()-start))
        p, cost = benchmark(lambda texts: predict(artifact, texts), validation, [0.5, 0.5], path, 1)
        save(directory / "holdout", gold, ids, calibration, p, cost)


def public(modes):
    from protocol import dataset, save
    _, _, texts, y, ids, calibration = dataset(public=True)
    for mode in modes:
        directory = OUT / "runs" / mode
        if mode == "previous_sparse":
            path = PREVIOUS / "model.joblib"
            threshold = json.loads((PREVIOUS / "metrics.json").read_text())["thresholds"]
            artifact = joblib.load(path)
            run = lambda batch: run_previous(artifact, batch)
        else:
            path = directory / "model.joblib"
            threshold = json.loads((directory / "holdout/metrics.json").read_text())["thresholds"]
            artifact = joblib.load(path)
            run = lambda batch: predict(artifact, batch)
        p, cost = benchmark(run, texts, threshold, path, REPEATS)
        save(directory / "public", y, ids, calibration, p, cost, threshold)


def infer(mode, source, destination):
    directory = OUT / "runs" / mode
    model = joblib.load(directory / "model.joblib")
    thresholds = json.loads((directory / "holdout/metrics.json").read_text())["thresholds"]
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function column")
        with destination.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("vulnerability", "MAT"))
            for batch in batches(row["Function"] for row in reader):
                p, _ = predict(model, batch)
                writer.writerows((p > thresholds).astype(np.int8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="stage", required=True)
    modes.add_parser("holdout").add_argument("--followup", action="store_true")
    modes.add_parser("public").add_argument("--followup", action="store_true")
    inference = modes.add_parser("predict")
    inference.add_argument("mode", choices=ALL_MODES)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "holdout":
        holdout(FOLLOWUP if args.followup else MODES)
    elif args.stage == "public":
        public(FOLLOWUP if args.followup else (*MODES, "previous_sparse"))
    else:
        infer(args.mode, args.input, args.output)


if __name__ == "__main__":
    main()
