"""Learn a MAT head over complete comments with lexical and comment-parser controls."""
import argparse
import json
from pathlib import Path
import re
from statistics import median
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round10"))
from students import (PUBLIC, ROOT, TRAIN, calibrated, dataset, group_codes, load, metric,
                      predict_bits, save, select_indices, sha256, write_json)
from comment_spans import LEXEME
from round2_report import bootstrap_delta

OUT = HERE / "mat"
KEYS = (r"TODO", r"FIXME", r"XXX", r"HACK", r"TO[ -]+DO", r"FIX[ -]+ME", r"WORK[ -]?AROUND",
        r"KLUDGE", r"BUG", r"TEMP(?:ORARY)?", r"TBD", r"FIX", r"REMOVE", r"BROKEN", r"SHOULD", r"REVISIT")
MARKERS = re.compile(r"(?<![a-z0-9])(?:" + "|".join(f"(?P<k{i}>{k})" for i,k in enumerate(KEYS)) + r")(?![a-z0-9])", re.I | re.ASCII)
LOOSE = re.compile(r"//[^\r\n]*|/\*[\s\S]*?(?:\*/|\Z)")
J = len(KEYS)
N_COLS = 9*J + 5


def collect(texts):
    # Separate evidence by lexical scope. Loose scanning is one learned feature channel.
    features = np.zeros((len(texts), N_COLS), dtype=np.float32)
    words = []
    for i, text in enumerate(texts):
        row = features[i]
        segments = []
        for span in LEXEME.finditer(text):
            fragment = span[0]
            is_comment = fragment.startswith(("//", "/*"))
            if is_comment:
                segments.append(fragment)
                row[-3] += fragment.startswith("//")
                row[-2] += fragment.startswith("/*")
            for marker in MARKERS.finditer(fragment):
                k = int(marker.lastgroup[1:])
                if is_comment:
                    row[k] += 1
                    row[J+k] += marker[0].isupper()
                    row[2*J+k] += not fragment[2:marker.start()].strip(" \t\r\n*")
                    row[3*J+k] += fragment.startswith("/*")
                    row[4*J+k] += marker.end() == len(fragment) or fragment[marker.end():marker.end()+1] == ":"
                else:
                    row[5*J+k] += 1
        words.append(" ".join(segments))
        for marker in MARKERS.finditer(text):
            row[6*J+int(marker.lastgroup[1:])] += 1
        if row[6*J:7*J].any():
            for span in LOOSE.finditer(text):
                for marker in MARKERS.finditer(text, *span.span()):
                    row[7*J+int(marker.lastgroup[1:])] += 1
            row[8*J:9*J] = np.maximum(0, row[6*J:7*J]-row[:J]-row[5*J:6*J])
        row[-5] = len(text)
        row[-4] = len(words[-1])
        row[-1] = text.count("\n")
    return np.log1p(features), words


def dense_columns(dense, mode):
    if mode == "comment":
        selected = np.r_[0:5*J, N_COLS-5:N_COLS]
    else:
        selected = np.arange(N_COLS)
    return csr_matrix(dense[:, selected])


def text_features(words):
    vec = HashingVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), n_features=2**17,
                            dtype=np.float32, norm="l2")
    return vec.transform(words)


def matrix(dense, sparse, mode):
    x = dense_columns(dense, mode)
    return hstack([x, sparse], format="csr") if mode == "ngrams" else x


def tune():
    train, labels, _ = load(TRAIN)
    fit, holdout = select_indices(False)
    with np.load(ROOT/"round8/frozen/holdout.npz") as fold:
        if holdout != fold["rows"].tolist():
            raise ValueError("Mismatched MAT validation rows")
        calibration, review = fold["folds"] == 3, fold["folds"] == 4
    start = time.perf_counter()
    x, words = collect([train[i] for i in fit])
    test, test_words = collect([train[i] for i in holdout])
    ngrams, test_ngrams = text_features(words), text_features(test_words)
    OUT.mkdir(parents=True, exist_ok=True)
    y, test_y = labels[fit, 1], labels[holdout]
    fixed_vulnerability = np.zeros(len(test_y))
    results = []
    for mode in ("comment", "scope", "ngrams"):
        a, b = matrix(x, ngrams, mode), matrix(test, test_ngrams, mode)
        for strength in (.05, .3, 3.):
            # Mild weighting gives the rare label adequate influence during fit.
            model = LogisticRegression(C=strength, max_iter=500, solver="liblinear", random_state=42)
            model.fit(a, y, sample_weight=np.where(y == 1, 3., 1.))
            p = model.predict_proba(b)[:, 1]
            both = np.column_stack((fixed_vulnerability, p))
            threshold = calibrated(test_y[calibration], both[calibration])[1]
            bits = both > (.5, threshold)
            record = dict(name=mode, inverse_regularization=strength, threshold=threshold,
                          calibration=metric(test_y[calibration], bits[calibration])["MAT"],
                          average_precision=average_precision_score(test_y[calibration, 1], p[calibration]),
                          review=metric(test_y[review], bits[review])["MAT"])
            results.append((record, model.coef_[0].astype(np.float32), model.intercept_[0], p))
            print(record, flush=True)
    winner = max(results, key=lambda r: (r[0]["calibration"]["f1"], r[0]["average_precision"]))
    record, weight, bias, p = winner
    config = OUT/"weights.npz"
    np.savez_compressed(config, weight=weight, bias=bias)
    np.savez_compressed(OUT/"holdout.npz", ids=holdout, p=p, y=test_y)
    write_json(OUT/"selected.json", dict(chosen=record, candidates=[r[0] for r in results],
        weights_sha256=sha256(config), fit_rows=len(fit), elapsed_seconds=time.perf_counter()-start))


def predict(texts, selected, weight, bias):
    dense, words = collect(texts)
    mode = selected["name"]
    x = matrix(dense, text_features(words) if mode == "ngrams" else None, mode)
    return expit(x @ weight + bias), dense, int(x.nnz)


def public():
    _, _, texts, y, ids, _ = dataset(public=True)
    _, _, columns = load(PUBLIC, audit=True)
    receipt = json.loads((OUT/"selected.json").read_text())
    weights = OUT/"weights.npz"
    if sha256(weights) != receipt["weights_sha256"]:
        raise ValueError("MAT head changed after calibration")
    with np.load(weights) as artifact:
        trials = []
        for _ in range(3):
            start = time.perf_counter()
            p, dense, nnz = predict(texts, receipt["chosen"], artifact["weight"], artifact["bias"])
            trials.append(time.perf_counter()-start)
    old = predict_bits(ROOT/"round10/student_context/student_lexical/predictions.csv", ids,
                       {"thresholds":json.loads((ROOT/"round10/student_context/policy.json").read_text())["lexical_control"]["thresholds"]})
    candidate = np.column_stack((old[:, 0], p > receipt["chosen"]["threshold"]))
    groups = group_codes(zip(columns["Projectname"], columns["Filepath"], strict=True))
    positive_missed = (y[:,1] == 1) & ~old[:,1]
    distribution = dict(positives=int(y[:,1].sum()), previous_missed=int(positive_missed.sum()),
        missed_with_standard_tag_in_comment=int((positive_missed & dense[:, :4].any(axis=1)).sum()),
        missed_with_standard_tag_anywhere=int((positive_missed & dense[:, 6*J:6*J+4].any(axis=1)).sum()),
        missed_with_extended_marker_in_comment=int((positive_missed & dense[:, :J].any(axis=1)).sum()))
    result = dict(scores=metric(y, candidate), ci=bootstrap_delta(y, old, candidate, groups),
                  cost=dict(wall_seconds=median(trials), trials=trials, active_features=nnz), distribution=distribution)
    np.savez_compressed(OUT/"public.npz", ids=ids, p=p, y=y)
    write_json(OUT/"analysis.json",result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "public"))
    args = parser.parse_args()
    tune() if args.stage == "tune" else public()
