"""Test whether MAT cues transfer across projects when grouped by syntactic role."""
import csv
import json
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "round14"))
from audit import dataset, split, write_json, checksum, LEXEME, TAG
from mat import cut_at_budget, evaluate

WORDS = re.compile(r"[A-Za-z][a-z]*|[A-Z]+(?=[A-Z][a-z]|[^a-zA-Z]|$)|[0-9]+")
IDENT = re.compile(r"\b[A-Za-z_]\w*\b", re.ASCII)
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(", re.ASCII)
FLOW = frozenset(("if", "else", "for", "while", "switch", "case", "return", "sizeof", "alignof"))
CANDIDATES = {
    "untyped": (False, 0),
    "by_role": (True, 0),
    "by_role_erasure": (True, .15),
}


def phrase(role, value, typed):
    words = [w.casefold() for w in WORDS.findall(value) if not w.isdigit()]
    if not words:
        return
    prefix = f"{role}/" if typed else ""
    for word in words:
        yield prefix + word
    for left, right in zip(words, words[1:]):
        yield prefix + left + " " + right


def inspect(text):
    markers = False
    comments, literals, pieces = [], [], []
    start = 0
    for span in LEXEME.finditer(text):
        pieces.extend((text[start:span.start()], " "))
        start = span.end()
        if span[0].startswith(("//", "/*")):
            comments.append(span[0])
            markers |= TAG.search(text, *span.span()) is not None
        else:
            literals.append(span[0])
    pieces.append(text[start:])
    declaration, _, body = "".join(pieces).partition("{")
    return markers, comments, literals, declaration, body


def roles(view, typed, erase):
    markers, comments, literals, declaration, body = view
    if not erase:
        for text in comments:
            for line in text.splitlines():
                yield from phrase("comment", line, typed)
    for text in literals:
        yield from phrase("literal", text, typed)
    for name in IDENT.findall(declaration):
        yield from phrase("declaration", name, typed)
    for name in CALL.findall(body):
        if name not in FLOW:
            yield from phrase("call", name, typed)
    for name in IDENT.findall(body):
        if name not in FLOW:
            yield from phrase("body", name, typed)


def untyped_features(view):
    return roles(view, typed=False, erase=view[0])


def typed_features(view):
    return roles(view, typed=True, erase=view[0])


def prepare(texts, ids):
    tick = time.perf_counter()
    inputs = [inspect(texts[i]) for i in ids]
    base = np.fromiter((x[0] for x in inputs), dtype=bool)
    has_comment = np.fromiter((bool(x[1]) for x in inputs), dtype=bool)
    print("prepared", len(ids), round(time.perf_counter() - tick, 3), flush=True)
    return inputs, base, has_comment


def vectorizer(typed):
    return HashingVectorizer(n_features=2**18, binary=True, norm="l2", alternate_sign=False,
                             dtype=np.float64, analyzer=typed_features if typed else untyped_features)


def probabilities(head, x, base):
    return np.where(base, 2., head.predict_proba(x)[:, 1])


def metrics(gold, base, comment, pred):
    recalled = gold & ~base & pred
    return dict(**evaluate(gold, base, pred), recovered_with_comment=int((recalled & comment).sum()),
                recovered_without_comment=int((recalled & ~comment).sum()),
                baseline_missed_with_comment=int((gold & ~base & comment).sum()),
                baseline_missed_without_comment=int((gold & ~base & ~comment).sum()))


def fit(texts, label, groups):
    train, train_base, _ = prepare(texts, groups["fit"])
    dev, dev_base, dev_comment = prepare(texts, groups["dev"])
    train_y = label[groups["fit"]]
    dev_y = label[groups["dev"]]
    dev_fp = int((dev_base & ~dev_y).sum())
    models, result = {}, {}
    for typed in (False, True):
        vec = vectorizer(typed)
        x = vec.transform(train)
        d = vec.transform(dev)
        for name, (candidate_typed, erasure_weight) in CANDIDATES.items():
            if candidate_typed != typed:
                continue
            mask = ~train_base | (train_y & bool(erasure_weight))
            weight = np.where(train_base, erasure_weight * 12, np.where(train_y, 12., 1.))
            head = LogisticRegression(C=2., solver="liblinear", max_iter=300, random_state=42)
            head.fit(x[mask], train_y[mask], sample_weight=weight[mask])
            models[name] = head
            score = probabilities(head, d, dev_base)
            threshold = cut_at_budget(score, dev_y, dev_fp)
            result[name] = dict(threshold=threshold, metrics=metrics(dev_y, dev_base, dev_comment,
                                 score > threshold), residual_ap=float(average_precision_score(
                                     dev_y[~dev_base], score[~dev_base])), fit_rows=int(mask.sum()))
            print("dev", name, result[name], flush=True)
    selected = max(result, key=lambda method: (
        result[method]["metrics"]["recovered"] - result[method]["metrics"]["lost"],
        result[method]["residual_ap"],
    ))
    if result[selected]["metrics"]["candidate"]["tp"] <= result[selected]["metrics"]["base"]["tp"]:
        selected = "baseline"
    return models, dict(selected=selected, methods=result, fit_rows=len(train), dev_rows=len(dev),
                        train_data_sha256=checksum(ROOT / "data/raw/train-00000-of-00001.parquet"),
                        public_data_sha256=checksum(ROOT / "data/raw/test-00000-of-00001.parquet"),
                        partition_sha256=checksum(ROOT / "round13/partition.npz"), code_sha256=checksum(Path(__file__)))


def test(name, ids, texts, label, project, models, policy):
    tick = time.perf_counter()
    inputs, base, comment = prepare(texts, ids)
    gold, families = label[ids], project[ids]
    scores = {"baseline": 2. * base}
    for typed in (False, True):
        features = vectorizer(typed).transform(inputs)
        scores.update({key: probabilities(models[key], features, base)
                       for key, (candidate_typed, _) in CANDIDATES.items() if typed == candidate_typed})
    report = {}
    for key, p in scores.items():
        fixed_t = .5 if key == "baseline" else policy["methods"][key]["threshold"]
        fixed, matched = p > fixed_t, np.zeros_like(base)
        thresholds = {}
        for family in sorted(set(families)):
            which = families == family
            budget = int((base[which] & ~gold[which]).sum())
            thresholds[family] = cut_at_budget(p[which], gold[which], budget)
            matched[which] = p[which] > thresholds[family]
            if int((matched[which] & ~gold[which]).sum()) != budget:
                raise ValueError(f"FP budget changed for {name}/{family}/{key}")
        report[key] = dict(fixed=metrics(gold, base, comment, fixed),
                           matched=metrics(gold, base, comment, matched),
                           project_thresholds=thresholds,
                           residual_ap=float(average_precision_score(gold[~base], p[~base])))
        with (HERE / f"{name}_{key}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("row", "project", "gold", "baseline", "comment", "score", "fixed", "matched"))
            writer.writerows(zip(ids, families, gold.astype(int), base.astype(int), comment.astype(int),
                                 p, fixed.astype(int), matched.astype(int), strict=True))
    print("test", name, "seconds", round(time.perf_counter()-tick, 3), flush=True)
    return report


def main():
    texts, labels, projects, _ = dataset()
    groups = split()
    models, policy = fit(texts, labels[:, 1].astype(bool), groups)
    joblib.dump(models, HERE / "heads.joblib", compress=3)
    policy["weights_sha256"] = checksum(HERE / "heads.joblib")
    write_json(HERE / "policy.json", policy)
    train_report = {key: test(key, groups[key], texts, labels[:, 1].astype(bool), projects, models, policy)
                    for key in ("test_a", "test_b")}
    public, public_y, source, _ = dataset(public=True)
    # The public diagnostic uses the one pooled FP, as does the official metric.
    train_report["public"] = test("public", np.arange(len(public)), public, public_y[:, 1].astype(bool),
                                  np.repeat("public", len(source)), models, policy)
    write_json(HERE / "results.json", train_report)


if __name__ == "__main__":
    main()
