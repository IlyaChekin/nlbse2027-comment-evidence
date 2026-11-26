"""Sparse lexical, structural forest and rare-shingle memory classifiers."""
import argparse
from collections import Counter, defaultdict
import json
import re
import time
from zlib import crc32

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

TOKEN = r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s\w]"
COMMENTS = re.compile(r'(?:"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27)|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)')
STRUCTURE = ("if", "else", "for", "while", "switch", "return", "throw", "try",
             "catch", "new", "TODO", "FIXME", "NOTE", "HACK", "{", "}", "(",
             ")", "[", "]", ";", "->", "&&", "||", "==", "#", "*", "&")


def comments(text):
    return " ".join(m["comment"] for m in COMMENTS.finditer(text) if m["comment"])


def sparse_features(texts):
    word = HashingVectorizer(n_features=2**17, token_pattern=TOKEN, ngram_range=(1, 2),
                             dtype=np.float32)
    char = HashingVectorizer(n_features=2**17, analyzer="char", ngram_range=(3, 4),
                             dtype=np.float32)
    annotation = HashingVectorizer(n_features=2**15, token_pattern=TOKEN, ngram_range=(1, 2),
                                   dtype=np.float32)
    return hstack((word.transform(texts), char.transform(texts),
                   annotation.transform(map(comments, texts))), format="csr")


def vocabulary_roles(texts):
    """Separate annotation wording, executable text and a bounded long suffix."""
    code, annotation, suffix = [], [], []
    for text in texts:
        fragments = []

        def replace(match):
            if match["comment"] is None:
                return match[0]
            fragments.append(match["comment"])
            return " "

        code.append(COMMENTS.sub(replace, text))
        annotation.append(" ".join(fragments))
        suffix.append(code[-1][-2048:] if len(code[-1]) > 2048 else "")
    return code, annotation, suffix


def sparse_roles(texts):
    code, annotation, suffix = vocabulary_roles(texts)
    word = dict(token_pattern=TOKEN, ngram_range=(1, 2))
    char = dict(analyzer="char", ngram_range=(3, 5))
    settings = ((code, word, 17, 1.), (code, char, 17, 1.),
                (annotation, word, 15, 1.5), (annotation, char, 16, 1.5),
                (suffix, word, 15, 0.75))
    features = []
    for texts, config, bits, scale in settings:
        vector = HashingVectorizer(n_features=2**bits, dtype=np.float32, norm=None,
                                   **config).transform(texts)
        vector.data = np.sign(vector.data) * np.log1p(np.abs(vector.data))
        vector = normalize(vector, copy=False)
        vector.data *= scale
        features.append(vector)
    return hstack(features, format="csr")


VECTORIZERS = {"sparse": sparse_features, "sparse_roles": sparse_roles}


def structural_row(text):
    lines = text.splitlines()
    annotation = comments(text)
    words = re.findall(TOKEN, text)
    count = Counter(words)
    indentation = [len(line) - len(line.lstrip()) for line in lines]
    depths = np.cumsum([int(word == "{") - int(word == "}") for word in words])
    return [len(text), len(lines), len(words), len(count), len(annotation),
            sum(not line.strip() for line in lines), max(map(len, lines), default=0),
            max(indentation, default=0), np.mean(indentation) if indentation else 0,
            int(depths.max(initial=0)), len(annotation)/max(1, len(text)),
            sum(word.isdigit() for word in words), sum("_" in word for word in words),
            *(text.count(key) if not key.isalnum() else count[key] for key in STRUCTURE)]


def structural_features(texts):
    return np.asarray(list(map(structural_row, texts)), dtype=np.float32)


def shingles(text):
    tokens = re.findall(TOKEN, text)
    codes = {crc32(" ".join(tokens[i:i+5]).encode()) for i in range(max(1, len(tokens)-4))}
    return sorted(codes)[:64]


class ShingleMemory:
    def fit(self, texts, labels):
        keys = list(map(shingles, texts))
        postings = defaultdict(list)
        for row, signature in enumerate(keys):
            for key in signature:
                if len(postings[key]) <= 128:
                    postings[key].append(row)
        self.postings = {key: ids for key, ids in postings.items() if len(ids) <= 128}
        self.sizes = np.asarray([sum(key in self.postings for key in signature) for signature in keys])
        self.labels = labels
        self.prior = labels.mean(axis=0)
        return self

    def predict_proba(self, texts):
        probabilities = np.empty((len(texts), 2))
        traversed = 0
        for row, text in enumerate(texts):
            hits = Counter()
            usable = 0
            for key in shingles(text):
                postings = self.postings.get(key)
                if postings:
                    hits.update(postings)
                    traversed += len(postings)
                    usable += 1
            if not hits:
                probabilities[row] = self.prior
                continue
            ids = np.fromiter(hits.keys(), dtype=np.int64)
            overlap = np.fromiter(hits.values(), dtype=np.float32)
            similarity = overlap / (usable + self.sizes[ids] - overlap)
            top = np.lexsort((ids, -similarity))[:5]
            weights = similarity[top] ** 3
            probabilities[row] = ((self.labels[ids[top]] * weights[:, None]).sum(axis=0)
                                  + 0.15 * self.prior) / (weights.sum() + 0.15)
        return probabilities, dict(postings_traversed=traversed, indexed_keys=len(self.postings))


def fit(kind, texts, labels):
    if kind in VECTORIZERS:
        joint = labels @ np.array([2, 1])
        model = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=35, random_state=42, n_jobs=2)
        model.fit(VECTORIZERS[kind](texts), joint, sample_weight=np.array([1., 8., 2., 10.])[joint])
        return model
    if kind == "forest":
        model = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=3, max_depth=20,
                                     max_features=0.7, class_weight="balanced", n_jobs=4,
                                     random_state=42)
        return model.fit(structural_features(texts), labels)
    return ShingleMemory().fit(texts, labels)


def predict(kind, model, texts):
    if kind in VECTORIZERS:
        features = VECTORIZERS[kind](texts)
        probabilities = model.predict_proba(features)
        return np.column_stack((probabilities[:, 2:].sum(axis=1),
                                probabilities[:, [1, 3]].sum(axis=1))), dict(nonzero_features=int(features.nnz))
    if kind == "forest":
        features = structural_features(texts)
        return np.column_stack([value[:, 1] for value in model.predict_proba(features)]), dict(
            feature_columns=features.shape[1], trees=len(model.estimators_))
    return model.predict_proba(texts)


def main():
    from protocol import HERE, dataset, save, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=(*VECTORIZERS, "forest", "memory", "all"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if args.public and not args.full:
        parser.error("Public confirmation requires a frozen full-fit model")
    training, y, evaluation, labels, ids, calibration = dataset(args.full, args.public)
    root = HERE / "runs" / ("full" if args.full else "pilot")
    kinds = ("sparse", "forest", "memory") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        out = root / kind
        out.mkdir(parents=True, exist_ok=True)
        path = out / "model.joblib"
        thresholds = None
        t = time.perf_counter()
        if args.public:
            thresholds = json.loads((out / "metrics.json").read_text())["thresholds"]
            model = joblib.load(path)
        else:
            model = fit(kind, training, y)
            pending = path.with_suffix(".part")
            joblib.dump(model, pending, compress=3)
            pending.replace(path)
            write_json(out / "config.json", dict(kind=kind, seed=42, fit_rows=len(y)))
        fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        probabilities, detail = predict(kind, model, evaluation)
        cost = dict(platform="Banana CPU", inference_seconds=time.perf_counter()-t, fit_or_load_seconds=fit_seconds,
                    bytes=path.stat().st_size, rows=len(ids), **detail)
        save(out / "public" if args.public else out, labels, ids, calibration, probabilities, cost, thresholds)


if __name__ == "__main__":
    main()
