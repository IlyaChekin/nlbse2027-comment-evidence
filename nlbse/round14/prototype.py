"""Test whether training-only near copies can predict MAT when its tag was lost."""

import argparse
from collections import Counter
import csv
import json
import time

import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from audit import HERE, ROOT, dataset, split, write_json, checksum
from mat import WORD, cut_at_budget, evaluate, example


OUT = HERE / "prototype"
N_FEATURES = 2**20
REPRESENTATIONS = ("raw", "tagged", "margin", "missing_margin", "agreement")


def shingles(code):
    words = WORD.findall(code)
    for length in (3, 5):
        for start in range(len(words)-length+1):
            span = words[start:start+length]
            if any(word[0].isalpha() and len(word) >= 3 for word in span):
                yield " ".join(span)


VEC = HashingVectorizer(analyzer=shingles, binary=True, norm=None, alternate_sign=False,
                       n_features=N_FEATURES, dtype=np.float32)


def prepare(values):
    base, _, _, codes = zip(*(example(t) for t in values), strict=True)
    return np.asarray(base, dtype=bool), VEC.transform(codes)


def weight(x, idf):
    ar = x.multiply(idf).tocsr()
    ar.eliminate_zeros()
    return normalize(ar, copy=False)


def load_index(name):
    directory = OUT / name
    receipt = json.loads((directory / "policy.json").read_text())
    if not all(checksum(directory/file) == value for file, value in receipt["sha256"].items()):
        raise ValueError("Training-only prototype index changed")
    with np.load(directory / "metadata.npz") as f:
        families = f["families"]
        idf = f["idf"]
    return receipt, idf, load_npz(directory / "training.npz"), families


def compact_max(query, collection):
    if not collection.shape[0]:
        return np.zeros(query.shape[0], dtype=np.float32)
    index = collection.T.tocsr()
    out = np.empty(query.shape[0], dtype=np.float32)
    for at in range(0, len(out), 192):
        comparisons = query[at:at+192] @ index
        out[at:at+192] = comparisons.max(axis=1).toarray().ravel()
    return out


def scores(x, base, training, families):
    negative, tagged, untagged = [compact_max(x, training[families == value]) for value in (0, 1, 2)]
    signal = np.maximum(tagged, untagged)
    def margin(s):
        return s * np.maximum(s-negative, 0)
    return dict(zip(REPRESENTATIONS, (np.where(base, 2., v) for v in
                (signal, tagged, margin(signal), margin(untagged), margin(np.sqrt(tagged*untagged)))), strict=True))


def tune(texts, y, parts, name):
    started = time.perf_counter()
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    fit, dev = parts["fit"], parts["dev"]
    flags, feature = prepare([texts[i] for i in fit])
    counts = np.bincount(feature.indices, minlength=N_FEATURES)
    # Drop very common buckets and one-off hashed groups before any matching.
    idf = np.where((counts >= 2) & (counts <= 512), np.log((len(fit)+1)/(1+counts)), 0).astype(np.float32)
    training = weight(feature, idf)
    positive = y[fit, 1].astype(bool)
    families = np.where(positive, np.where(flags, 1, 2), 0).astype(np.int8)
    save_npz(directory/"training.npz", training)
    np.savez_compressed(directory/"metadata.npz", idf=idf, families=families, rows=fit)
    del feature
    base, xd = prepare([texts[i] for i in dev])
    label = y[dev, 1].astype(bool)
    predictions = scores(weight(xd, idf), base, training, families)
    budget = int((base & ~label).sum())
    rows = {}
    for key, p in predictions.items():
        cut = cut_at_budget(p, label, budget)
        rows[key] = dict(threshold=cut, dev=evaluate(label, base, p > cut),
                         residual_average_precision=average_precision_score(label[~base],p[~base]))
        print(name, key, rows[key], flush=True)
    choose = max(rows, key=lambda key: (rows[key]["dev"]["recovered"], rows[key]["residual_average_precision"]))
    write_json(directory/"policy.json", dict(chosen=choose, candidates=rows, reference_shape=training.shape,
               reference_nonzero=training.nnz, seconds=time.perf_counter()-started,
               training_label_families=dict(Counter(map(int, families))),
               sha256={k: checksum(directory/k) for k in ("training.npz", "metadata.npz")}))


def test():
    receipt, idf, training, families = load_index("project")
    texts, y, project, _ = dataset()
    parts = split()
    out = {}
    for group in ("test_a", "test_b"):
        rows = parts[group]
        label = y[rows,1].astype(bool)
        base, x = prepare([texts[i] for i in rows])
        values = scores(weight(x, idf), base, training, families)
        budget = int((base & ~label).sum())
        out[group] = {}
        for key, p in values.items():
            cutoff = receipt["candidates"][key]["threshold"]
            exact = p > cut_at_budget(p, label, budget)
            fixed = p > cutoff
            out[group][key] = dict(exact=evaluate(label,base,exact),frozen=evaluate(label,base,fixed))
            with (OUT / f"{group}_{key}.csv").open("w",newline="") as stream:
                w = csv.writer(stream)
                w.writerow(("row","project","label","base","score","fixed","matched"))
                w.writerows(zip(rows,project[rows],label.astype(int),base.astype(int),p,fixed.astype(int),exact.astype(int),strict=True))
        print(group,out[group][receipt["chosen"]],flush=True)
    write_json(OUT/"results.json", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("tune", "test", "official_fit"))
    stage = p.parse_args().stage
    if stage == "test":
        test()
        return
    texts, y, _, _ = dataset()
    if stage == "tune":
        tune(texts, y, split(), "project")
    else:
        from pipeline import select_indices
        fit, internal = select_indices(False)
        with np.load(ROOT/"round8/frozen/holdout.npz") as f:
            if f["rows"].tolist() != internal:
                raise ValueError("Internal calibration has moved")
            dev = f["rows"][f["folds"] == 3]
        tune(texts, y, {"fit": np.asarray(fit), "dev": dev}, "official")


if __name__ == "__main__":
    main()
