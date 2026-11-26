"""Frozen project-disjoint evaluation of MAT cues around corrupted or missing tags."""

import argparse
from collections import Counter
import csv
from functools import lru_cache
import json
import re
import time

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from audit import HERE, ROOT, LEXEME, LOOSE_COMMENT, STEM, TAG, VARIANT, DEBT, dataset, split, write_json, checksum


MODELS = ("soft_boundary", "marker_geometry", "comment_words", "code_only", "masked_comment_code", "combined")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s\w]")
GEOMETRY = FeatureHasher(n_features=2**13, input_type="dict", alternate_sign=False, dtype=np.float32)
COMMENT = HashingVectorizer(n_features=2**16, analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32)
CODE = HashingVectorizer(n_features=2**17, token_pattern=WORD.pattern, ngram_range=(1, 3),
                         lowercase=False, dtype=np.float32)


def edge(char):
    if not char:
        return "end"
    if char == "_":
        return "underscore"
    if char.isdigit():
        return "digit"
    if char.isupper():
        return "capital"
    if char.islower():
        return "lower"
    return "boundary"


def example(text, code_required=True):
    comments, fragments, tagged, previous = [], [], False, 0
    for m in LEXEME.finditer(text):
        if m[0].startswith(("//", "/*")):
            body = m[0][2:] if m[0].startswith("//") else m[0][2:].removesuffix("*/")
            comments.append(body)
            tagged |= TAG.search(text, *m.span()) is not None
            if code_required:
                fragments.extend((text[previous:m.start()], " "))
                previous = m.end()
    if code_required:
        fragments.append(text[previous:])
    code = "".join(fragments)
    lead = text.lstrip().startswith(("//", "/*"))
    feat = {"has_comment": bool(comments), "leading_comment": lead}
    if not tagged:
        for body in comments:
            feat["split_tag"] = feat.get("split_tag", 0) + bool(VARIANT.search(body))
            feat["debt_phrase"] = feat.get("debt_phrase", 0) + bool(DEBT.search(body))
            for m in STEM.finditer(body):
                tag = m[0].lower()
                left = body[m.start()-1] if m.start() else ""
                right = body[m.end():m.end()+1]
                anchor = not body[:m.start()].strip("* \t\r\n")
                group = f"{edge(left)}/{edge(right)}"
                values = (f"stem/{tag}", "stem/any", f"shape/{group}", f"shape/{tag}/{group}",
                          f"initial/{int(anchor)}/{group}", f"case/{m[0].isupper()}/{group}")
                for value in values:
                    feat[value] = feat.get(value, 0) + 1
        # Delimiters occurring inside quoted literals or code are scored separately.
        feat["tag_outside_comment"] = bool(TAG.search(text))
        if feat["tag_outside_comment"]:
            feat["tag_in_loose_scope"] = any(TAG.search(text, *m.span()) for m in LOOSE_COMMENT.finditer(text))
    return tagged, dict(feat), "\n".join(comments), code


def extract(texts, code_required=True):
    base, features, comments, code = zip(*(example(text, code_required) for text in texts), strict=True)
    stem = np.fromiter(("stem/any" in x for x in features), dtype=bool, count=len(texts))
    return np.asarray(base, dtype=bool), stem, list(features), list(comments), list(code)


def vectors(data):
    return GEOMETRY.transform(data[2]), COMMENT.transform(data[3]), CODE.transform(data[4])


def view(name, vectors):
    markers, words, code = vectors
    if name == "marker_geometry":
        return markers
    if name == "comment_words":
        return hstack((markers, words), format="csr")
    if name in {"code_only", "masked_comment_code"}:
        return code
    return hstack((markers, words, code), format="csr")


def cut_at_budget(score, gold, max_fp, floor=0.):
    # Strict threshold retains entire score groups, with row order playing no role.
    negative = np.sort(score[~gold])[::-1]
    return float(max(floor, negative[max_fp])) if max_fp < len(negative) else float(floor)


def evaluate(gold, base, pred):
    result = {}
    for key, out in (("base", base), ("candidate", pred)):
        tp, fp, fn = int((gold & out).sum()), int((~gold & out).sum()), int((gold & ~out).sum())
        result[key] = dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(2*tp+fp+fn, 1))
    result.update(recovered=int((gold & ~base & pred).sum()), lost=int((gold & base & ~pred).sum()),
                  extra_fp=result["candidate"]["fp"]-result["base"]["fp"])
    return result


def candidates(texts, label, use, calibration, *, name="project"):
    t = time.perf_counter()
    a = extract([texts[i] for i in use])
    d = extract([texts[i] for i in calibration])
    xf, xd = vectors(a), vectors(d)
    gold = label[use, 1].astype(bool)
    dev_y = label[calibration, 1].astype(bool)
    model_records = []
    model_objects = {}
    for mode in MODELS:
        for strength in ((.12, 1.) if mode not in {"soft_boundary", "marker_geometry"} else (.12, .8, 4.) if mode == "marker_geometry" else (0.,)):
            key = f"{mode}/{strength:g}"
            if mode == "soft_boundary":
                score = d[1].astype(np.float32)
            else:
                pool = np.ones(len(a[0]), dtype=bool) if mode == "masked_comment_code" else ~a[0]
                # For the masked condition the model sees only code, including examples with known tags removed.
                sample = np.where(gold[pool], 3., 1.)
                if mode == "masked_comment_code":
                    sample = np.where(a[0][pool] & gold[pool], 1.5, sample)
                model = LogisticRegression(C=strength, solver="liblinear", max_iter=300, random_state=42)
                model.fit(view(mode, xf)[pool], gold[pool], sample_weight=sample)
                score = model.predict_proba(view(mode, xd))[:, 1]
                model_objects[key] = model
            score = np.where(d[0], 2., score)
            cutoff = cut_at_budget(score, dev_y, int((~dev_y & d[0]).sum()))
            metrics = evaluate(dev_y, d[0], score > cutoff)
            ap = average_precision_score(dev_y[~d[0]], score[~d[0]])
            model_records.append(dict(key=key, mode=mode, strength=strength, threshold=cutoff,
                                      metrics=metrics, residual_average_precision=ap))
            print(name, key, cutoff, metrics, "residual AP", round(ap, 5), flush=True)
    best_by_mode = {m: max((r for r in model_records if r["mode"] == m),
                           key=lambda r: (r["metrics"]["recovered"], r["residual_average_precision"])) for m in MODELS}
    chosen = max(best_by_mode.values(), key=lambda r: (r["metrics"]["recovered"], r["residual_average_precision"]))
    catalog = HERE / f"{name}_models.joblib"
    joblib.dump({r["key"]: model_objects[r["key"]] for r in best_by_mode.values() if r["key"] in model_objects},
                catalog, compress=3)
    recipe = dict(chosen=chosen, by_mode=best_by_mode, candidates=model_records, model_sha256=checksum(catalog),
                  n_fit=len(use), n_calibration=len(calibration), seconds=time.perf_counter()-t)
    write_json(HERE / f"{name}_policy.json", recipe)


@lru_cache(maxsize=2)
def load_policy(name):
    recipe = json.loads((HERE / f"{name}_policy.json").read_text())
    catalog = HERE / f"{name}_models.joblib"
    if checksum(catalog) != recipe["model_sha256"]:
        raise ValueError("MAT heads changed after calibrating thresholds")
    return recipe, joblib.load(catalog)


def score_view(texts, choices, heads):
    data = extract(texts)
    variables = vectors(data)
    output = {}
    for mode, item in choices.items():
        point = data[1].astype(np.float32) if mode == "soft_boundary" else heads[item["key"]].predict_proba(view(mode, variables))[:, 1]
        output[mode] = np.where(data[0], 2., point)
    return data[0], output


def score_one(texts, mode, model):
    requires_code = mode in {"code_only", "masked_comment_code", "combined"}
    base, stem, features, comments, code = extract(texts, requires_code)
    if mode == "soft_boundary":
        probability = stem.astype(np.float32)
    else:
        markers = GEOMETRY.transform(features) if mode in {"marker_geometry", "comment_words", "combined"} else None
        words = COMMENT.transform(comments) if mode in {"comment_words", "combined"} else None
        code_vec = CODE.transform(code) if requires_code else None
        probability = model.predict_proba(view(mode, (markers, words, code_vec)))[:, 1]
    return np.where(base, 2., probability)


def heldout():
    texts, y, projects, _ = dataset()
    groups = split()
    policy, weights = load_policy("project")
    choices = policy["by_mode"]
    records = {}
    for group in ("test_a", "test_b"):
        ids = groups[group]
        label = y[ids, 1].astype(bool)
        base, scores = score_view([texts[i] for i in ids], choices, weights)
        budget = int((base & ~label).sum())
        records[group] = {}
        for mode, score in scores.items():
            threshold = cut_at_budget(score, label, budget)
            same = score > threshold
            fixed = score > choices[mode]["threshold"]
            result = dict(exact=evaluate(label, base, same), frozen=evaluate(label, base, fixed),
                          exact_threshold=threshold, frozen_threshold=choices[mode]["threshold"],
                          by_project={p: evaluate(label[m], base[m], same[m]) for p in sorted(set(projects[ids]))
                                      if (m := projects[ids] == p).any()})
            records[group][mode] = result
            print(group, mode, result["exact"], result["frozen"], flush=True)
            with (HERE / f"{group}_{mode}.csv").open("w", newline="") as stream:
                w = csv.writer(stream)
                w.writerow(("row", "project", "label", "base", "score", "fixed", "matched"))
                w.writerows(zip(ids, projects[ids], label.astype(int), base.astype(int), score,
                                fixed.astype(int), same.astype(int), strict=True))
    write_json(HERE / "project_results.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "test", "official_fit"))
    stage = parser.parse_args().stage
    if stage == "test":
        heldout()
        return
    texts, y, _, _ = dataset()
    if stage == "fit":
        parts = split()
        candidates(texts, y, parts["fit"], parts["dev"])
    else:
        from pipeline import select_indices
        fit, held = select_indices(False)
        with np.load(ROOT / "round8/frozen/holdout.npz") as f:
            if held != f["rows"].tolist():
                raise ValueError("Official internal calibration order changed")
            calibration = f["rows"][f["folds"] == 3]
        candidates(texts, y, np.asarray(fit), calibration, name="official")


if __name__ == "__main__":
    main()
