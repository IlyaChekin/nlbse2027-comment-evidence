"""Measure C/C++ comment recovery and markerless function-shape features."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "round14"))
from audit import DEBT, LEXEME, STEM, TAG, dataset, split
from mat import cut_at_budget, evaluate, example
from project_split import checksum
from pipeline import write_json


SPLICE = re.compile(r"\\(?:\r\n|[\n\r])")
WORDS = re.compile(r"[A-Za-z]+")
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")
TOKEN_SHAPE = re.compile(r"\b(?:if|else|for|while|switch|case|goto|return|break|continue|assert)\b")
DEBT_WORDS = re.compile(r"(?i)\b(?:remove|replace|simplify|future|workaround|temporary|retry|dummy|stub|unimplemented|broken|unsupported|deprecated|compatibility|until|should|needs?)\b")
ISSUE = re.compile(r"(?i)(?:\b(?:bug|issue|ticket)\s*[#: -]?\s*\d|https?://|\b[A-Z]{2,}-\d{2,}\b)")
DISABLED = re.compile(r"(?m)^\s*#\s*(?:if\s+0\b|error\b|warning\b)")
FALLBACK = re.compile(r"\b(?:fallback|dummy|stub|unsupported|unimplemented|temporary|deprecated)\b", re.I)
CONST_RETURN = re.compile(r"\breturn\s*(?:[-+]?\s*\d+|false|true|nullptr|NULL|[A-Z_][A-Z_0-9]*)\s*;")
FIXES = ("phase2", "unicode")
FLOW = ("if", "else", "for", "while", "switch", "case", "goto", "return", "break", "continue", "assert")
RULES = ("baseline", *FIXES)


def bodies(text):
    return [m for m in LEXEME.finditer(text) if m[0].startswith(("//", "/*"))]


def tagged(text):
    return any(TAG.search(text, *m.span()) is not None for m in bodies(text))


@lru_cache(maxsize=2048)
def near_marker(word):
    if len(word) not in (3, 4, 5, 6) or word in ("todo", "fixme", "xxx", "hack"):
        return False
    for marker in ("todo", "fixme", "xxx", "hack"):
        if len(word) == len(marker):
            diff = [i for i, (a, b) in enumerate(zip(word, marker, strict=True)) if a != b]
            if len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0]+1
                                  and word[diff[0]:diff[1]+1] == marker[diff[0]:diff[1]+1][::-1]):
                return True
        else:
            short, long = sorted((word, marker), key=len)
            if len(long) == len(short)+1 and any(long[:i]+long[i+1:] == short for i in range(len(long))):
                return True
    return False


def features(text):
    # Keep line/column geometry in the unmodified view. Test lexical recovery independently.
    base, legacy, comment, code = example(text)
    spans = bodies(text) if comment or legacy["has_comment"] else []
    phase2 = not base and "\\" in text and SPLICE.search(text) is not None and tagged(SPLICE.sub("", text))
    normalized = phase2
    if not base and not normalized and not text.isascii():
        normalized = any(TAG.search(unicodedata.normalize("NFKC", m[0])) is not None for m in spans)
    length = max(1, len(text))
    first_body = code.find("{")
    first_text_brace = text.find("{")
    in_body = [m for m in spans if first_text_brace >= 0 and m.start() > first_text_brace]
    starts = [m.start()/length for m in spans]
    lines = [part.lstrip(" \t*") for part in comment.splitlines()]
    head = [word.group().lower() for line in lines if (word := WORDS.match(line))]
    all_words = WORDS.findall(comment.lower())
    code_ids = IDENT.findall(code)
    flow = TOKEN_SHAPE.findall(code)
    chars = len(code)
    statements = code.count(";")
    anatomy = (
        bool(spans), len(spans), len(in_body), bool(text.lstrip().startswith(("//", "/*"))),
        len(comment), len(comment)/length, min(starts, default=1.), max(starts, default=0.),
        sum(not text[text.rfind("\n", 0, m.start())+1:m.start()].strip() for m in spans),
        sum(m[0].startswith("//") for m in spans), len(lines), len(all_words), len(set(all_words)),
        len(DEBT_WORDS.findall(comment)), sum(DEBT_WORDS.fullmatch(w) is not None for w in head),
        bool(DEBT.search(comment)), bool(ISSUE.search(comment)), bool(TAG.search(code)),
        sum(near_marker(w) for w in head), sum(len(w) >= 4 and near_marker(w) for w in all_words),
        sum(STEM.search(m[0]) is not None for m in spans), bool(legacy.get("split_tag")),
        sum(line.endswith(("?", "!")) for line in lines), comment.count(":"), comment.count("?"),
    )
    shape = (
        chars, code.count("\n")+1, len(code_ids), len(set(code_ids)), statements, first_body,
        code.count("{"), code.count("}"), code.count("#"), code.count("("),
        *[flow.count(word) for word in FLOW], len(CONST_RETURN.findall(code)),
        bool(DISABLED.search(code)), len(FALLBACK.findall(code)),
        sum(1 for w in code_ids if w.isupper()), sum(1 for w in code_ids if "_" in w),
        max(map(len, code_ids), default=0),
    )
    return base, phase2, normalized, anatomy, shape


def transform(texts):
    result = [features(text) for text in texts]
    base, splice, normalized, anatomy, shape = zip(*result, strict=True)
    return (np.asarray(base), np.asarray(splice), np.asarray(normalized),
            np.asarray(anatomy, dtype=np.float32), np.asarray(shape, dtype=np.float32))


def groups(name):
    if name == "project":
        return split()
    from pipeline import select_indices
    fit, held = select_indices(False)
    with np.load(ROOT / "round8/frozen/holdout.npz") as raw:
        if held != raw["rows"].tolist():
            raise ValueError("Official holdout moved")
        return {"fit": np.asarray(fit), "dev": raw["rows"][raw["folds"] == 3],
                "review": raw["rows"][raw["folds"] == 4]}


def prepare(texts, rows, key):
    selection = list(texts) if rows is None else [texts[i] for i in rows]
    start = time.perf_counter()
    result = transform(selection)
    print("features", key, len(selection), round(time.perf_counter()-start, 3), flush=True)
    return result


def matrices(data):
    joint = np.column_stack(data[3:])
    return dict(anatomy=data[3], shape=data[4], anatomy_shape=joint,
                erasure=data[4], erasure_plus_residual=joint)


def fit_heads(data, label):
    base = data[0]
    heads = {}
    for key, x in matrices(data).items():
        if key == "erasure":
            # Remove all comments, then ask whether the function used to carry a visible marker.
            mask, target = np.ones(len(base), dtype=bool), label & base
        elif key == "erasure_plus_residual":
            mask, target = np.ones(len(base), dtype=bool), label
            # Remove all comment features for visible markers, so they cannot identify positives.
            x = x.copy()
            x[base, :data[3].shape[1]] = 0
        else:
            mask, target = ~base, label
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                               learning_rate=.08, min_samples_leaf=55,
                                               l2_regularization=3., random_state=42,
                                               early_stopping=False)
        weights = np.where(target[mask], 12., 1.)
        if key == "erasure_plus_residual":
            weights[base[mask]] *= .14
        model.fit(x[mask], target[mask], sample_weight=weights)
        heads[key] = model
        print("fit", key, int(mask.sum()), int(target[mask].sum()), flush=True)
    return heads


def scores(data, heads):
    base, splice, normalized, _, _ = data
    out = {"baseline": 2.*base, "phase2": 2.*base+splice, "unicode": 2.*base+normalized}
    for key, matrix in matrices(data).items():
        probability = heads[key].predict_proba(matrix)[:, 1]
        out[key] = np.where(base, 2., probability)
    return out


def cutoff(key, score, y, b):
    if key in RULES:
        return .5
    return cut_at_budget(score, y, int((~y & b).sum()))


def records(data, predictions, label, thresholds=None):
    base, splice, normalized, anatomy, _ = data
    has_comment = anatomy[:, 0] > 0
    missed = label & ~base
    result = {}
    for key, p in predictions.items():
        t = cutoff(key, p, label, base)
        fixed_t = t if thresholds is None else thresholds[key]
        masks = {"fixed": p > fixed_t}
        if key in RULES:
            # Binary transformations have no learned ranking within ties.
            # Add the whole tied set only if it fits the fixed FP budget.
            allowed = ((~label & masks["fixed"]).sum() <= (~label & base).sum())
            t = .5 if allowed else 1.5
        masks["matched"] = p > t
        value = dict(fixed_threshold=fixed_t, matched_threshold=t,
                     residual_ap=float(average_precision_score(label[~base], p[~base])),
                     missed_with_comment=int((missed & has_comment).sum()),
                     missed_without_comment=int((missed & ~has_comment).sum()))
        for mode, bits in masks.items():
            value[mode] = evaluate(label, base, bits)
            value[mode]["recovered_with_comment"] = int((missed & has_comment & bits).sum())
            value[mode]["recovered_without_comment"] = int((missed & ~has_comment & bits).sum())
        result[key] = (value, masks)
    bias = {"new_phase2": int(splice.sum()), "new_unicode_or_phase2": int(normalized.sum()),
            "phase2_hits": int((label & splice).sum()), "unicode_or_phase2_hits": int((label & normalized).sum())}
    return result, bias


def fit(scope):
    texts, labels, _, _ = dataset()
    index = groups(scope)
    fit_ids, dev = index["fit"], index["dev"]
    fit_data = prepare(texts, fit_ids, "fit")
    heads = fit_heads(fit_data, labels[fit_ids, 1].astype(bool))
    dev_data = prepare(texts, dev, "dev")
    stats, bias = records(dev_data, scores(dev_data, heads), labels[dev, 1].astype(bool))
    catalog = HERE / f"{scope}.joblib"
    joblib.dump(heads, catalog, compress=3)
    points = {k: v for k, (v, _) in stats.items()}
    def priority(key):
        result = points[key]
        return (result["fixed"]["extra_fp"] <= 0,
                result["fixed"]["recovered"] - result["fixed"]["lost"],
                key == "baseline", result["residual_ap"])
    selected = max(points, key=priority)
    policy = dict(scope=scope, selected=selected, development=points, audit=bias,
                  training_rows=len(fit_ids), calibration_rows=len(dev), weights_sha256=checksum(catalog),
                  partition_sha256=checksum(ROOT/"round13/partition.npz"),
                  data_manifest_sha256=checksum(ROOT/"data/manifest.json"),
                  code_sha256=checksum(Path(__file__)))
    write_json(HERE / f"{scope}_policy.json", policy)
    print("selected", selected, "development", [(k, v["fixed"]["recovered"], v["fixed"]["extra_fp"])
                                               for k, v in points.items()], flush=True)


def snapshot(scope):
    path = HERE / f"{scope}_policy.json"
    policy = json.loads(path.read_text())
    catalog = HERE / f"{scope}.joblib"
    for key, target in (("weights_sha256", catalog), ("partition_sha256", ROOT/"round13/partition.npz"),
                        ("data_manifest_sha256", ROOT/"data/manifest.json"), ("code_sha256", Path(__file__))):
        if policy[key] != checksum(target):
            raise ValueError(f"Hash mismatch for {target}")
    return policy, joblib.load(catalog)


def test(scope, public=False):
    policy, heads = snapshot(scope)
    text, labels, projects, _ = dataset(public)
    if public:
        index = {"public": np.arange(len(text))}
        with (ROOT / "round16/context_public/full_public.csv").open(newline="") as file:
            prior = list(csv.DictReader(file))
        public_y = np.asarray([int(row["label_mat"]) for row in prior], dtype=bool)
        if len(prior) != len(text) or not np.array_equal(public_y, labels[:, 1]):
            raise ValueError("Round 16 reference has moved")
    else:
        parts = groups(scope)
        index = {name: parts[name] for name in ("test_a", "test_b")} if scope == "project" else {"review": parts["review"]}
    config = {k: r["fixed_threshold"] for k, r in policy["development"].items()}
    all_results = {}
    for group, ids in index.items():
        data = prepare(text, ids, group)
        if public and not np.array_equal(data[0], [int(row["mat"]) for row in prior]):
            raise ValueError("Original MAT head differs from the archive")
        gold = labels[ids, 1].astype(bool)
        predictions = scores(data, heads)
        results, audit = records(data, predictions, gold, config)
        all_results[group] = dict(audit=audit, methods={key: result for key, (result, _) in results.items()})
        for label in ("fixed", "matched"):
            summary = [(key, result[label]["recovered"], result[label]["extra_fp"])
                       for key, (result, _) in results.items()]
            print(group, label, summary, flush=True)
        with (HERE / f"{scope}_{group}.csv").open("w", newline="") as file:
            keys = list(predictions)
            fields = ["row", "project", "label_mat", "base_mat", "has_comment"]
            fields += [f"{key}_{kind}" for key in keys for kind in ("score", "fixed", "matched")]
            csvfile = csv.writer(file)
            csvfile.writerow(fields)
            project = projects[ids]
            for i, row in enumerate(ids):
                values = [row, project[i], int(gold[i]), int(data[0][i]), int(data[3][i, 0])]
                values.extend(value for k in keys for value in (predictions[k][i],
                              int(results[k][1]["fixed"][i]), int(results[k][1]["matched"][i])))
                csvfile.writerow(values)
    suffix = "public" if public else "test"
    write_json(HERE / f"{scope}_{suffix}.json", {"selected": policy["selected"], **all_results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "test", "public"))
    parser.add_argument("--scope", choices=("project", "official"), default="project")
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.scope)
    else:
        test(args.scope, public=args.stage == "public")


if __name__ == "__main__":
    main()
