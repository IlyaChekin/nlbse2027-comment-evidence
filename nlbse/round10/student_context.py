"""Calibrate a full-comment MAT readout for the compact multitask encoder."""
import argparse
import csv
import json
from pathlib import Path
from functools import partial
from statistics import median

import numpy as np
from scipy.special import expit, logit
from transformers import AutoTokenizer

from students import (CAP, FAMILIES, HERE, PUBLIC, ROOT, dataset, group_codes,
                      load, load_selected, metric, predict_bits, save, scores, write_json)
from comment_spans import LEXEME
from frontier import TAG, TAGS, calibrated
from round2_report import bootstrap_delta
from score_runtime import competition_formula

FAMILY = "t5_small"
RUN = HERE / "runs" / FAMILY
OUTPUT = HERE / "student_context"
FEATURES = 16


def folder(family):
    return OUTPUT if family == FAMILY else HERE / "unix_context"


def feature_block(texts, tokenizer, *, family=FAMILY, deferred_offsets=False):
    unix = family == "unixcoder9"
    content_cap = CAP - 4 if unix else CAP
    data = tokenizer(texts, truncation=True, max_length=content_cap,
                     add_special_tokens=not unix, return_offsets_mapping=not deferred_offsets)
    sequences = data["input_ids"]
    if unix:
        role = tokenizer.convert_tokens_to_ids("<encoder-only>")
        if role == tokenizer.unk_token_id:
            raise ValueError("Official UniXcoder encoder-only token is missing")
        prefix = [tokenizer.bos_token_id, role, tokenizer.eos_token_id]
        sequences = [[*prefix, *row, tokenizer.eos_token_id] for row in sequences]
    side = np.zeros((len(texts), 5), dtype=np.float32)
    side[:, 4] = np.log1p([len(t) for t in texts])
    extra = np.zeros((len(texts), FEATURES), dtype=np.float32)
    for i, text in enumerate(texts):
        if not TAG.search(text):
            continue
        if deferred_offsets:
            offsets = data.encodings[i].offsets
        else:
            offsets = data["offset_mapping"][i]
        limit = max(end for _, end in offsets) if len(offsets) == content_cap else len(text)
        for chunk in LEXEME.finditer(text):
            if not chunk[0].startswith(("//", "/*")):
                continue
            for tag in TAG.finditer(text, *chunk.span()):
                key = TAGS[tag[0].lower()]
                side[i, key] = 1
                extra[i, key] += 1
                extra[i, 4 + key] += tag.start() >= limit
                extra[i, 8 + key] += not text[chunk.start()+2:tag.start()].strip(" \t\r\n*")
                extra[i, 12 + key] += tag[0].isupper()
    return sequences, side, np.log1p(extra)


def features(p, extra, variant):
    raw = logit(np.clip(p[:, 1:2], 1e-6, 1-1e-6))
    if variant == "tags":
        return np.column_stack((raw, extra[:, :4] > 0))
    if variant == "location":
        return np.column_stack((raw, extra[:, :8]))
    if variant == "structured":
        return np.column_stack((raw, extra))
    return extra


def probability(p, extra, record):
    x = features(p, extra, record["features"])
    return expit(x @ np.asarray(record["weight"], dtype=np.float32) + record["bias"])


def tune(family=FAMILY):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    run = HERE / "runs" / family
    output_folder = folder(family)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / FAMILIES[family], local_files_only=True)
    _, _, texts, y, ids, _ = dataset(full=True)
    selection = json.loads((run / "selected.json").read_text())
    stem = Path(selection["checkpoint"]).stem
    reference = selection["settings"]["thresholds"]
    with np.load(run / f"{stem}_holdout.npz") as cache:
        if ids != cache["ids"].tolist() or not np.array_equal(y, cache["y"]):
            raise ValueError("Student and target features are misaligned")
        p, folds = cache["p"], cache["fold"]
    extra = np.concatenate([feature_block(texts[i:i+1024], tokenizer, family=family)[2] for i in range(0, len(texts), 1024)])
    fit, calibration, review = folds <= 2, folds == 3, folds == 4
    records = []
    for variant in ("tags", "location", "structured", "lexical_only"):
        x = features(p, extra, variant)
        scale = np.maximum(x[fit].std(0), .025)
        for c in (.03, .15, .6):
            classifier = LogisticRegression(C=c, max_iter=700).fit(x[fit] / scale, y[fit, 1])
            record = dict(features=variant, inverse_regularization=c,
                          weight=(classifier.coef_[0] / scale).tolist(), bias=float(classifier.intercept_[0]))
            joint = np.column_stack((p[:, 0], probability(p, extra, record)))
            cutoff = calibrated(y[calibration], joint[calibration])[1]
            decisions = joint > (reference[0], cutoff)
            record.update(thresholds=[reference[0], cutoff],
                          calibration_ap=average_precision_score(y[calibration, 1], joint[calibration, 1]),
                          calibration=metric(y[calibration], decisions[calibration]),
                          review=metric(y[review], decisions[review]))
            records.append(record)
    def calibration_key(record):
        return record["calibration"]["MAT"]["f1"], record["calibration_ap"]

    main = max((r for r in records if r["features"] != "lexical_only"), key=calibration_key)
    ablation = max((r for r in records if r["features"] == "lexical_only"), key=calibration_key)
    old = metric(y[review], p[review] > reference)
    comment_outside = extra[review, 4:8].any(axis=1)
    output = dict(selected=main, lexical_control=ablation, candidates=records, baseline_review=old,
                  selection_rows=[int(s.sum()) for s in (fit, calibration, review)],
                  selected_model_sha256=selection["settings"]["sha256"],
                  review_rows_with_omitted_tag=int(comment_outside.sum()))
    output_folder.mkdir(parents=True, exist_ok=True)
    write_json(output_folder / "policy.json", output)
    print("review MAT", old["MAT"]["f1"], main["features"], main["review"]["MAT"]["f1"],
          "lexical", ablation["review"]["MAT"]["f1"], flush=True)


def load_policy(family=FAMILY):
    policy = json.loads((folder(family) / "policy.json").read_text())
    config = json.loads((HERE / "runs" / family / "selected.json").read_text())
    if policy["selected_model_sha256"] != config["settings"]["sha256"]:
        raise ValueError("The comment readout was fitted with a different model")
    return policy


def runner(model, tokenizer, texts, record=None, fused=True, family=FAMILY):
    def apply(p, x):
        p[:, 1] = probability(p, x, record)

    kwargs = {} if record is None else {"prepare": partial(feature_block, family=family), "readout": apply}
    return scores(model, tokenizer, texts, family, fused=fused and family == FAMILY, **kwargs)


def public(repetitions, family=FAMILY):
    policy = load_policy(family)
    model, tokenizer, t = load_selected(family)
    output_folder = folder(family)
    _, _, texts, labels, ids, cal = dataset(public=True)
    _, _, fields = load(PUBLIC, audit=True)
    prefix = "student" if family == FAMILY else "unix"
    correction = f"{prefix}_context"
    options = {correction: policy["selected"], f"{prefix}_lexical": policy["lexical_control"]}
    if family == FAMILY:
        options = {"student_fused": None, **options}
    results = {}
    for name, record in options.items():
        cuts = t if record is None else record["thresholds"]
        runner(model, tokenizer, texts[:2048], record, family=family)
        timing, reference = [], None
        for _ in range(repetitions):
            p, cost = runner(model, tokenizer, texts, record, family=family)
            if reference is not None and not np.array_equal(reference, p > cuts):
                raise ValueError("Different binary outcomes across runs")
            reference = p > cuts
            timing.append(cost)
            print(name, cost, flush=True)
        medians = {k: median(x[k] for x in timing) for k in timing[0]}
        save(output_folder / name, labels, ids, cal, p, medians, cuts)
        results[name] = json.loads((output_folder / name / "metrics.json").read_text())
        results[name]["local_formula"] = competition_formula(results[name]["scores"]["mean"]["f1"],
                                     medians["forward_seconds"], medians["analytic_gflops"])
        results[name]["repeats"] = timing
    group_ids = group_codes(zip(fields["Projectname"], fields["Filepath"], strict=True))
    old = predict_bits(HERE / "runs" / family / "public/predictions.csv", ids, {"thresholds": t})
    anchor = predict_bits(ROOT / "round9/combined/balanced_length/predictions.csv", ids, {"thresholds": (.5, .5)})
    for name, run in results.items():
        bits = predict_bits(output_folder / name / "predictions.csv", ids, {"thresholds": run["thresholds"]})
        run["changes_vs_eager"] = np.count_nonzero(old != bits, axis=0).tolist()
        run["delta_vs_eager"] = bootstrap_delta(labels, old, bits, group_ids)
        run["delta_vs_balanced"] = bootstrap_delta(labels, anchor, bits, group_ids)
    bits = predict_bits(output_folder / correction / "predictions.csv", ids,
                              {"thresholds": policy["selected"]["thresholds"]})
    extra = np.concatenate([feature_block(texts[i:i+1024], tokenizer, family=family)[2] for i in range(0, len(texts), 1024)])
    outside = extra[:, 4:8].any(axis=1)
    rescued = (labels[:, 1] == 1) & ~old[:, 1] & bits[:, 1]
    added_false = (labels[:, 1] == 0) & ~old[:, 1] & bits[:, 1]
    results["comment_breakdown"] = dict(omitted_tag_rows=int(outside.sum()),
        positives_recovered=int(rescued.sum()), recovered_with_omitted_tag=int((rescued & outside).sum()),
        new_false_positives=int(added_false.sum()))
    write_json(output_folder / "analysis.json", results)


def predict(source, target, fused, *, lexical=False, family=FAMILY):
    policy = load_policy(family)["lexical_control" if lexical else "selected"]
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV requires a Function column")
        texts = [row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Every row must supply a function")
    model, tokenizer, _ = load_selected(family)
    p, _ = runner(model, tokenizer, texts, policy, fused=fused, family=family)
    with target.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("vulnerability", "MAT"))
        writer.writerows((p > policy["thresholds"]).astype(int).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "public", "predict"))
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--family", choices=FAMILIES, default=FAMILY)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.stage == "tune":
        tune(args.family)
    elif args.stage == "public":
        public(args.repeat, args.family)
    else:
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        predict(args.input, args.output, not args.eager, family=args.family)
