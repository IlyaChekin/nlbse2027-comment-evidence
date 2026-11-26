"""Fit a four-block exit for CodeT5-small and defer inputs that need the full encoder."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.nn.functional import linear

from students import BATCH, dataset, group_codes, load, load_selected, metric, predict_bits, save, write_json
from students import PUBLIC, ROOT, batch, batch_gflops, batch_plan, ordered_batches
from student_context import FAMILY, HERE, RUN, feature_block, load_policy, probability
from frontier import as_common_scale, calibrated
from fused_attention import fused_block, padded_stride_bias, pool, prefix_fused
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT = HERE / "compact_exit"
EARLY = 4


def routed(p, extra, rule):
    uncertain = (p > rule["lower"]) & (p < rule["upper"])
    return uncertain | extra[:, :4].any(axis=1)


def logistic_head(x, y, strength):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), .025)
    model = LogisticRegression(C=strength, max_iter=700).fit((x-mean)/scale, y)
    weight = model.coef_[0] / scale
    return weight, float(model.intercept_[0] - weight @ mean)


@torch.inference_mode()
def evaluate(model, tokenizer, texts, rule=None, comment=None, *, deferred_offsets=False):
    n = len(texts)
    stack = model.encoder.encoder
    model.eval()
    collected = np.empty((n, model.config.d_model + 5), dtype=np.float32) if rule is None else None
    extra = np.empty((n, 16), dtype=np.float32)
    predictions = np.empty((n, 2), dtype=np.float32)
    costs = np.empty(n, dtype=np.float64)
    exits = 0
    bias_cache = {}
    forward = 0.
    start_time = time.perf_counter()
    if rule is not None:
        weight = torch.tensor(rule["weight"], device="cuda", dtype=torch.float32)[None, :]
        bias = torch.tensor([rule["bias"]], device="cuda", dtype=torch.float32)

    def prepare(offset):
        return feature_block(texts[offset:offset+2048], tokenizer, deferred_offsets=deferred_offsets)

    with torch.autocast("cuda", dtype=torch.float16):
        for offset, (seq, side, added) in ordered_batches(range(0, n, 2048), prepare, True):
            extra[offset:offset+len(seq)] = added
            side = torch.as_tensor(side, device="cuda")
            for ids in batch_plan(list(range(len(seq))), seq, BATCH):
                x, mask = batch(ids, seq, tokenizer.pad_token_id)
                at = offset + np.asarray(ids)
                step = time.perf_counter()
                h, position_bias = prefix_fused(model, x, mask, EARLY, bias_cache)
                f = torch.column_stack((pool(stack, h, mask), side[ids]))
                full = batch_gflops("codet5p", model, 1, x.shape[1])
                initial = full * EARLY / len(model.layers)
                costs[at] = full if rule is None else initial
                if rule is None:
                    collected[at] = f.float().cpu().numpy()
                    continue
                with torch.autocast("cuda", enabled=False):
                    p_early = linear(f.float(), weight, bias).sigmoid().flatten()
                    wants = (p_early > rule["lower"]) & (p_early < rule["upper"])
                    wants |= side[ids, :4].bool().any(dim=1)
                needed = wants.nonzero().flatten()
                chosen = needed.cpu().numpy()
                early = p_early.cpu().numpy()
                early_mat = probability(np.column_stack((early, early)), added[ids], comment["lexical_control"])
                outputs = as_common_scale(np.column_stack((early, early_mat)),
                                          [rule["cutoff"], comment["lexical_control"]["thresholds"][1]])
                if len(chosen):
                    width = max(len(seq[ids[i]]) for i in chosen)
                    late_h, late_mask = h[needed, :width], mask[needed, :width]
                    late_bias = padded_stride_bias(position_bias[needed, :, :width, :width])
                    for block in stack.block[EARLY:]:
                        late_h = fused_block(block, late_h, late_bias)
                    late = (model.head(pool(stack, late_h, late_mask)) + model.side(side[ids][needed]))
                    late = late.float().sigmoid().cpu().numpy()
                    late[:, 1] = probability(late, added[np.asarray(ids)[chosen]], comment["selected"])
                    outputs[chosen] = as_common_scale(late, comment["selected"]["thresholds"])
                    late_cost = batch_gflops("codet5p", model, 1, width) * (len(model.layers)-EARLY)/len(model.layers)
                    costs[at[chosen]] += late_cost
                predictions[at] = outputs
                exits += len(ids) - len(chosen)
                forward += time.perf_counter()-step
    if rule is None:
        return collected, extra, costs
    return predictions, dict(wall_seconds=time.perf_counter()-start_time, forward_seconds=forward,
                             analytic_gflops=float(costs.sum()), early_exits=exits)


def tune():
    comment = load_policy()
    model, tokenizer, _ = load_selected(FAMILY)
    _, _, texts, y, ids, _ = dataset(full=True)
    config = json.loads((RUN / "selected.json").read_text())
    stem = Path(config["checkpoint"]).stem
    with np.load(RUN / f"{stem}_holdout.npz") as cache:
        if ids != cache["ids"].tolist() or not np.array_equal(y, cache["y"]):
            raise ValueError("Holdout alignment differs")
        late, folds = cache["p"], cache["fold"]
    x, extra, cost = evaluate(model, tokenizer, texts)
    fit, cal, review = folds <= 2, folds == 3, folds == 4
    late[:, 1] = probability(late, extra, comment["selected"])
    late_bits = late > comment["selected"]["thresholds"]
    lexical = probability(late, extra, comment["lexical_control"]) > comment["lexical_control"]["thresholds"][1]
    candidates = []
    predicted = {}
    for strength in (.05, .3, 1.):
        weight, bias = logistic_head(x[fit], y[fit, 0], strength)
        p = (1 / (1 + np.exp(-np.clip(x @ weight + bias, -30, 30)))).astype(np.float32)
        threshold = calibrated(y[cal], np.column_stack((p[cal], late[cal, 1])))[0]
        predicted[strength] = dict(weight=weight.tolist(), bias=bias, cutoff=threshold)
        early = np.column_stack((p > threshold, lexical))
        for lower in (.01, .025, .05, .075, .1, .15, .2):
            for upper in (.72, .82, .89, .93, .96, .985):
                route = routed(p, extra, {"lower": lower, "upper": upper})
                bits = np.where(route[:, None], late_bits, early)
                f1 = metric(y[cal], bits[cal])["mean"]["f1"]
                relative = (cost[cal]*(EARLY+(len(model.layers)-EARLY)*route[cal])/len(model.layers)).sum()/cost[cal].sum()
                objective = .6*f1-.2*254812/3e6*relative
                candidates.append(dict(strength=strength, lower=lower, upper=upper, cal_f1=f1,
                                       cal_estimate=objective, review=metric(y[review], bits[review]),
                                       review_exit_rate=float((~route[review]).mean())))
    base = metric(y[cal], late_bits[cal])["mean"]["f1"]
    allowed = [r for r in candidates if r["cal_f1"] >= base-.004]
    chosen = max(allowed or candidates, key=lambda r: r["cal_estimate"])
    rule = dict(chosen, **predicted[chosen["strength"]])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "policy.json", dict(rule=rule, candidates=candidates,
               model_sha256=config["settings"]["sha256"], baseline_review=metric(y[review], late_bits[review]),
               baseline_calibration_f1=base))
    print("compact exit review", rule["review"], rule["review_exit_rate"], flush=True)


def pipeline(lexical=False):
    rule = json.loads((OUT / "policy.json").read_text())
    comment = load_policy()
    if rule["model_sha256"] != comment["selected_model_sha256"]:
        raise ValueError("The early readout weights refer to a different encoder")
    model, tokenizer, _ = load_selected(FAMILY)
    if lexical:
        comment["selected"] = comment["lexical_control"]
    return model, tokenizer, rule["rule"], comment


def public(repeat, lexical=False, deferred_offsets=False):
    model, tokenizer, rule, comment = pipeline(lexical)
    output = OUT / ("lexical_deferred" if deferred_offsets else "lexical" if lexical else "public")
    _, _, texts, labels, ids, cal = dataset(public=True)
    _, _, fields = load(PUBLIC, audit=True)
    evaluate(model, tokenizer, texts[:2048], rule, comment, deferred_offsets=deferred_offsets)
    reference, timings = None, []
    for _ in range(repeat):
        p, receipt = evaluate(model, tokenizer, texts, rule, comment, deferred_offsets=deferred_offsets)
        bits = p > .5
        if reference is not None and not np.array_equal(bits, reference):
            raise ValueError("Predictions differ between repetitions")
        reference = bits
        timings.append(receipt)
        print(receipt, flush=True)
    if deferred_offsets:
        prior_offsets = predict_bits(OUT / "lexical/predictions.csv", ids, {"thresholds": (.5, .5)})
        if not np.array_equal(reference, prior_offsets):
            raise ValueError("Deferred token offsets changed a decision")
    cost = {k: median(r[k] for r in timings) for k in timings[0]}
    save(output, labels, ids, cal, p, cost, (.5, .5))
    record = json.loads((output / "metrics.json").read_text())
    record["local_formula"] = competition_formula(record["scores"]["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])
    groups = group_codes(zip(fields["Projectname"], fields["Filepath"], strict=True))
    prior = predict_bits(ROOT / "round9/combined/balanced_length/predictions.csv", ids, {"thresholds": (.5, .5)})
    record["interval_vs_balanced"] = bootstrap_delta(labels, prior, reference, groups)
    record["repeats"] = timings
    write_json(output / "analysis.json", record)


def predict(source, target, *, lexical=False, deferred_offsets=False):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV requires a Function column")
        texts = [row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Every row must supply a function")
    model, tokenizer, rule, comment = pipeline(lexical)
    p, _ = evaluate(model, tokenizer, texts, rule, comment, deferred_offsets=deferred_offsets)
    with target.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("vulnerability", "MAT"))
        writer.writerows((p > .5).astype(int).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "public", "predict"))
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--lexical", action="store_true")
    parser.add_argument("--deferred-offsets", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.deferred_offsets and not args.lexical:
        parser.error("--deferred-offsets requires --lexical")
    if args.stage == "tune":
        tune()
    elif args.stage == "public":
        public(args.repeat, args.lexical, args.deferred_offsets)
    else:
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        predict(args.input, args.output, lexical=args.lexical, deferred_offsets=args.deferred_offsets)
