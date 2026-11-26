"""Audit paired predictions and summarize the shared-task portfolio."""
import csv
import hashlib
import json
from statistics import median

import numpy as np

from protocol import HERE, ROOT, split
from pipeline import PUBLIC, SPLIT, TRAIN, load, metric, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula

RUNS = HERE / "runs"
ROUND4 = ROOT / "round4"
LABELS = ("vulnerability", "MAT")
PILOT = ("sparse", "sparse_roles", "forest", "memory", "byte", "codebert", "graphcodebert", "codet5p")


def read(path):
    return json.loads(path.read_text())


def predict_bits(path, ids, config):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if ids != [int(row["row"]) for row in rows]:
        raise ValueError(f"Unexpected rows in {path}")
    prob = np.asarray([[float(r[f"p_{label}"]) for label in LABELS] for r in rows])
    result = prob > config["thresholds"]
    if "pred_vulnerability" in reader.fieldnames:
        bits = np.asarray([[int(r[f"pred_{label}"]) for label in LABELS] for r in rows])
        if not np.array_equal(result, bits):
            raise ValueError(f"Stored predictions do not match thresholds in {path}")
    return result


def verify(y, p, receipt):
    scores = metric(y, p)
    for name in scores:
        for key, value in scores[name].items():
            if abs(value - receipt["scores"][name][key]) > 1e-10:
                raise ValueError(f"Unexpected result for {name} {key}: {value}")


def changes(y, a, b):
    return {
        label: dict(rescued=int(((a[:,j] != y[:,j]) & (b[:,j] == y[:,j])).sum()),
                    spoiled=int(((a[:,j] == y[:,j]) & (b[:,j] != y[:,j])).sum()),
                    extra_tp=int(((~a[:,j]) & b[:,j] & (y[:,j] == 1)).sum()),
                    extra_fp=int(((~a[:,j]) & b[:,j] & (y[:,j] == 0)).sum()),
                    removed_tp=int((a[:,j] & (~b[:,j]) & (y[:,j] == 1)).sum()),
                    removed_fp=int((a[:,j] & (~b[:,j]) & (y[:,j] == 0)).sum()))
        for j, label in enumerate(LABELS)
    }


def group_codes(keys):
    seen = {}
    return [seen.setdefault(key, len(seen)) for key in keys]


def pilot(train_hashes):
    _, labels, _ = load(TRAIN)
    ids, cal = split()
    y = labels[ids][~cal]
    groups = group_codes(train_hashes[i] for i, use in zip(ids, ~cal) if use)
    rows = {}
    predictions = {}
    for name in PILOT:
        run = RUNS / "pilot" / name
        receipt = read(run / "metrics.json")
        bits = predict_bits(run / "predictions.csv", ids, receipt)[~cal]
        verify(y, bits, receipt)
        predictions[name] = bits
        rows[name] = receipt
    pairs = (("sparse", "sparse_roles"), ("codebert", "graphcodebert"), ("codebert", "codet5p"),
             ("sparse_roles", "byte"), ("codebert", "sparse_roles"))
    paired = {f"{b}_minus_{a}": dict(changes=changes(y, predictions[a], predictions[b]),
                                   delta95=bootstrap_delta(y, predictions[a], predictions[b], groups))
              for a, b in pairs}
    with (HERE / "pilot_table.csv").open("w", newline="") as file:
        out = csv.writer(file)
        out.writerow(("family", "vulnerability_F1", "MAT_F1", "mean_F1", "runtime_s", "platform",
                      "dense_GFLOPs_if_applicable", "bytes"))
        for name, r in rows.items():
            cost = r["cost"]
            timing = cost.get("inference_seconds", cost.get("forward_and_pack_seconds", 0)
                              + cost.get("tokenize_seconds", 0))
            out.writerow((name, *(r["scores"][label]["f1"] for label in (*LABELS, "mean")), timing,
                          cost["platform"], cost.get("analytic_dense_gflops"), cost["bytes"]))
    return dict(models=rows, paired=paired, rows=len(y))


def public(train_hashes):
    texts, y, fields = load(PUBLIC, audit=True)
    ids = list(range(len(y)))
    groups = group_codes(zip(fields["Projectname"], fields["Filepath"]))
    window_predictions = np.load(ROUND4 / "public_predictions.npz")
    if not np.array_equal(window_predictions["y"], y):
        raise ValueError("Public labels differ from audited dataset")
    r = read(ROUND4 / "public_results.json")
    variants = r["variants"]
    bits = {name: window_predictions[name] > value["thresholds"] for name, value in variants.items()}
    old_heads = bits["control"]
    sparse_path = RUNS / "full/sparse_roles/public"
    sparse = read(sparse_path / "metrics.json")
    bits["sparse_roles"] = predict_bits(sparse_path / "predictions.csv", ids, sparse)
    verify(y, bits["sparse_roles"], sparse)
    for name, value in variants.items():
        verify(y, bits[name], value)
    check = ROUND4 / "entry_verify/gpu_full"
    verify_receipt = read(check / "metrics.json")
    deployed = predict_bits(check / "predictions.csv", ids, verify_receipt)
    if not np.array_equal(deployed, bits["score_candidate"]):
        raise ValueError("The local entry implementation differs from the selected window")
    t5_root = RUNS / "full/codet5p"
    t5_runs = t5_root / "portable_results"
    t5 = {}
    selection_file = t5_runs / "selected.json"
    if selection_file.exists():
        selection = read(selection_file)
        if selection != read(t5_root / "portable/frozen.json"):
            raise ValueError("The CodeT5+ selection differs from the portable configuration")
        for mode in ("head", "comment"):
            run = t5_runs / mode / "public"
            result = read(run / "metrics.json")
            if result["thresholds"] != selection["thresholds"][mode]:
                raise ValueError("CodeT5+ public evaluation did not use the frozen thresholds")
            name = f"t5_{mode}"
            bits[name] = predict_bits(run / "predictions.csv", ids, result)
            verify(y, bits[name], result)
            cost = result["cost"]
            t5[name] = dict(**result, forward_seconds_median=cost["forward_seconds_median"],
                            wall_seconds_median=median(cost["runtime_seconds"]),
                            analytic_gflops=cost["analytic_dense_gflops"], extra_windows=cost["extra_windows"],
                            formula_using_local_time_and_analytic_gflops=competition_formula(
                                result["scores"]["mean"]["f1"], cost["forward_seconds_median"],
                                cost["analytic_dense_gflops"]))
        for baseline, candidate in (("t5_head", "t5_comment"), ("score_candidate", f"t5_{selection['mode']}")):
            before, after = bits[baseline], bits[candidate]
            t5[f"{candidate}_minus_{baseline}"] = dict(changes=changes(y, before, after),
                                                       delta95=bootstrap_delta(y, before, after, groups))
        t5["frozen"] = selection
        t5["full_bf16_head"] = read(t5_root / "public/metrics.json")
        t5["notebook_profile_smoke"] = {mode: read(t5_root / "profile_smoke" / mode / "public/metrics.json")["cost"]
                                         for mode in ("head", "comment")}
    reference_keys = set(train_hashes.values())
    novel = np.asarray([hashlib.sha256(text.encode()).hexdigest() not in reference_keys for text in texts])
    masks = {"all": np.ones(len(y), dtype=bool), "fits_in_head": window_predictions["lengths"] <= 512,
             "long": window_predictions["lengths"] > 512, "comment_outside_head": window_predictions["missing_comment"],
             "no_exact_train_text": novel}
    differences = {}
    for name, pred in bits.items():
        if name != "control":
            gain = metric(y, pred)["mean"]["f1"] - variants["control"]["scores"]["mean"]["f1"]
            time_budget = None
            if name in variants:
                extra_ops = variants[name]["analytic_gflops"] - variants["control"]["analytic_gflops"]
                time_budget = (0.6 * gain - 0.2 * extra_ops / 3e6) / 0.0002
            differences[name] = dict(delta95=bootstrap_delta(y, old_heads, pred, groups),
                                     changes=changes(y, old_heads, pred), additional_seconds_to_break_even=time_budget)
    cohort = {name: dict(rows=int(mask.sum()), positives=y[mask].sum(axis=0).tolist(),
                         scores={k: metric(y[mask], p[mask]) for k, p in bits.items()})
              for name, mask in masks.items()}
    with (HERE / "public_table.csv").open("w", newline="") as file:
        out = csv.writer(file)
        out.writerow(("family", "vul_P", "vul_R", "vul_F1", "MAT_P", "MAT_R", "MAT_F1", "mean_P",
                      "mean_R", "mean_F1", "forward_s_median", "dense_GFLOPs", "extra_windows", "local_formula"))
        t5_models = {key: t5[key] for key in ("t5_head", "t5_comment") if key in t5}
        for name, receipt in {**variants, **t5_models, "sparse_roles": sparse}.items():
            out.writerow((name, *(receipt["scores"][label][m] for label in (*LABELS, "mean")
                                  for m in ("precision", "recall", "f1")),
                          receipt.get("forward_seconds_median"), receipt.get("analytic_gflops"),
                          receipt.get("extra_windows"), receipt.get("formula_using_local_time_and_analytic_gflops")))
    streaming = read(RUNS / "full/sparse_roles/streaming.json")
    profile = {k: read(ROUND4 / "entry_verify" / k / "metrics.json")["cost"]
               for k in ("profile_head", "profile_comment")}
    return dict(rows=len(y), verified_entry=verify_receipt, notebook_profile_smoke=profile,
                models=variants, codet5p=t5, sparse=sparse, sparse_streaming=streaming,
                sparse_zero_cost_score_cap=0.4 + 0.6*sparse["scores"]["mean"]["f1"],
                paired=differences, cohorts=cohort)


def main():
    with SPLIT.open(newline="") as file:
        hashes = {int(row["row"]): row["function_sha256"] for row in csv.DictReader(file)}
    report = dict(pilot=pilot(hashes), public=public(hashes))
    write_json(HERE / "analysis.json", report)
    print("Verified", report["pilot"]["rows"], "holdout and", report["public"]["rows"], "public rows")


if __name__ == "__main__":
    main()
