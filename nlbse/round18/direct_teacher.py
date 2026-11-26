"""Measure retained and complementary structural signal before distillation."""
import csv
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from structural_teacher import HERE, ROOT, OUT, checksum, inputs, json_at, write_json
from view_eval import offline, pinned, selected_views
from experiment import comparison, strict_budget
from evaluate import counts, exact_budget, raw


RECIPE = ("neural", "token", "ast", "mean", "stack", "blend_25", "blend_50", "blend_75")


def load_role(name, role, y, manifest):
    path = OUT / f"{name}.npz"
    if checksum(path) != manifest["holdout_sha256"][name]:
        raise ValueError("Structural scores have moved")
    with np.load(path) as f:
        if not np.array_equal(f["ids"], role) or not np.array_equal(f["y"], y):
            raise ValueError("Structural scores have a different row order or labels")
        return f["token"], f["ast"], f["combined"]


def examine():
    text, labels, projects, _, roles = inputs()
    route, _ = pinned()
    archived_cut = json_at(ROOT / "round13/study/policy.json")["compact_threshold"]
    teacher = json_at(OUT/"fit.json")
    start = time.perf_counter()
    entries = {}
    for name in ("probe", "dev", "test"):
        ids = roles[name]
        label = labels[ids, 0]
        predictions, _ = offline("projects", 42, ("reference",), [text[i] for i in ids])
        base, routed = selected_views(*predictions["reference"], route)
        token, ast, average = load_role(name, ids, labels[ids], teacher)
        entries[name] = dict(rows=ids, label=label, base=base, token=token, ast=ast, mean=average)
        print(name, len(ids), int(routed.sum()), round(time.perf_counter()-start, 2), flush=True)

    def encode(*p):
        return logit(np.column_stack(p).clip(1e-5, 1-1e-5))

    probe = entries["probe"]
    head = LogisticRegression(C=.3, max_iter=300, random_state=42)
    head.fit(encode(probe["base"], probe["token"], probe["ast"]), probe["label"])
    settings = {}
    output = {}
    folder = HERE/"direct_teacher"
    folder.mkdir(exist_ok=True)
    for role in ("dev", "test"):
        item = entries[role]
        ids, y, base, token, ast, mean = (item[k] for k in
                                         ("rows", "label", "base", "token", "ast", "mean"))
        budget = counts(y, raw(role)["compact"] > archived_cut)["fp"]
        reference, _ = exact_budget(base, y, budget)
        blend = encode(base, mean)
        stack = head.predict_proba(encode(base, token, ast))[:, 1]
        probs = dict(neural=base, token=token, ast=ast, mean=mean, stack=stack)
        for percent in (25, 50, 75):
            probs[f"blend_{percent}"] = expit(blend @ [1-percent/100, percent/100])
        output[role] = {}
        for mode in RECIPE:
            p = probs[mode]
            hit, threshold = exact_budget(p, y, budget)
            _, strict = strict_budget(p, y, budget)
            result = dict(equal=comparison(y, reference, hit), matched_cutoff=threshold)
            if role == "dev":
                settings[mode] = strict
            else:
                result["frozen"] = comparison(y, base > settings["neural"], p > settings[mode])
            output[role][mode] = result
            with (folder/f"{role}_{mode}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("row", "project", "label", "base_score", "score", "before_equal", "after_equal"))
                writer.writerows(zip(ids, projects[ids], y, base, p, reference.astype(int), hit.astype(int), strict=True))
        print(role, budget, {k: v["equal"]["delta_tp"] for k, v in output[role].items()}, flush=True)
    candidate = max(RECIPE, key=lambda k: (output["dev"][k]["equal"]["delta_tp"], k == "neural"))
    write_json(folder/"results.json", dict(selected=candidate, thresholds=settings, coef=head.coef_.tolist(),
               intercept=head.intercept_.tolist(), teacher_hash=teacher["models_sha256"], **output))


if __name__ == "__main__":
    examine()
