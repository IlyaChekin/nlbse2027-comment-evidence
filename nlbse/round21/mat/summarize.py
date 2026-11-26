"""Recount decisions and translate the remaining MAT errors into score scenarios."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def counts(data, truth, pred):
    pair = [(r[truth] == "1", r[pred] == "1") for r in data]
    return tuple(sum(test(row) for row in pair) for test in (
        lambda x: x[0] and x[1], lambda x: not x[0] and x[1], lambda x: x[0] and not x[1],
    ))


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def check_annotation(policy):
    audit = json.loads((HERE / "annotation_results.json").read_text())
    key = policy["selected"]
    result = []
    for split in ("dev", "test_a", "test_b", "public"):
        rows = read(HERE / f"annotation_{split}.csv")
        if len(rows) != audit[split]["rows"]:
            raise ValueError(f"Unexpected row count for {split}")
        base = counts(rows, "gold", "base")
        after = counts(rows, "gold", f"{key}_frozen")
        for family in {r["project"] for r in rows}:
            project = [r for r in rows if r["project"] == family]
            if counts(project, "gold", "base")[1] != counts(project, "gold", f"{key}_frozen")[1]:
                raise ValueError(f"FP changed in {split}/{family}")
        target = audit[split]["metrics"][key]["frozen"]
        if any(x != tuple(target[n][c] for c in ("tp", "fp", "fn"))
               for n, x in (("base", base), ("candidate", after))):
            raise ValueError(f"Stored audit totals differ for {split}")
        result.append((split, len(rows), base[2], after[0]-base[0], base[1], after[1],
                       f1(*base), f1(*after), target["recovered_by_word_join"],
                       target["recovered_by_lexical_suffix_prefix"]))
    with (HERE / "comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("split", "rows", "baseline_fn", "extra_tp", "baseline_fp", "final_fp",
                    "baseline_mat_f1", "final_mat_f1", "word_join_tp", "attached_marker_tp"))
        w.writerows(result)


def scenarios(profile):
    data = read(HERE / "profile_marker.csv")
    base = read(HERE / "profile_reference.csv")
    ref, final = profile["reference"], profile["marker"]
    for n, sample in (("reference", base), ("marker", data)):
        quality = profile[n]["quality"]
        for t, cols in (("vulnerability", ("label_vulnerability","pred_vulnerability")),
                        ("MAT", ("label_mat","pred_MAT"))):
            if abs(f1(*counts(sample, *cols)) - quality[t]["f1"]) > 1e-12:
                raise ValueError(f"Unexpected F1 for {n}/{t}")
    if any(a["row"] != b["row"] or a["pred_vulnerability"] != b["pred_vulnerability"]
           for a, b in zip(data, base, strict=True)):
        raise ValueError("The marker must reproduce all vulnerability decisions")
    mat = counts(data, "label_mat", "pred_MAT")
    vul = counts(data, "label_vulnerability", "pred_vulnerability")
    original = json.loads((ROOT / "round14/audit_public.json").read_text())["public"]["false_negative_scope"]
    joined = mat[0] - counts(base, "label_mat", "pred_MAT")[0]
    visible = original["comment_without_base_tag"] - joined
    if mat[2] != visible + original["no_parsed_comment"]:
        raise ValueError("The remaining false negatives do not match the prior audit")
    target = .97
    def projected(k):
        return final["score"] + .3 * (f1(mat[0]+k, mat[1], mat[2]-k) - f1(*mat))
    required = next((k for k in range(mat[2]+1) if projected(k) >= target), None)
    need_vulnerability = f1(*vul) + (target-final["score"])/.3
    v_total = vul[0]+vul[2]
    best_vulnerability = f1(v_total, vul[1], 0)
    fp_ceiling = int((2*v_total / need_vulnerability) - 2*v_total)
    source = json.loads((ROOT/"round18/mat_residual/source_audit_public.json").read_text())
    observations = source["observations"]
    result = dict(
        baseline_score=ref["score"], candidate_score=final["score"], target=target,
        observed_mat=dict(tp=mat[0], fp=mat[1], fn=mat[2], f1=f1(*mat)),
        remaining_with_visible_comment=visible, remaining_without_comment=mat[2]-visible,
        upstream_tagged_comment_confirmed_absent=sum(r["mat_positive"] and r["observation"] in {
                "marker_in_adjacent_preceding_comment", "marker_inside_matched_span"} for r in observations),
        required_additional_mat_recoveries_at_same_fp=required,
        score_if_every_visible_comment_miss_recovered=projected(visible),
        required_recoveries_without_comment_if_all_visible_recovered=max(0, required-visible),
        required_vulnerability_f1_if_only_vulnerability_changes=need_vulnerability,
        vulnerability_f1_if_all_vulnerability_fn_removed_at_same_fp=best_vulnerability,
        vulnerability_fp_after_fixing_all_vulnerability_fn=vul[1],
        vulnerability_fp_allowed_if_all_fn_removed=fp_ceiling,
    )
    (HERE / "limits.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    policy = json.loads((HERE / "annotation_policy.json").read_text())
    profile = json.loads((HERE / "profile.json").read_text())
    for file, key, expected in (
        ("marker.py", "marker_sha256", profile["provenance"]),
        ("annotation.py", "code_sha256", policy),
        ("annotation_policy.json", "audit_policy_sha256", profile["provenance"]),
        ("profile_marker.csv", "predictions_sha256", profile["marker"]),
        ("profile_reference.csv", "predictions_sha256", profile["reference"]),
    ):
        if hashlib.sha256((HERE/file).read_bytes()).hexdigest() != expected[key]:
            raise ValueError(f"Reproducibility hash differs for {file}")
    check_annotation(policy)
    scenarios(profile)


if __name__ == "__main__":
    main()
