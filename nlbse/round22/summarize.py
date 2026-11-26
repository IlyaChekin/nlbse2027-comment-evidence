"""Recount project comparisons and compute the remaining gap to the score target."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_text())


def check():
    methods = (
        ("evaluation.json","results"),
        ("frozen_head/evaluation.json","results"),
        ("rule_evaluation.json","comparisons"),
        ("layout_evaluation.json",None),
    )
    out = []
    for rotation in ("test_a","test_b"):
        folder = HERE/rotation
        for name, key in methods:
            data = read(folder/name)
            models = data if key is None else data[key]
            for variant, cfg in models.items():
                group = cfg["project_results"]
                if any(v["delta_fp"] for v in group.values()) or sum(v["delta_tp"] for v in group.values())!=cfg["project_tp"]:
                    raise ValueError("A comparison violated its per-project budget")
                out.append((rotation,variant,cfg["project_tp"],cfg["frozen"]["delta_tp"],
                            cfg["frozen"]["delta_fp"],cfg["dev"]["delta_tp"]))
    with (HERE/"comparison.csv").open("w",newline="") as stream:
        w = csv.writer(stream)
        w.writerow(("rotation","variant","same_fp_delta_tp","fixed_delta_tp","fixed_delta_fp","dev_delta_tp"))
        w.writerows(out)

    report = read(HERE/"union_public/results.json")
    verified = read(HERE/"union_verification.json")
    scanner = HERE/"scan_union.mjs"
    if (verified["rows"],verified["differences"])!=(214586,0) or report["program_sha256"]!=hashlib.sha256(scanner.read_bytes()).hexdigest():
        raise ValueError("Static outcome verification receipt changed")
    f1 = report["quality"]["mean"]["f1"]
    for stat in report["stats"].values():
        cost = stat["cost"]
        for name in cost:
            if cost[name]!=median(t[name] for t in stat["trials"]):
                raise ValueError("Stored timing differs from raw trials")
        score = .6*f1+.2*(1-cost["forward_seconds"]/1000)+.2*(1-cost["analytic_gflops"]/3e6)
        if abs(score-stat["score"])>1e-12:
            raise ValueError("Stored competition score is inconsistent")
    final = report["stats"]["short_circuit"]
    target = .97
    gap = target-final["score"]
    quality = report["quality"]
    next_f1 = quality["vulnerability"]["f1"]+gap/.3
    required_mat = quality["MAT"]["f1"]+gap/.3
    n_mat = 505
    tp = int(round(quality["MAT"]["recall"]*n_mat))
    same_fp = 1
    def change(k):
        return 2*(tp+k)/(tp+k+n_mat+same_fp)-quality["MAT"]["f1"]
    required_recoveries = next(k for k in range(n_mat-tp+1) if change(k)*.3>=gap)
    result = dict(
        measured_local_score=final["score"],gap_to_target=gap,score_with_zero_runtime_and_gflops=.4+.6*f1,
        required_vulnerability_f1_if_only_vulnerability_changes=next_f1,
        required_mat_f1_if_only_mat_changes=required_mat,
        required_recoveries_at_same_fp=required_recoveries,
        score_if_18_visible_mat_misses_recovered=final["score"]+.3*change(18),
    )
    (HERE/"score_limits.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)


if __name__=="__main__":
    check()
