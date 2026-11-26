"""Verify transfer, fixed-budget comparisons, and public timings from saved predictions."""
from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path
import struct
import sys


HERE = Path(__file__).resolve().parent
FLOAT32 = struct.Struct("<f")


def probability(value):
    # CSV stores the shortest decimal that round-trips through the model's float32.
    return FLOAT32.unpack(FLOAT32.pack(float(value)))[0]


def checker():
    spec = spec_from_file_location("round17_checks", HERE.parent/"round17/summarize.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_results(source, folders, target, check):
    summary, groups = [], []
    for folder in folders:
        receipt = check.read(folder / "results.json")
        policy = check.read(folder / "policy.json")
        config = check.read(folder / "fit.json")
        for part in ("dev", "test"):
            records = receipt[part]
            seen = None
            for mode, row in records.items():
                frame = check.rows(folder / f"{part}_{mode}.csv")
                aligned = [(r["row"],r["project"],r["label"],r["mat"]) for r in frame]
                if seen is None:
                    seen = aligned
                if aligned != seen or len(frame) != row["rows"]:
                    raise ValueError("Inconsistent comparison size or labeled rows")
                equal = check.compare(frame,"label","baseline_equal","equal",row["equal"])
                local = check.compare(frame,"label","baseline_by_project","by_project",row["matched_each_project"])
                frozen = check.compare(frame,"label","baseline_frozen","frozen",row["frozen"])
                if equal["delta_fp"] or local["delta_fp"] or equal["fp"] != row["budget"]:
                    raise ValueError("Fixed-FP comparison changed the baseline budget")
                threshold = policy["modes"][mode]["threshold"]
                if any(int(r["frozen"]) != (probability(r["probability"]) > threshold) for r in frame):
                    raise ValueError("Frozen decision differs from score and threshold")
                summary.append(dict(rotation=folder.parent.name, seed=config["config"]["seed"], part=part,
                                    mode=mode, rows=len(frame), fp=equal["fp"],
                                    delta_tp=equal["delta_tp"], per_project_tp=local["delta_tp"],
                                    frozen_tp=frozen["delta_tp"], frozen_fp=frozen["delta_fp"],
                                    f1=equal["f1"], routed=row["routed"]))
                for project, record in row["by_project"].items():
                    group = [r for r in frame if r["project"]==project]
                    m = check.compare(group,"label","baseline_by_project","by_project",record)
                    if m["delta_fp"]:
                        raise ValueError("Fixed-FP comparison changed a project's budget")
                    groups.append(dict(rotation=folder.parent.name, seed=config["config"]["seed"], part=part,
                                       mode=mode, project=project, rows=len(group), fp=m["fp"],
                                       delta_tp=m["delta_tp"], f1=m["f1"]))
    check.HERE = source
    check.write(target, summary)
    check.write(target.replace(".csv","_by_project.csv"), groups)


def main():
    source = HERE if len(sys.argv)==1 else Path(sys.argv[1]).resolve()
    check = checker()
    folders = [p.parent for key in ("test_a", "test_b") for p in sorted((source/key).glob("*/results.json"))]
    if folders:
        project_results(source, folders, "project_results.csv", check)
    if (source/"official/results.json").exists():
        check.HERE = source
        check.measured(source/"official","public_results.csv")
    if source == HERE:
        frontier(check)


def frontier(check):
    public = HERE.parent/"round16/context_public"
    frame = check.rows(public/"full_public.csv")
    vuln = check.counts(frame, "label_vulnerability", "vulnerability")
    mat = check.counts(frame, "label_mat", "mat")
    reference = check.read(public/"results.json")["full"]
    cost = reference["cost"]
    efficiency = .2*max(1.-cost["forward_seconds"]/1000.,0.)
    efficiency += .2*max(1.-cost["analytic_gflops"]/3_000_000.,0.)
    if abs(reference["score"]-.3*(check.f1(vuln)+check.f1(mat))-efficiency) > 1e-12:
        raise ValueError("The count-based formula failed to reproduce the measured baseline")
    positives = vuln["tp"]+vuln["fn"]
    mat_positives = mat["tp"]+mat["fn"]
    table = []
    for repaired_mat in (0,1,5,10,19,30,38):
        mattp = mat["tp"]+repaired_mat
        mat_f1 = 2*mattp/(mattp+mat_positives+mat["fp"])
        goal = (.97-efficiency)/.3-mat_f1
        for fp_removed in (0,50,100,200,350,500,700):
            fp = vuln["fp"]-fp_removed
            tp = max(vuln["tp"],math.ceil((positives+fp)*goal/(2-goal)))
            perfect = 2*positives/(2*positives+fp)
            table.append(dict(repaired_mat=repaired_mat, removed_vulnerability_fp=fp_removed,
                              needed_extra_vulnerability_tp=tp-vuln["tp"], available_vulnerability_fn=vuln["fn"],
                              reachable_at_budget=tp<=positives,
                              score_with_complete_vulnerability_recall=.3*(mat_f1+perfect)+efficiency))
    check.HERE = HERE
    check.write("target_frontier.csv",table)


if __name__ == "__main__":
    main()
