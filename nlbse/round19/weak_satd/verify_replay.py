"""Recount held-out results, including false positives matched within projects."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    spec = spec_from_file_location("r16_check", HERE.parents[1]/"round16/summarize.py")
    checks = module_from_spec(spec)
    spec.loader.exec_module(checks)
    summary = []
    for rotation in ("test_a","test_b"):
        folder = HERE/rotation
        cfg = checks.load(folder/"policy.json")
        report = checks.load(folder/"results.json")
        expected = checks.rows(folder/"reference_test.csv")
        for mode, metrics in report["modes"].items():
            rows = checks.rows(folder/f"{mode}_test.csv")
            for base, row in zip(expected,rows,strict=True):
                if any(base[key]!=row[key] for key in ("row","project","label","ref_equal","ref_each_project")):
                    raise ValueError("Test labels or reference predictions changed")
            matched = checks.compare(rows,"label","ref_equal","matched",metrics["matched"])
            each = checks.compare(rows,"label","ref_each_project","matched_each_project",metrics["each_project"])
            frozen = checks.compare(rows,"label","ref_frozen","frozen",metrics["frozen"])
            if matched["delta_fp"] or each["delta_fp"] or each["fp"] != report["reference_budget"]:
                raise ValueError("Expected a fixed false-positive budget")
            for project, metrics in metrics["by_project"].items():
                local = [row for row in rows if row["project"]==project]
                sample = checks.compare(local,"label","ref_each_project","matched_each_project",metrics)
                if sample["delta_fp"]:
                    raise ValueError("A project's false-positive budget changed")
            summary.append(dict(rotation=rotation, method=mode, selected=mode==cfg["selected"],rows=len(rows),
                                fixed_fp=matched["fp"], delta_tp=matched["delta_tp"],
                                per_project_tp=each["delta_tp"], matched_F1=matched["f1"],
                                frozen_tp=frozen["delta_tp"], frozen_fp=frozen["delta_fp"]))
    checks.HERE = HERE
    checks.write("project_results.csv",summary)
    print(f"Verified {len(summary)} held-out rule comparisons",flush=True)


if __name__ == "__main__":
    main()
