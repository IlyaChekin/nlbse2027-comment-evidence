"""Verify the structural distillation predictions with the shared round 17 checker."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    spec = spec_from_file_location("round17_report", HERE.parent / "round17/summarize.py")
    checks = module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.HERE = HERE
    checks.project_suite([HERE / "projects" / str(seed) for seed in (42, 101)], "project_results.csv")
    if (HERE / "official/results.json").is_file():
        checks.measured(HERE / "official", "public_results.csv")
    folder = HERE / "direct_teacher"
    if (folder / "results.json").is_file():
        summary = checks.read(folder / "results.json")
        table = []
        for role in ("dev", "test"):
            for mode, record in summary[role].items():
                frame = checks.rows(folder / f"{role}_{mode}.csv")
                value = checks.compare(frame, "label", "before_equal", "after_equal", record["equal"])
                if value["delta_fp"]:
                    raise ValueError("Direct teacher comparison changed the FP budget")
                table.append(dict(role=role, mode=mode, selected=mode == summary["selected"], rows=len(frame),
                                  fp=value["fp"], delta_tp=value["delta_tp"], f1=value["f1"]))
        checks.write("direct_results.csv", table)


if __name__ == "__main__":
    main()
