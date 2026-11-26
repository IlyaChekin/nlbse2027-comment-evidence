"""Freeze the pilot choice and report comparable public-validation results."""
import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
PILOTS = ("pilot_bce_512", "pilot_weighted_512", "pilot_bce_256")


def load(run, name="metrics.json"):
    return json.loads((RUNS / run / name).read_text())


def select():
    runs = {name: load(name) for name in PILOTS}
    ceiling = max(run["scores"]["mean"]["f1"] for run in runs.values())
    eligible = [name for name, run in runs.items() if ceiling - run["scores"]["mean"]["f1"] <= 0.005]
    pick = min(eligible, key=lambda name: (
        runs[name]["cost"]["padded_tokens"], -runs[name]["scores"]["mean"]["f1"], name == "pilot_weighted_512"))
    length = load(pick, "config.json")["length"]
    (REPORTS / "selection.json").write_text(json.dumps(dict(pick=pick, length=length,
        rule="highest mean F1 within 0.005, then fewest padded tokens, then higher mean F1 and unweighted BCE",
        eligible=eligible), indent=2) + "\n")
    print(pick, length)


def summarize():
    rows = []
    for name in ("official_smoke", *PILOTS, "final_fit", "official_public", "final_public"):
        result = load(name)
        rows.append(dict(run=name,
                         **{f"{label}_{met}": value for label, scores in result["scores"].items()
                            for met, value in scores.items()}, **result["cost"]))
    with (REPORTS / "experiments.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    submission = ROOT / "submission"
    submission.mkdir(exist_ok=True)
    shutil.copyfile(RUNS / "final_fit/thresholds.json", submission / "thresholds.json")
    shutil.copyfile(RUNS / "final_public/predictions.csv", submission / "public_validation_predictions.csv")
    print(json.dumps(rows[-2:], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "summarize"))
    args = parser.parse_args()
    (select if args.mode == "select" else summarize)()
