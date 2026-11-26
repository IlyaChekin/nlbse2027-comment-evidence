"""Aggregate and verify the archived MAT predictions without the raw source data."""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def source(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def binary(rows, key):
    return [r[key] == "1" for r in rows]


def tally(gold, bits):
    return {"tp": sum(g and p for g,p in zip(gold,bits,strict=True)),
            "fp": sum(not g and p for g,p in zip(gold,bits,strict=True)),
            "fn": sum(g and not p for g,p in zip(gold,bits,strict=True))}


def confirm(expected, base, actual, gold):
    old, new = tally(gold,base), tally(gold,actual)
    if any(expected[k][field] != result[field] for k,result in (("base",old),("candidate",new)) for field in result):
        raise ValueError("Archived predictions do not match the run report")
    if sum(g and a and not b for g,a,b in zip(gold,actual,base,strict=True)) != expected["recovered"]:
        raise ValueError("Incorrect number of repaired false negatives")


def verify_project(folder, name, study):
    report = json.loads((folder/name).read_text())
    for group, entries in report.items():
        for variant, result in entries.items():
            rows = source(folder/f"{group}_{variant.replace('/','_')}.csv")
            base, gold = binary(rows,"base"), binary(rows,"label")
            for key,column in (("exact","matched"),("frozen","fixed")):
                confirm(result[key],base,binary(rows,column),gold)
            matched, frozen = result["exact"], result["frozen"]
            if matched["extra_fp"] != 0:
                raise ValueError("The matched comparison has a different FP budget")
            yield dict(study=study,group=group,method=variant,rows=len(rows),baseline_fn=matched["base"]["fn"],
                       matched_fp=matched["base"]["fp"],matched_recovered=matched["recovered"],
                       matched_f1=matched["candidate"]["f1"],frozen_recovered=frozen["recovered"],
                       frozen_delta_fp=frozen["extra_fp"],frozen_f1=frozen["candidate"]["f1"])


def verify_public(path):
    report = json.loads(path.read_text())
    baseline,candidate = [source(path.with_name(f"{name}.csv")) for name in ("reference","candidate")]
    if any((a["row"],a["label_MAT"],a["pred_vulnerability"]) != (b["row"],b["label_MAT"],b["pred_vulnerability"])
           for a,b in zip(baseline,candidate,strict=True)):
        raise ValueError("Public candidate changed the paired rows or vulnerability predictions")
    base,gold = binary(baseline,"pred_MAT"),binary(baseline,"label_MAT")
    confirm(report["candidate"]["mat"],base,binary(candidate,"pred_MAT"),gold)
    confirm(report["matched_fp"],base,binary(candidate,"MAT_matched_budget"),gold)
    if report["matched_fp"]["extra_fp"] != 0:
        raise ValueError("Wrong public FP budget")
    return dict(training=path.parent.parent.name,method=path.parent.name,reference_score=report["reference"]["score"],
                candidate_score=report["candidate"]["score"],baseline_mat_f1=report["reference"]["metrics"]["MAT"]["f1"],
                frozen_mat_f1=report["candidate"]["metrics"]["MAT"]["f1"],
                matched_recovered=report["matched_fp"]["recovered"],frozen_recovered=report["candidate"]["mat"]["recovered"],
                frozen_delta_fp=report["candidate"]["mat"]["extra_fp"],
                mat_seconds=report["candidate"]["cost"]["mat_seconds"])


def table(path, rows):
    with (HERE/path).open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def main():
    records = [*verify_project(HERE,"project_results.json","lexical"),
               *verify_project(HERE/"prototype","results.json","prototype"),
               *verify_project(HERE/"readout","results.json","readout")]
    public = [verify_public(path) for path in sorted(HERE.glob("public/*/*/analysis.json"))]
    table("results.csv",records)
    table("public_results.csv",public)
    print(f"Verified {len(records)} project comparisons and {len(public)} public comparisons",flush=True)


if __name__=="__main__":
    main()
