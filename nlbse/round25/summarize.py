"""Verify matched project budgets, frozen predictions, and the delivered archive."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(file):
    return json.loads(file.read_text())


def sha(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def comparisons():
    records=[]
    for directory in ("length","retrieval","views","specialists","skip_sparse"):
        for part in ("test_a","test_b"):
            data=read(HERE/directory/part/"results.json")
            for k,v in data.get("results",data.get("options")).items():
                p=v.get("per_project",v.get("project"))
                matched=v.get("matched",v.get("same_fp"))
                frozen=v.get("frozen",v.get("review"))
                assert all(row["before"]["fp"]==row["after"]["fp"] for row in p.values())
                assert sum(row["delta_tp"] for row in p.values())==matched["delta_tp"]
                records.append((directory,part,k,matched["delta_tp"],frozen["delta_tp"],frozen["delta_fp"]))
    with (HERE/"comparison.csv").open("w",newline="") as stream:
        table=csv.writer(stream)
        table.writerow(("family","rotation","variant","same_fp_tp","frozen_tp","frozen_fp"))
        table.writerows(records)


def check():
    probe=HERE/"views/public"
    results=read(probe/"results.json")
    selected=results["selected"]
    for name,row in results["results"].items():
        assert sha(probe/f"{name}.csv")==row["predictions_sha256"]
        cost=row["cost"]
        assert all(cost[k]==median(trial[k] for trial in row["trials"]) for k in cost)
        score=.6*row["quality"]["mean"]["f1"]+.2*(1-cost["forward_seconds"]/1000)+.2*(1-cost["analytic_gflops"]/3e6)
        assert abs(score-row["score"])<1e-12
        assert row["equal_fp"]["delta_fp"]==0
    receipt=read(HERE/"release_verification/receipt.json")
    zip_path=ROOT/"submission/NLBSE2027_round25_sections_tool.zip"
    assert sha(zip_path)==receipt["archive_sha256"] and receipt["identical_binary_decisions"]==61310
    actual=HERE/"release_verification/release_predictions.csv"
    assert sha(actual)==receipt["predictions_sha256"]
    with actual.open(newline="") as a,(probe/f"{selected}.csv").open(newline="") as b:
        for observed,expected in zip(csv.DictReader(a),csv.DictReader(b),strict=True):
            assert (observed["vulnerability"],observed["MAT"])==(expected["pred_vulnerability"],expected["pred_MAT"])
    print(selected,"verified",results["results"][selected]["score"],flush=True)


if __name__=="__main__":
    comparisons()
    check()
