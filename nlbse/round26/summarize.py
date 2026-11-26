"""Recount the paired project and internal comparisons and verify their score sources."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROJECTS=("test_a","test_b")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def group_scores():
    result={}
    rows=[]
    for family in ("disagreement","ranking","shape_margin","specialist"):
        results={part:read(HERE/family/part/"results.json") for part in (*PROJECTS,"official")}
        official=results["official"]["options"]
        meta=[]
        for key,dev in official.items():
            counts=[]
            cal=dev.get("cal",dev.get("calibration"))
            for part in PROJECTS:
                v=results[part]["options"][key]
                detail=v.get("per_project",v.get("project"))
                if any(x["before"]["fp"]!=x["after"]["fp"] or x["delta_fp"]!=0 for x in detail.values()):
                    raise ValueError(f"Matched comparison changed a budget for {family}/{part}/{key}")
                matched=v["matched"]
                if matched["delta_tp"]!=sum(x["delta_tp"] for x in detail.values()):
                    raise ValueError("Project aggregate does not reproduce")
                frozen=v["frozen"]
                counts.append(matched["delta_tp"])
                rows.append((family,key,part,matched["delta_tp"],frozen["delta_tp"],frozen["delta_fp"],
                             cal["delta_tp"],dev["proxy"]))
            if min(counts)>=0 and max(counts)>0:
                meta.append((dev["proxy"],key,*counts,cal["delta_tp"]))
        result[family]=dict(support=len(meta),best_supported_internal=max(meta,default=None))
    p=HERE/"candidate_summary.json"
    p.write_text(json.dumps(result,indent=2)+"\n")
    with (HERE/"comparisons.csv").open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("method","variant","held_out","extra_tp_same_fp","frozen_extra_tp","frozen_extra_fp",
                    "official_calibration_tp","official_proxy_score"))
        w.writerows(rows)
    return result


def main():
    value=group_scores()
    folder=HERE/"compact/official"
    current=read(folder/"manifest.json")
    assert sum(rec["held"] for rec in current["folds"].values())==current["rows"]
    assert sha(folder/"scores.npz")==current["sha256"]
    assert current["sparse_oof_sha256"]==sha(ROOT/"round24/official/token_oof.npz")
    for kind in ("disagreement","ranking"):
        for part in (*PROJECTS,"official"):
            record=read(HERE/kind/part/"results.json")
            assert sha(HERE/kind/part/"training.npz")==record["oof_sha256"]
            assert sha(HERE/kind/part/"models.joblib")==record["weights_sha256"]
    for part in (*PROJECTS,"official"):
        census=HERE/"error_census"/part
        assert sha(census/"evidence.npz")==read(census/"census.json")["evidence_sha256"]
        specialist=HERE/"specialist"/part
        assert sha(specialist/"readouts.joblib")==read(specialist/"results.json")["weights_sha256"]
    prev=ROOT/"round25/release_verification"
    artifact=ROOT/"submission/NLBSE2027_round25_sections_tool.zip"
    signed=read(prev/"receipt.json")
    assert sha(artifact)==signed["archive_sha256"] and signed["identical_binary_decisions"]==61310
    print(value)
    print("Round 25 release hash verified. Compact out-of-file scores:",current["rows"])


if __name__=="__main__":
    main()
