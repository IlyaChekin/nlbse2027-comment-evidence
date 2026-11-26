"""Check that archived comparisons used the same per-project FP budgets."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def matched(groups):
    if any(x["delta_fp"]!=0 or x["before"]["fp"]!=x["after"]["fp"] for x in groups.values()):
        raise ValueError("A held-out project's false-positive budget changed")
    return sum(x["delta_tp"] for x in groups.values())


def rows():
    for part in ("test_a","test_b"):
        scope=read(HERE/part/"results.json")
        sc_policy=read(HERE/part/"policy.json")
        names=read(HERE/"call_roles"/part/"results.json")
        advice=read(HERE/"adviser"/part/"results.json")
        latent=read(HERE/"latent_expert"/part/"evaluation.json")
        for label,variants,chosen,selection,group_key,score_key,frozen_key in (
            ("anonymous_structure",scope["results"],scope["selected"],sc_policy["candidates"],
                "per_project","project_tp","frozen"),
            ("call_names",names["test"],names["selected"],names["cal"],"per_project","matched","frozen"),
            ("adviser",advice["recipes"],None,advice["recipes"],"per_project","project_tp","fixed"),
            ("latent",latent["results"],None,latent["results"],"project_results","project_tp","frozen"),
        ):
            for name,variant in variants.items():
                tp=matched(variant[group_key])
                if tp!=variant[score_key]:
                    raise ValueError(f"Group aggregate differs for {label}/{part}/{name}")
                dev=selection[name]["cal" if label=="anonymous_structure" else "dev"]
                fixed=variant[frozen_key]
                yield part,label,name,name==chosen,tp,fixed["delta_tp"],fixed["delta_fp"],dev["delta_tp"]


def main():
    with (HERE/"comparison.csv").open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("rotation","family","variant","best_on_development","delta_tp_same_project_fp",
                    "delta_tp_frozen","delta_fp_frozen","delta_tp_development"))
        w.writerows(rows())
    policy=read(HERE/"adviser/official/policy.json")
    if policy["selected"]!="reference" or any(x["projected_delta_score"]>0 for x in policy["options"].values()):
        raise ValueError("A policy passed calibration and must be evaluated")
    print("Both rotations recounted. The adviser did not pass the official internal cost comparison.")


if __name__=="__main__":
    main()
