"""Build the deployment registry and a table from the saved benchmark receipts."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads((HERE/path).read_text())


def main():
    budget=read("length_batches/analysis.json")["selected_budget"]
    plan={
        "balanced":{"type":"adaptive","mode":"asym_score","rules":"frozen/asymmetric.json","budget":budget,
                    "experiment":"combined","result":"balanced_length"},
        "fast":{"type":"adaptive","mode":"score","budget":budget,"experiment":"combined","result":"fast_length"},
        "same":{"type":"fixed","mode":"query_saliency_50","experiment":"fused","result":"fused"},
        "compact":{"type":"fixed","mode":"query_saliency_50","budget":budget,
                   "experiment":"length_batches","result":"selected"},
        "quality":{"type":"fixed","mode":"query_even_65","experiment":"fused_quality","result":"fused"},
        "eager":{"type":"fixed","mode":"query_saliency_50","fused":False,
                 "experiment":"fused","result":"eager"},
    }
    baseline=read("../round8/tokens_final/analysis.json")["results"]["query_saliency_50"]
    rows={"prior_release":baseline}
    for mode,item in plan.items():
        exp,result=item["experiment"],item["result"]
        item["evidence"]=f"{exp}/{result}/predictions.csv"
        record=read(f"{exp}/analysis.json")["results"][result]
        rows[mode]=record
    with (HERE/"results.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("mode","mean_f1","vulnerability_f1","mat_f1","end_to_end_seconds",
                         "forward_seconds","estimated_gflops","local_formula"))
        for key,value in rows.items():
            scores,cost=value["scores"],value["cost"]
            writer.writerow((key,*(scores[k]["f1"] for k in ("mean","vulnerability","MAT")),
                             *(cost[k] for k in ("wall_seconds","forward_seconds","analytic_gflops")),
                             value["local_formula"]))
    (HERE/"release_policy.json").write_text(json.dumps(
        {"default":"balanced","checkpoint_sha256":read("cache/manifest.json")["encoder_sha256"],
         "modes":plan},indent=2)+"\n")


if __name__=="__main__":
    main()
