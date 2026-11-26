"""Confirm the frozen quality-first variant on the full public validation set."""
import csv
import json

import numpy as np
import torch
from transformers import RobertaTokenizerFast

from ideas import DEVICE, encode, load_model, measure, wide
from pipeline import PUBLIC, load, metric, write_json
from protocol import OUT, ROOT, SIX, TWELVE, save
from round2_report import bootstrap_delta


def main():
    config = json.loads((OUT / "extra_context/metrics.json").read_text())["config"]
    texts, y, scope = load(PUBLIC, audit=True)
    data = encode(texts, RobertaTokenizerFast.from_pretrained(SIX, local_files_only=True))
    model = load_model(TWELVE)
    mix = torch.tensor(config["tail_fraction"], device=DEVICE)
    th = torch.tensor(config["thresholds"], device=DEVICE)

    def infer(rows):
        first, last = wide(model, data, rows)
        p = first + (last-first)*mix
        return p, p > th, 0

    ids = list(range(len(y)))
    p, bits, cost = measure("public_extra_context", data, ids, infer, sum(x.numel() for x in model.parameters()),
                            profile_operations=False)
    save("public_extra_context", ids, y, p, bits, cost, {**config, "scope": "full public 30655", "timed_profile": False})
    baseline_run = ROOT / "runs/own12_packed_public"
    before = json.loads((baseline_run / "metrics.json").read_text())
    with (baseline_run / "predictions.csv").open(newline="") as infile:
        baseline_rows = list(csv.DictReader(infile))
    assert [int(x["row"]) for x in baseline_rows] == ids
    base_p = np.array([[float(r["p_vulnerability"]), float(r["p_MAT"])] for r in baseline_rows])
    base = base_p > before["thresholds"]
    label_scope = data.lengths > 512
    group_map = {}
    groups = [group_map.setdefault((project, file), len(group_map))
              for project, file in zip(scope["Projectname"], scope["Filepath"], strict=True)]
    three = {
        "first_at_baseline_thresholds": base,
        "first_at_frozen_context_thresholds": base_p > config["thresholds"],
        "first_plus_last_at_baseline_thresholds": p > before["thresholds"],
        "first_plus_last_at_frozen_context_thresholds": bits,
    }
    receipt = dict(rows=len(ids), long_rows=int(label_scope.sum()),
                   paired_delta_95=bootstrap_delta(y, base, bits, groups),
                   comparisons={name: dict(all=metric(y, pred), over_512=metric(y[label_scope], pred[label_scope]))
                                for name, pred in three.items()})
    write_json(OUT / "public_context_confirmation.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
