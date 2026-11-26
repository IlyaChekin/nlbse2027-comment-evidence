"""Predict vulnerability and MAT from the Function column using the frozen release."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import torch

from runtime import HERE, refine

ROOT = HERE.parents[1]
SOURCE = ROOT / "round16"
sys.path.insert(0, str(SOURCE))
from predict import full_context
from efficient_aug import run
import small


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((HERE/"entry.json").read_text())
    os.environ.setdefault("NLBSE_SCANNER_WORKERS", str(config["worker_limit"]))
    for file,key in (("scan.mjs","scanner_sha256"),("rules_manifest.json","rule_source_sha256"),
                     ("models.joblib","model_sha256")):
        if digest(HERE/file) != config[key]:
            raise ValueError(f"Frozen artifact changed: {file}")
    prior = json.loads((SOURCE/"entry.json").read_text())
    weight = SOURCE/prior["checkpoint"]
    if digest(weight) != prior["checkpoint_sha256"] or config["checkpoint_sha256"] != prior["checkpoint_sha256"]:
        raise ValueError("Frozen encoder checkpoint changed")
    with args.input.open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function input column")
        texts = [row["Function"] for row in reader]
    if not texts or any(value is None for value in texts):
        raise ValueError("Every row must supply a function")
    tokenizer = small.tokenizer()
    model = small.Encoder(len(tokenizer),prior["width"]).cuda()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    for layer in model.local:
        layer.use_shifted = True
    from types import SimpleNamespace
    pipe = SimpleNamespace(encoder=model,tok=tokenizer,student={"thresholds":(.5,.5)},encode=full_context)
    p,_ = run(pipe,texts,config["route"])
    p[:,0] = refine(texts,p[:,0],config,joblib.load(HERE/"models.joblib"))
    with args.output.open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>(config["threshold"],.5)).astype(int).tolist())


if __name__ == "__main__":
    main()
