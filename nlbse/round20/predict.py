"""Run the frozen compact encoder and the selected static-rule subset."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import torch


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
COMPACT=ROOT/"round16"
STATIC=ROOT/"round19/weak_satd"
sys.path[:0]=[str(COMPACT),str(STATIC)]
from predict import full_context
from efficient_aug import run
from runtime import combine,scanner
import small


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def require_hash(path,expected):
    if checksum(path)!=expected:
        raise ValueError(f"Frozen dependency changed: {path.name}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((HERE/"entry.json").read_text())
    prior=json.loads((COMPACT/"entry.json").read_text())
    os.environ.setdefault("NLBSE_SCANNER_WORKERS",str(config["worker_limit"]))
    weight=COMPACT/prior["checkpoint"]
    program=HERE/config["program"] if config["program"] else None
    require_hash(weight,config["checkpoint_sha256"])
    require_hash(STATIC/"rules_manifest.json",config["source_sha256"])
    require_hash(HERE/"head.joblib",config["head_sha256"])
    if program:
        require_hash(program,config["program_sha256"])
        require_hash(HERE/"scan.mjs",config["scanner_sha256"])
    else:
        require_hash(STATIC/"scan.mjs",config["scanner_sha256"])
    csv.field_size_limit(16*1024*1024)
    with args.input.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function input column")
        text=[row["Function"] for row in reader]
    if not text or any(value is None for value in text):
        raise ValueError("Every row must supply a function")
    tokenizer=small.tokenizer()
    model=small.Encoder(len(tokenizer),prior["width"]).cuda()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    for layer in model.local:
        layer.use_shifted=True
    pipe=SimpleNamespace(encoder=model,tok=tokenizer,student={"thresholds":(.5,.5)},encode=full_context)
    p,_=run(pipe,text,config["route"])
    p0=p[:,0]
    gate=config["gate"]
    enabled=(p0>gate[0]) & (p0<gate[1])
    if enabled.any():
        found,raw=scanner([t for t,a in zip(text,enabled,strict=True) if a],tuple(config["columns"]),program)
        if found!=["rule_"+tag for tag in config["columns"]]:
            raise ValueError("The selected rules returned different columns")
        head=joblib.load(HERE/"head.joblib")
        p[enabled,0]=combine(p0[enabled],raw,"linear_all",{"linear":head})
    with args.output.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows((p>(config["threshold"],.5)).astype(int).tolist())


if __name__=="__main__":
    main()
