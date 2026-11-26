"""Run the single compact checkpoint with the frozen full-context policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round12"))
from efficient_aug import run
import small


def full_context(texts,tokenizer):
    result=[]
    for i in range(0,len(texts),1024):
        result.extend(tokenizer(texts[i:i+1024],truncation=False,verbose=False)["input_ids"])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    entry=json.loads((HERE/"entry.json").read_text())
    weight=HERE/entry["checkpoint"]
    with weight.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=entry["checkpoint_sha256"]:
            raise ValueError("The frozen encoder weights changed")
    with args.input.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Expected a Function input column")
        texts=[row["Function"] for row in reader]
    if not texts or any(s is None for s in texts):
        raise ValueError("Every row must supply a function")
    token=small.tokenizer()
    model=small.Encoder(len(token),entry["width"]).cuda()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    for layer in model.local:
        layer.use_shifted=True
    config=entry["route"]
    pipe=SimpleNamespace(encoder=model,tok=token,student={"thresholds":(.5,.5)},encode=full_context)
    p,_=run(pipe,texts,config)
    with args.output.open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("vulnerability","MAT"))
        w.writerows((p>(entry["vulnerability_threshold"],.5)).astype(int).tolist())


if __name__=="__main__":
    main()
