"""Predict with one of the frozen and verified round-9 configurations."""

import argparse
import csv
from functools import partial
import json
from pathlib import Path

from cache_hybrid import TOKEN_CONFIG
from common import sha256
from fused_attention import HERE, PORTABLE, DeferredOffsets, load_checkpoint, predict
from length_batches import length_buckets
from packed_suffix import PackedPipeline
from windows import chunks

MODES = ("balanced", "fast", "same", "compact", "quality", "eager")


def run(texts, mode):
    policies = json.loads((HERE/"release_policy.json").read_text())
    spec = policies["modes"][mode]
    budget = spec.get("budget",0)
    batcher = partial(length_buckets,budget=budget) if budget else chunks
    if spec["type"]=="adaptive":
        pipeline = PackedPipeline(rules=HERE/spec["rules"] if spec.get("rules") else None,
                                  batcher=batcher)
        return pipeline.predict(texts,spec["mode"])[0]

    config = json.loads(TOKEN_CONFIG.read_text())
    if config["encoder_sha256"] != sha256(PORTABLE/"model.safetensors"):
        raise ValueError("The readout belongs to a different checkpoint")
    model,tok = load_checkpoint(PORTABLE,"cuda")
    p,_ = predict(model,DeferredOffsets(tok),texts,config["configurations"][spec["mode"]],
                  spec["mode"],spec.get("fused",True),{},batcher=batcher)
    return p


def main(source, output, mode):
    with source.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV requires a Function column")
        texts=[r["Function"] for r in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Each input record needs function text")
    scores=run(texts,mode)
    with output.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("vulnerability", "MAT"))
        writer.writerows((scores>.5).astype(int).tolist())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=MODES,default="balanced")
    args=parser.parse_args()
    main(args.input,args.output,args.mode)
