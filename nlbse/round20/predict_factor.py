"""Predict two labels with a compact conditional model and an exact rule gate."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
COMPACT=HERE.parent/"round16"
STATIC=HERE.parent/"round19/weak_satd"
sys.path[:0]=[str(COMPACT),str(STATIC)]
from predict import full_context
from runtime import scanner
import small
from conditional import Policy,choose,estimate,valid,weak
from rule_model import RuleEncoder
from normalize_code import batch as canonicalize
from students import batch,batch_plan,lexical_features,ordered_batches


def require_hash(path,expected):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=expected:
            raise ValueError(f"Frozen dependency changed: {path.name}")


@torch.inference_mode()
def infer(model,tokenizer,code,auxiliary=False):
    probabilities=np.empty((len(code),2),dtype=np.float32)
    auxiliary_scores=np.empty((len(code),22),dtype=np.float32) if auxiliary else None
    def prepare(first):
        segment=code[first:first+2048]
        return full_context(segment,tokenizer),lexical_features(segment)
    with torch.autocast("cuda",dtype=torch.float16):
        for first,(seq,lex) in ordered_batches(range(0,len(code),2048),prepare,True):
            probabilities[first:first+len(seq),1]=lex[:,:4].any(axis=1)
            side=torch.as_tensor(lex,device="cuda")
            for rows in batch_plan(list(range(len(seq))),seq,small.BATCH):
                x,mask=batch(rows,seq,tokenizer.pad_token_id)
                logits=model(x,mask,side[rows])
                indices=first+np.asarray(rows)
                if auxiliary:
                    p=logits.float().sigmoid().cpu().numpy()
                    probabilities[indices,0]=p[:,0]
                    auxiliary_scores[indices]=p[:,2:]
                else:
                    probabilities[indices,0]=logits[:,0].float().sigmoid().cpu().numpy()
    return probabilities,auxiliary_scores


def decisions(text,config,model,tok,heads):
    p,aux=infer(model,tok,text,auxiliary=True)
    route=config["route"]
    second=(p[:,0]>route["lower"])&(p[:,0]<route["upper"])
    if second.any():
        canonical=canonicalize([s for s,on in zip(text,second,strict=True) if on])
        q,_=infer(model.encoder,tok,canonical)
        p[second,0]=.5*(p[second,0]+q[:,0])
    values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
    policy=Policy(**config["policy"])
    observed=estimate(heads,values)
    score,no,yes,active=choose(values,observed,policy,config["gate"])
    if active.any():
        cols,raw=scanner([s for s,on in zip(text,active,strict=True) if on],program=HERE/config["program"])
        if cols!=config["rule_columns"]:
            raise ValueError("Frozen scanner returned a different rule set")
        score[active]=np.where(valid(raw),np.where(weak(raw),yes[active],no[active]),score[active])
    p[:,0]=score
    return (p>(config["threshold"],.5)).astype(int)


def main(decide=decisions):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    config=json.loads((HERE/"factor_entry.json").read_text())
    os.environ.setdefault("NLBSE_SCANNER_WORKERS",str(config["worker_limit"]))
    os.environ["NLBSE_WEAK_SOURCE"]=str(STATIC)
    for file,key in (("factor.pt","checkpoint_sha256"),("conditional_heads.joblib","head_sha256"),
                     (config["program"],"program_sha256"),("scan.mjs","scanner_sha256")):
        require_hash(HERE/file,config[key])
    require_hash(STATIC/"rules_manifest.json",config["source_sha256"])
    prior=json.loads((COMPACT/"entry.json").read_text())
    tok=small.tokenizer()
    model=RuleEncoder(small.Encoder(len(tok),prior["width"]),conditional=True).cuda()
    model.load_state_dict(torch.load(HERE/"factor.pt",map_location="cpu",weights_only=True))
    for block in model.encoder.local:
        block.use_shifted=True
    model.eval()
    heads=joblib.load(HERE/"conditional_heads.joblib")
    csv.field_size_limit(16*1024*1024)
    with a.input.open(encoding="utf-8-sig",newline="") as stream:
        table=csv.DictReader(stream)
        if "Function" not in (table.fieldnames or ()):
            raise ValueError("Expected a Function input column")
        text=[row["Function"] for row in table]
    if not text or any(s is None for s in text):
        raise ValueError("Every row must supply a function")
    with a.output.open("w",newline="") as stream:
        out=csv.writer(stream)
        out.writerow(("vulnerability","MAT"))
        out.writerows(decide(text,config,model,tok,heads).tolist())


if __name__=="__main__":
    main()
