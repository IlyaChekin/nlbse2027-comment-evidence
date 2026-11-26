"""Cache frozen prefix features and late-update targets on official training groups."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fused_attention import (HERE, EARLY, PORTABLE, Policy, DeferredOffsets, JointHead,
                             as_common_scale, dataset, finish_fused, lexical_features,
                             load_checkpoint, ordered_batches, pool, prefix_fused,
                             use_inputs)
from pipeline import TRAIN, load, select_indices, write_json
from common import sha256
from windows import BLOCK, chunks, pack

TOKEN_CONFIG = HERE.parent/"round8/tokens_frozen/policy.json"
ACTIONS = ("query_saliency_50", "query_even_65")


@torch.inference_mode()
def encode(model, tokenizer, texts, config, actions):
    native_heads = {name: JointHead.from_dict(config[name]["head"]) for name in actions}
    policies = {name: Policy(**config[name]["spec"]) for name in actions}
    stack = model.encoder.encoder
    n = len(texts)
    features = np.empty((n, model.encoder.config.d_model+7), dtype=np.float32)
    widths = np.zeros(n, dtype=np.int16)
    targets = {name: np.empty((n,2), dtype=np.float32) for name in actions}
    cache = {}
    start = time.perf_counter()

    def prepare_at(at):
        local = texts[at:at+BLOCK]
        return use_inputs(local, tokenizer, wide=True), lexical_features(local)

    with torch.autocast("cuda",dtype=torch.float16):
        for at,(data,side) in ordered_batches(range(0,n,BLOCK),prepare_at,True):
            sequences=data.views["head"]
            widths[at:at+len(sequences)] = [len(seq) for seq in sequences]
            for group in chunks(sequences,range(len(sequences))):
                x,mask=pack(sequences,group,data.pad,"cuda")
                state,bias=prefix_fused(model,x,mask,EARLY,cache)
                pooled=pool(stack,state,mask)
                shallow=model.head(pooled).float().cpu().numpy()
                idx=at+np.asarray(group)
                features[idx]=np.column_stack((pooled.float().cpu().numpy(),shallow,side[group]))
                for name in actions:
                    logits=finish_fused(model,state,bias,mask,policies[name],EARLY).float().cpu().numpy()
                    p=native_heads[name](np.column_stack((logits,side[group])))
                    targets[name][idx]=as_common_scale(p,config[name]["thresholds"])
            if at//BLOCK%8==0:
                print("cached",at+len(sequences),"of",n,"in",round(time.perf_counter()-start,2),flush=True)
    return features,widths,targets,time.perf_counter()-start


def persist(output,name,model,tokenizer,texts,y,ids,config,actions):
    features,widths,targets,elapsed=encode(model,tokenizer,texts,config,actions)
    part=output/f"{name}.part.npz"
    np.savez(part,features=features.astype(np.float16),widths=widths,y=y,ids=ids,**targets)
    part.replace(output/f"{name}.npz")
    return elapsed


def main(output):
    spec=json.loads(TOKEN_CONFIG.read_text())
    config=spec["configurations"]
    model,tokenizer=load_checkpoint(PORTABLE,"cuda")
    tokenizer=DeferredOffsets(tokenizer)
    official,y,_=load(TRAIN)
    fit,tune=select_indices(pilot=False)
    if set(fit)&set(tune):
        raise ValueError("Train fit and holdout overlap")
    output.mkdir(parents=True,exist_ok=True)
    fit_seconds=persist(output,"fit",model,tokenizer,[official[i] for i in fit],y[fit],fit,config,(ACTIONS[1],))
    _,_,texts,labels,ids,_=dataset(full=True)
    if set(fit)&set(ids):
        raise ValueError("The student cache overlaps training holdout")
    tune_seconds=persist(output,"holdout",model,tokenizer,texts,labels,ids,config,ACTIONS)
    write_json(output/"manifest.json",dict(
        encoder_sha256=sha256(PORTABLE/"model.safetensors"), token_config_sha256=sha256(TOKEN_CONFIG),
        depth=EARLY, input_cap=768, fit_rows=len(fit), holdout_rows=len(ids),
        fit_seconds=fit_seconds, holdout_seconds=tune_seconds, hardware=torch.cuda.get_device_name()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"cache")
    main(parser.parse_args().output)
