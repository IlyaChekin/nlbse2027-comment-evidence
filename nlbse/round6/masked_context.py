"""Isolate the content of the exact same selected comment window."""
import argparse
import time

import numpy as np
import torch

from common import (HERE, PORTABLE, PUBLIC, THRESHOLDS, TRAIN, dimensions, forward_rows,
                    load, load_checkpoint, prepare, save, write_json)
from comment_spans import comment_bits
from compare import dense_gflops
from continuation_select import best_thresholds
from project_model import OUT as PROJECT, family
from windows import BLOCK, BODY, chunks, comment_start, pack

OUT = HERE / "masking"
ARMS = ("full", "comment_only", "code_only")


def content_masks(texts, tokenizer, inputs, rows):
    masks = {}
    for ix in range(0,len(rows),128):
        indices = rows[ix:ix+128]
        batch = [texts[i] for i in indices]
        encoded = tokenizer(batch,add_special_tokens=False,return_offsets_mapping=True,verbose=False)
        for row,text,ids,offsets in zip(indices,batch,encoded["input_ids"],encoded["offset_mapping"],strict=True):
            comments = comment_bits(text,offsets)
            omitted = [i for i in range(BODY,len(comments)) if comments[i]]
            if not omitted:
                raise ValueError("Masking requested for an example with no omitted comment")
            start = comment_start(len(ids),omitted,126)
            view = inputs.views["cover128"][row]
            expected = [tokenizer.bos_token_id,*ids[start:start+126],tokenizer.eos_token_id]
            if view != expected:
                raise ValueError("Masking changed the selected content window")
            positions = np.asarray(comments[start:start+len(view)-2],dtype=bool)
            if not positions.any():
                raise ValueError("Selected frame did not include any comments")
            masks[row] = positions
    return masks


@torch.inference_mode()
def forward_masked(model,inputs,eligible,masks,arm):
    view = inputs.views["cover128"]
    p = np.full((len(view),2),np.nan,dtype=np.float32)
    ops, elapsed = 0.,0.
    dims = dimensions(model)
    with torch.autocast("cuda",dtype=torch.float16):
        for start in range(0,len(view),BLOCK):
            rows = np.flatnonzero(eligible[start:start+BLOCK])+start
            for group in chunks(view,rows):
                tokens, original = pack(view,group,inputs.pad,"cpu")
                mask = original.clone()
                for i, row in enumerate(group):
                    match = masks[row]
                    if arm == "comment_only":
                        mask[i,1:1+len(match)] = torch.as_tensor(match,dtype=mask.dtype)
                    elif arm == "code_only":
                        mask[i,1:1+len(match)] = torch.as_tensor(~match,dtype=mask.dtype)
                    if mask[i,0] != 1 or mask[i,len(match)+1] != 1:
                        raise ValueError("Special tokens changed under the intervention")
                t = time.perf_counter()
                output = model(tokens.cuda(),mask.cuda()).float().sigmoid()
                p[group] = output.cpu().numpy()
                elapsed += time.perf_counter()-t
                ops += dense_gflops([tuple(tokens.shape)],dims)
    if not np.isfinite(p[eligible]).all():
        raise ValueError("Masking produced non-finite scores")
    return p,dict(forward_seconds=elapsed,analytic_gflops=ops)


def calibration(model,tokenizer,texts,y,ids):
    data = prepare([texts[i] for i in ids],tokenizer,("head",),metadata=False)
    p = forward_rows(model,data.views["head"],np.arange(len(ids)),data.pad,"cuda")
    threshold = best_thresholds(y[ids],p)
    write_json(PROJECT/"thresholds.json",dict(thresholds=threshold,rows=len(ids),
                                             source="head only, calibration projects"))
    return threshold


def run(split):
    output = OUT/split
    output.mkdir(parents=True,exist_ok=True)
    if split == "public":
        texts,y,fields = load(PUBLIC,audit=True)
        ids = np.arange(len(y))
        model,tokenizer = load_checkpoint(PORTABLE,"cuda")
        thresholds = THRESHOLDS
    else:
        all_texts, all_labels, columns = load(TRAIN,audit=True)
        model,tokenizer = load_checkpoint(PROJECT/"checkpoint","cuda")
        with np.load(PROJECT/"split.npz") as stored:
            ids = stored["heldout"]
            thresholds = calibration(model,tokenizer,all_texts,all_labels,stored["calibration"])
        texts = [all_texts[i] for i in ids]
        y = all_labels[ids]
        fields = {name:[columns[name][i] for i in ids] for name in ("Projectname","Filepath")}
    start = time.perf_counter()
    inputs = prepare(texts,tokenizer,("head","cover128"))
    preparation_seconds = time.perf_counter()-start
    head_cost = dict(forward_seconds=0.,analytic_gflops=0.)
    p = forward_rows(model,inputs.views["head"],np.arange(len(y)),inputs.pad,"cuda",head_cost)
    take = inputs.missing & (p[:,1]<=thresholds[1])
    rows = np.flatnonzero(take)
    t = time.perf_counter()
    masks = content_masks(texts,tokenizer,inputs,rows)
    mask_preparation_seconds = time.perf_counter()-t
    np.savez_compressed(output/"rows.npz",ids=ids,eligible=take,lengths=inputs.lengths,omitted=inputs.omitted,
                        crossing=inputs.crossing,group=np.asarray([family(x) for x in fields["Projectname"]]),
                        comment_in_window=[int(masks[i].sum()) if i in masks else 0 for i in range(len(y))],
                        content_in_window=[len(row)-2 if row else 0 for row in inputs.views["cover128"]])
    saved_meta = dict(apparatus=torch.cuda.get_device_name(),rows=len(y),preparation_seconds=preparation_seconds,
                      mask_preparation_seconds=mask_preparation_seconds)
    save(output/"head",y,ids,np.zeros(len(y),dtype=bool),p,
         dict(head_cost,extra_windows=0,**saved_meta),thresholds)
    costs = []
    for arm in ARMS:
        second,cost = forward_masked(model,inputs,take,masks,arm)
        full = p.copy()
        full[rows,1] = np.maximum(p[rows,1],second[rows,1])
        costs.append(cost["analytic_gflops"])
        save(output/arm,y,ids,np.zeros(len(y),dtype=bool),full,
             dict(forward_seconds=head_cost["forward_seconds"]+cost["forward_seconds"],
                  analytic_gflops=head_cost["analytic_gflops"]+cost["analytic_gflops"],
                  extra_windows=len(rows),**saved_meta),thresholds)
    if any(cost!=costs[0] for cost in costs):
        raise ValueError("Ablations differ in their dense operation shapes")
    write_json(output/"design.json",dict(thresholds=thresholds,window_calls=len(rows),
                                         identical_shapes=True,head_gflops=head_cost["analytic_gflops"],
                                         additional_gflops=costs[0],mask="exclude via attention and mean pooling"))
    print("Masked",split,len(y),"functions",len(rows),"matched extra windows",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("public","project"))
    run(parser.parse_args().split)
