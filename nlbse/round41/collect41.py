"""Cache forward-only GraphCodeBERT representations for signed experiment rows."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, RobertaModel

from source41 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, load, supported
from students import ordered_batches

MODEL=ROOT/'models/graphcodebert'
DEST=HERE/'graph'
CAP=192
BLOCK=2048
BATCH=112


def network():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    config._attn_implementation='sdpa'
    model=RobertaModel.from_pretrained(MODEL,config=config,local_files_only=True,
                                      add_pooling_layer=False).cuda().eval()
    return tok,model


def tokenize(text,tok):
    # Preserve both ends under one fixed token budget.
    inputs=tok(text,add_special_tokens=True,truncation=False,verbose=False)['input_ids']
    h=CAP//2
    return [row if len(row)<=CAP else row[:h]+row[-(CAP-h):] for row in inputs]


def operations(cfg,n,seq):
    d=cfg.hidden_size
    return n*cfg.num_hidden_layers*2.*(4*seq*d*d+2*seq*seq*d+
                                        2*seq*d*cfg.intermediate_size)/1e9


@torch.inference_mode()
def encode(rows,tok,net,target):
    start=perf_counter()
    charged=0.
    ops=0.
    def prepare(first):
        return tokenize(rows[first:first+BLOCK],tok)
    with torch.autocast('cuda',dtype=torch.float16):
        for first,seq in ordered_batches(range(0,len(rows),BLOCK),prepare,True):
            for take in f.base.batch_plan(list(range(len(seq))),seq,BATCH):
                ids,mask=f.base.batch(take,seq,tok.pad_token_id)
                started=perf_counter()
                hidden=net(input_ids=ids,attention_mask=mask).last_hidden_state
                valid=mask.bool()[:,:,None]
                mean=hidden.masked_fill(~valid,0).sum(1)/valid.sum(1)
                maximum=hidden.masked_fill(~valid,-torch.inf).amax(1)
                z=torch.cat((mean,maximum),1).half().cpu().numpy()
                charged+=perf_counter()-started
                ops+=operations(net.config,*ids.shape)
                target[first+np.asarray(take)]=z
    return dict(forward_seconds=charged,wall_seconds=perf_counter()-start,analytic_gflops=ops)


def main():
    DEST.mkdir(exist_ok=True,parents=True)
    plans={}
    index=[]
    for rotation in PARTS:
        rows,_,proof=load(rotation)
        plans[rotation]=proof
        for frame in rows.values():
            index.append(frame['ids'][supported(frame)])
    ids=np.unique(np.concatenate(index))
    names=f.inputs()[0]
    tok,net=network()
    file=DEST/'features.npy'
    pending=file.with_suffix('.partial.npy')
    target=np.lib.format.open_memmap(pending,mode='w+',dtype=np.float16,
                         shape=(len(ids),2*net.config.hidden_size))
    price=encode([names[i] for i in ids],tok,net,target)
    target.flush()
    del target
    pending.replace(file)
    roster=DEST/'rows.npz'
    np.savez_compressed(roster,ids=ids)
    proof=dict(rows=len(ids),cap=CAP,model='GraphCodeBERT',phases=plans,cost=price,
               features_sha256=checksum(file),rows_sha256=checksum(roster),
               model_sha256=checksum(MODEL/'pytorch_model.bin'),
               code_sha256=checksum(Path(__file__)))
    write_json(DEST/'receipt.json',proof)
    print(proof,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
