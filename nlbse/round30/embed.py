"""Freeze task-independent UniXcoder representations of the training corpus."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, RobertaModel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'round26')]
from disagreement import f, checksum, json_at, write_json
from train import cohorts
from students import ordered_batches

CACHE=HERE/'embeddings'
MODEL=ROOT/'models/unixcoder9'
CAP=384


def tokens(texts,tok):
    prefix=[tok.bos_token_id,tok.convert_tokens_to_ids('<encoder-only>'),tok.eos_token_id]
    if prefix[1]==tok.unk_token_id:
        raise ValueError('Encoder-only token missing')
    rows=tok(texts,truncation=True,max_length=CAP-4,add_special_tokens=False)['input_ids']
    return [[*prefix,*r,tok.eos_token_id] for r in rows]


def collect():
    texts,y,_,_,_=f.inputs()
    weak,truth,digest=cohorts()
    ids=np.flatnonzero(weak==0)
    assert np.array_equal(y[ids,0],truth[ids])
    src=MODEL/'pytorch_model.bin'
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    cfg._attn_implementation='sdpa'
    model=RobertaModel.from_pretrained(MODEL,config=cfg,local_files_only=True,
                                       add_pooling_layer=False).cuda().eval()
    CACHE.mkdir(exist_ok=True)
    tmp=CACHE/'features.part.npy'
    raw=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=(len(ids),2*cfg.hidden_size))
    block=4096
    start=time.perf_counter()
    def prepare(offset):
        return tokens([texts[i] for i in ids[offset:offset+block]],tok)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for offset,seq in ordered_batches(range(0,len(ids),block),prepare,True):
            for group in f.base.batch_plan(list(range(len(seq))),seq,96):
                ids_cuda,mask=f.base.batch(group,seq,tok.pad_token_id)
                b=model(input_ids=ids_cuda,attention_mask=mask).last_hidden_state
                m=mask.bool().unsqueeze(-1)
                pooled=(b.masked_fill(~m,0).sum(1)/m.sum(1))
                peak=b.masked_fill(~m,-torch.inf).amax(1)
                data=torch.cat((pooled,peak),1).to(torch.float16).cpu().numpy()
                raw[offset+np.asarray(group)]=data
            if offset%(block*4)==0:
                print(offset,len(ids),round(time.perf_counter()-start,2),flush=True)
    raw.flush()
    del raw
    destination=CACHE/'features.npy'
    tmp.replace(destination)
    archived=CACHE/'rows.npz'
    np.savez_compressed(archived,ids=ids,y=truth[ids])
    record=dict(rows=len(ids),dimensions=2*cfg.hidden_size,layers=cfg.num_hidden_layers,
                cap=CAP,seconds=time.perf_counter()-start,source_model_sha256=checksum(src),
                cohort_sha256=digest,data_sha256=checksum(destination),rows_sha256=checksum(archived))
    write_json(CACHE/'manifest.json',record)
    print(record,flush=True)


if __name__=='__main__':
    collect()
