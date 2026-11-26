"""Replay the published CodeT5+ embedding with native Transformers and pinned weights."""
import argparse
from hashlib import sha1
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize
from transformers import RobertaTokenizerFast, T5Config, T5EncoderModel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'round46'))
from views46 import DEST as WINDOWS, f, checksum, json_at, write_json, CAPS, boundaries, seal, tokenizers

MODEL=HERE/'model'
DEST=HERE/'embeddings'
REVISION='d9f3a534af4252f04b10cd9f78e05037542d10f6'
ASSETS=('config.json','added_tokens.json','merges.txt','special_tokens_map.json','tokenizer.json',
        'tokenizer_config.json','vocab.json','modeling_codet5p_embedding.py','configuration_codet5p_embedding.py')
BLOCK=2048


def source():
    reference=json_at(MODEL/'upstream.json')
    assert reference['sha']==REVISION and reference['id']=='Salesforce/codet5p-110m-embedding'
    assert reference['cardData']['license']=='bsd-3-clause'
    entries={x['rfilename']:x for x in reference['siblings']}
    for k in ASSETS:
        payload=(MODEL/k).read_bytes()
        assert entries[k]['size']==len(payload)
        assert entries[k]['blobId']==sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    return entries


def local_tokenizer():
    q=RobertaTokenizerFast.from_pretrained(MODEL,local_files_only=True)
    assert q.is_fast and (q.bos_token_id,q.eos_token_id,q.pad_token_id)==(1,2,0)
    return q


def windows(texts,controls,encoder,caps=CAPS):
    original=[q(texts,add_special_tokens=False,return_offsets_mapping=True,verbose=False)['offset_mapping']
              for q in controls]
    result={}
    for cap in caps:
        shared,complete=zip(*(boundaries(row,spans,cap,(1,4)) for row,spans in
                             zip(texts,zip(*original,strict=True),strict=True)),strict=True)
        result[cap]=encoder(list(shared),add_special_tokens=True,verbose=False)['input_ids'],complete
    return result


def pack():
    assets=source()
    manifest=json_at(WINDOWS/'receipt.json')
    assert manifest['roster_sha256']==checksum(WINDOWS/'rows.npz')
    index=np.load(WINDOWS/'rows.npz')['ids']
    docs=f.inputs()[0]
    possible=tokenizers()
    tokenizer=local_tokenizer()
    started=perf_counter()
    storage={k:[] for k in CAPS}
    seen={k:[] for k in CAPS}
    for i in range(0,len(index),BLOCK):
        text=[docs[row] for row in index[i:i+BLOCK]]
        for cap,(rows,full) in windows(text,possible,tokenizer).items():
            storage[cap].extend(rows)
            seen[cap].extend(full)
        if i%(10*BLOCK)==0:
            print('tokenized',min(i+BLOCK,len(index)),round(perf_counter()-started,2),flush=True)
    seconds=perf_counter()-started
    DEST.mkdir(exist_ok=True,parents=True)
    report={}
    for cap in CAPS:
        ref=WINDOWS/f'qwen_{cap}.npz'
        assert manifest['windows'][f'qwen_{cap}']['sha256']==checksum(ref)
        with np.load(ref) as x:
            assert np.array_equal(x['full'],seen[cap])
        cuts,flat,lengths=seal(storage[cap])
        p=DEST/f'tokens_{cap}.npz'
        np.savez_compressed(p,ends=cuts,tokens=flat,lengths=lengths,full=seen[cap])
        assert min(lengths)>=2
        report[str(cap)]=dict(file_sha256=checksum(p),full=int(sum(seen[cap])),median=float(np.median(lengths)),
                           maximum=int(max(lengths)),total_tokens=int(lengths.sum()))
    write_json(DEST/'views.json',dict(rows=len(index),window_receipt=checksum(WINDOWS/'receipt.json'),
                 roster_sha256=manifest['roster_sha256'],revision=REVISION,views=report,seconds=seconds,
                 assets={k:checksum(MODEL/k) for k in ASSETS},upstream=checksum(MODEL/'upstream.json'),
                 lfs=assets['pytorch_model.bin']['lfs']['sha256'],code_sha256=checksum(Path(__file__))))


def encoder():
    entries=source()
    weights=MODEL/'pytorch_model.bin'
    expected=entries[weights.name]['lfs']['sha256']
    assert weights.stat().st_size==entries[weights.name]['size'] and checksum(weights)==expected
    raw=json_at(MODEL/'config.json')
    assert raw['model_type']=='codet5p_embedding' and raw['embed_dim']==256
    config=T5Config.from_dict({k:v for k,v in raw.items() if k not in ('model_type','auto_map','architectures')})
    config._attn_implementation='eager'
    config.use_cache=False
    pretrained=torch.load(weights,map_location='cpu',weights_only=True)
    z={k:pretrained.pop('proj.'+k) for k in ('weight','bias')}
    if 'encoder.embed_tokens.weight' not in pretrained:
        pretrained['encoder.embed_tokens.weight']=pretrained['shared.weight']
    assert torch.equal(pretrained['shared.weight'],pretrained['encoder.embed_tokens.weight'])
    model=T5EncoderModel(config)
    model.load_state_dict(pretrained,strict=True)
    projection=nn.Linear(config.d_model,raw['embed_dim'])
    projection.load_state_dict(z,strict=True)
    return model.eval().cuda(),projection.eval().cuda(),expected


@torch.inference_mode()
def verify_example(model,projection,tok):
    x=tok("def print_hello_world():\tprint('Hello World!')",return_tensors='pt').to('cuda')
    value=normalize(projection(model(**x).last_hidden_state[:,0,:]),dim=-1)[0]
    expected=torch.tensor([.0185,.0229,-.0315,-.0307,-.1421,-.0575,-.0275],device='cuda')
    error=float(abs(expected-value[:len(expected)]).max())
    assert error<.0008, (error,value[:len(expected)])
    return error


@torch.inference_mode()
def encode(sequence,model,projection,cap,projected,raw):
    # Archive native first-token hidden states alongside the normalized projection.
    started=perf_counter()
    charge,flops=0.,0.
    cfg=model.config
    batch_size=128 if cap==96 else 64
    for first in range(0,len(sequence),4096):
        block=sequence[first:first+4096]
        for take in f.base.batch_plan(list(range(len(block))),block,batch_size):
            x,mask=f.base.batch(take,block,cfg.pad_token_id)
            tick=perf_counter()
            hidden=model(input_ids=x,attention_mask=mask).last_hidden_state[:,0,:]
            embed=normalize(projection(hidden).float(),dim=-1)*16.
            projected[first+np.asarray(take)]=embed.half().cpu().numpy()
            raw[first+np.asarray(take)]=hidden.half().cpu().numpy()
            charge+=perf_counter()-tick
            # T5 encoder has two feed-forward matrices and full attention.
            n,l=x.shape
            d=cfg.d_model
            flop=2*n*cfg.num_layers*(4*l*d*d+2*l*d*cfg.d_ff+2*l*l*d) + 2*n*d*256
            flops+=flop/1e9
        if first%(5*4096)==0:
            print('embedding',cap,min(first+4096,len(sequence)),round(perf_counter()-started,2),flush=True)
    return dict(forward_seconds=charge,wall_seconds=perf_counter()-started,analytic_gflops=flops)


def extract(cap):
    manifest=json_at(DEST/'views.json')
    assert manifest['code_sha256']==checksum(Path(__file__))
    assert manifest['window_receipt']==checksum(WINDOWS/'receipt.json')
    p=DEST/f'tokens_{cap}.npz'
    assert manifest['views'][str(cap)]['file_sha256']==checksum(p)
    model,projection,weight=encoder()
    assert weight==manifest['lfs']
    error=verify_example(model,projection,local_tokenizer())
    model.to(dtype=torch.bfloat16)
    projection.to(dtype=torch.bfloat16)
    with np.load(p) as z:
        ends,flat=z['ends'],z['tokens']
    seq=[flat[a:b] for a,b in zip(ends[:-1],ends[1:],strict=True)]
    path=DEST/str(cap)
    path.mkdir(exist_ok=True,parents=True)
    outputs={'projection.npy':(len(seq),256),'raw.npy':(len(seq),model.config.d_model)}
    buffers={n:np.lib.format.open_memmap(path/(n+'.partial'),mode='w+',dtype=np.float16,shape=k)
             for n,k in outputs.items()}
    cost=encode(seq,model,projection,cap,buffers['projection.npy'],buffers['raw.npy'])
    for name,store in buffers.items():
        store.flush()
        (path/(name+'.partial')).replace(path/name)
    info=dict(rows=len(seq),model_sha256=weight,views_sha256=checksum(DEST/'views.json'),example_max_error=error,
              dimensions=outputs,cost=cost,code_sha256=checksum(Path(__file__)),
              hashes={name:checksum(path/name) for name in outputs},dtype='bfloat16')
    write_json(path/'receipt.json',info)
    print(info,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('pack','96','384'))
    mode=p.parse_args().stage
    pack() if mode=='pack' else extract(int(mode))
