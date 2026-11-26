"""Measure a genuinely shortened decoder on a fixed corpus sample."""
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from views46 import HERE,ROOT,DEST as WINDOWS,checksum,f,json_at,make_views,tokenizers,write_json
from extract46 import encode,weights

DEST = HERE/'micro'


def main():
    start = perf_counter()
    plan = json_at(WINDOWS/'receipt.json')
    assert checksum(WINDOWS/'rows.npz')==plan['roster_sha256']
    index = np.load(WINDOWS/'rows.npz')['ids']
    sel = np.random.default_rng(4603).choice(len(index),size=1500,replace=False)
    corpus = f.inputs()[0]
    data = [corpus[k] for k in index[sel]]
    tokenizer = tokenizers()
    make_views(data[:64],tokenizer)
    production = []
    quantized = None
    for _ in range(5):
        t = perf_counter()
        quantized,_ = make_views(data,tokenizer)
        production.append(perf_counter()-t)
    for (name,cap),examples in quantized.items():
        with np.load(WINDOWS/f'{name}_{cap}.npz') as saved:
            packed_rows,ends = saved['tokens'],saved['ends']
            assert all(np.array_equal(z,packed_rows[ends[r]:ends[r+1]]) for z,r in zip(examples,sel,strict=True))
    packed = list(quantized['qwen',96])
    model,pad,asset = weights('qwen')
    width = model.config.hidden_size*3
    assert model.config.num_hidden_layers==24
    a = np.empty((128,width),np.float16)
    b = np.empty_like(a)
    c = np.empty_like(a)
    cache = np.empty((len(packed),width),np.float16)
    encode(packed[:128],model,pad,'qwen',96,a,b)
    complete = model.layers
    shorter = torch.nn.ModuleList(complete[:12])
    model.layers = shorter
    model.config.num_hidden_layers = len(shorter)
    encode(packed[:128],model,pad,'qwen',96,c)
    assert np.array_equal(b,c)
    trials = {'full':[],'partial':[]}
    layers = {'full':complete,'partial':shorter}
    for i in range(5):
        modes = ('full','partial') if i%2==0 else ('partial','full')
        for key in modes:
            model.layers = layers[key]
            model.config.num_hidden_layers = len(model.layers)
            trials[key].append(encode(packed,model,pad,'qwen',96,cache))
    DEST.mkdir(exist_ok=True,parents=True)
    rows = DEST/'rows.npz'
    np.savez_compressed(rows,ids=index[sel],packed_length=np.array(list(map(len,packed)),np.int32))
    card = dict(rows=len(data),model_sha256=asset,rows_sha256=checksum(rows),
                tokenizer_trials=production,tokenizer_median=median(production),
                forward=trials,median={k:{n:median(x[n] for x in v) for n in v[0]} for k,v in trials.items()},
                intermediate_exact_rows=len(c),same_states=np.array_equal(b,c),
                code_sha256=checksum(Path(__file__)),extractor_sha256=checksum(HERE/'extract46.py'),
                wall_seconds=perf_counter()-start)
    write_json(DEST/'receipt.json',card)
    print(card,flush=True)


if __name__=='__main__':
    main()
