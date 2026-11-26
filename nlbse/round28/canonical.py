"""Cache identifier-normalized lexical features for an independent classifier."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
from scipy.sparse import save_npz, vstack

from pairs import HERE, f, checksum, json_at, write_json
from normalize_code import batch as canonicalize
from sparse_model import vectorize

CACHE = HERE/'canonical'
PATH = CACHE/'matrix.npz'


def vectors(texts):
    return vectorize(canonicalize(texts),'token')


def cache(workers=5):
    texts,y,_,_,_=f.inputs()
    source=f.ROOT/'data/raw/train-00000-of-00001.parquet'
    expected=json_at(f.R19/'partition.json')['train_sha256']
    if checksum(source)!=expected:
        raise ValueError('Training corpus changed')
    start=time.perf_counter()
    slices=[texts[i:i+1024] for i in range(0,len(texts),1024)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        blocks=list(pool.map(vectors,slices,chunksize=2))
    x=vstack(blocks,format='csr')
    CACHE.mkdir(exist_ok=True)
    save_npz(PATH,x)
    result=dict(rows=x.shape[0],columns=x.shape[1],nnz=x.nnz,matrix_sha256=checksum(PATH),
                source_sha256=expected,code_sha256=checksum(Path(__file__)),seconds=time.perf_counter()-start)
    write_json(CACHE/'manifest.json',result)
    print(result,flush=True)


if __name__=='__main__':
    cache()
