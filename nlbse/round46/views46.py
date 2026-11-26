"""Create identical source windows for a causal decoder and the UniXcoder control."""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'round41'))
from source41 import PARTS, checksum, f, json_at, load, parts, write_json

DEST = HERE/'views'
MODEL = HERE/'model'
UNIX = ROOT/'models/unixcoder9'
REVISION = '8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301'
CAPS = (96,384)
TYPES = ('qwen','unix')
BLOCK = 1024
ASSETS = ('LICENSE','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')


def negative(frame):
    actual = frame['old']&(frame['flag']==0)
    assert np.array_equal(actual,frame['observed_regime']==0)
    return actual


def source():
    proof = json_at(MODEL/'upstream.json')
    assert proof['id']=='Qwen/Qwen2.5-Coder-0.5B' and proof['sha']==REVISION
    assert proof['cardData']['license']=='apache-2.0'
    entries = {q['rfilename']:q for q in proof['siblings']}
    from hashlib import sha1
    for name in ASSETS:
        raw = (MODEL/name).read_bytes()
        assert len(raw)==entries[name]['size']
        digest = sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        assert digest==entries[name]['blobId']
    return proof,entries


def tokenizers():
    q = AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    u = AutoTokenizer.from_pretrained(UNIX,local_files_only=True,trust_remote_code=False)
    assert q.is_fast and u.is_fast and q.eos_token_id is not None and u.eos_token_id is not None
    return q,u


def bookends(q,u):
    mode = u.convert_tokens_to_ids('<encoder-only>')
    assert mode!=u.unk_token_id
    return {'qwen':([], [q.eos_token_id]),'unix':([u.bos_token_id,mode,u.eos_token_id],[u.eos_token_id])}


def boundaries(text,tokens,cap,prefix):
    already = all(len(r)+extra<=cap for r,extra in zip(tokens,prefix,strict=True))
    if already:
        return text,True
    keep = (cap-12)//2
    left = min(r[min(len(r),keep)-1][1] if len(r) else len(text) for r in tokens)
    right = max(r[max(0,len(r)-keep)][0] if len(r) else 0 for r in tokens)
    assert left<=right
    return text[:left]+'\n'+text[right:],False


def seal(sequences):
    lengths = np.fromiter(map(len,sequences),dtype=np.int32,count=len(sequences))
    store = np.empty(lengths.sum(),np.int32)
    end = np.r_[0,np.cumsum(lengths)]
    for i,row in enumerate(sequences):
        store[end[i]:end[i+1]] = row
    return end,store,lengths


def make_views(texts,toks):
    inserts = bookends(*toks)
    original = [t(texts,add_special_tokens=False,return_offsets_mapping=True,verbose=False) for t in toks]
    buffer = {}
    full = {}
    for cap in CAPS:
        windows = [boundaries(text,pair,cap,(1,4)) for text,pair in
                   zip(texts,zip(*(x['offset_mapping'] for x in original),strict=True),strict=True)]
        view,kept = zip(*windows,strict=True)
        full[cap] = kept
        for name,t in zip(TYPES,toks,strict=True):
            raw = t(list(view),add_special_tokens=False,verbose=False)['input_ids']
            a,b = inserts[name]
            packed = [a+z+b for z in raw]
            if any(len(z)>cap for z in packed):
                raise ValueError('Common text view exceeded the planned token cap')
            buffer[name,cap] = packed
    return buffer,full


def main():
    proof,_ = source()
    manifests = {}
    chunks = []
    for part in PARTS:
        z,_,mark = load(part)
        manifests[part] = mark
        for r in (z['oof'],*parts(z)):
            chunks.append(r['ids'][negative(r)])
    ids = np.unique(np.concatenate(chunks))
    records = f.inputs()[0]
    toks = tokenizers()
    start = perf_counter()
    buffers = {(k,c):[] for k in TYPES for c in CAPS}
    kept = {cap:[] for cap in CAPS}
    for start_row in range(0,len(ids),BLOCK):
        strings = [records[j] for j in ids[start_row:start_row+BLOCK]]
        values,coverage = make_views(strings,toks)
        for tag,items in values.items():
            buffers[tag].extend(items)
        for cap,options in coverage.items():
            kept[cap].extend(options)
        if start_row%(BLOCK*20)==0:
            print('tokenized',min(len(ids),start_row+BLOCK),round(perf_counter()-start,2),flush=True)
    cost = perf_counter()-start
    DEST.mkdir(exist_ok=True,parents=True)
    saved = {}
    for (name,cap),items in buffers.items():
        ends,pack,lengths = seal(items)
        file = DEST/f'{name}_{cap}.npz'
        np.savez_compressed(file,ends=ends,tokens=pack,lengths=lengths,full=kept[cap])
        saved[f'{name}_{cap}'] = dict(sha256=checksum(file),tokens=int(lengths.sum()),
                  full_source=int(sum(kept[cap])),length_quantiles=np.quantile(lengths,[.5,.9,1.]).tolist())
    roster=DEST/'rows.npz'
    np.savez_compressed(roster,ids=ids)
    receipts = dict(rows=len(ids),matrix=manifests,revision=REVISION,tokenize_seconds=cost,
                    upstream_sha256=checksum(MODEL/'upstream.json'),
                    assets={k:checksum(MODEL/k) for k in ASSETS},roster_sha256=checksum(roster),
                    unix_tokenizer={p.name:checksum(p) for std in ('*.json','merges.txt','vocab.*') for p in UNIX.glob(std)},
                    windows=saved,code_sha256=checksum(Path(__file__)))
    write_json(DEST/'receipt.json',receipts)
    print(receipts,flush=True)


if __name__=='__main__':
    main()
