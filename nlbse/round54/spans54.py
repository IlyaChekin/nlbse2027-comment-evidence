"""Map scanner evidence from original fit text into contest token coordinates."""
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from causal54 import DEST as CAUSAL, HERE, ROOT, PARTS, checksum, f, json_at, spans, stored_pairs, write_json

sys.path.insert(0,str(ROOT/'round39'))
from location39 import PROGRAM, scan

DEST=HERE/'spans'


def mapped(original,text,lines):
    source=tuple(m for m in spans(original) if m['comment'] is None)
    target=tuple(m for m in spans(text) if m['comment'] is None)
    if tuple(x[0] for x in source)!=tuple(x[0] for x in target):
        raise ValueError('Source and target tokens changed')
    starts=np.fromiter((m.start() for m in source),np.int32)
    newline=np.flatnonzero(np.frombuffer(original.encode('utf-32-le'),dtype='<u4')==ord('\n'))
    assert len(newline)==original.count('\n')
    line=np.searchsorted(newline,starts,side='right')+1
    out=[]
    for wanted in lines:
        ids=np.flatnonzero(line==wanted) if wanted else np.empty(0,dtype=np.int64)
        if not len(ids):
            out.append((-1,-1,-1,-1))
        else:
            a,b=ids[0],ids[-1]
            out.append((a,b+1,target[a].start(),target[b].end()))
    return out


def main():
    DEST.mkdir(exist_ok=True)
    paired,plan=stored_pairs()
    oracle=CAUSAL/'oracle.npz'
    assert checksum(oracle)==json_at(CAUSAL/'oracle.json')['oracle_sha256']
    assert checksum(PROGRAM)==json_at(ROOT/'round39/locations/receipt.json')['extended_scanner_sha256']
    fit={p:set(plan['cohorts'][p+'/fit']) & paired.keys() for p in PARTS}
    for part in PARTS:
        assert not fit[part] & set(plan['cohorts'][part+'/cal'])
        assert not fit[part] & set(plan['cohorts'][part+'/held'])
    rows=np.array(sorted(set.union(*fit.values())),dtype=np.int64)
    original=[paired[i]['original'] for i in rows]
    started=perf_counter()
    values=scan(original)
    elapsed=perf_counter()-started
    text=f.inputs()[0]
    with np.load(oracle) as z:
        loc=np.searchsorted(z['ids'],rows)
        assert np.array_equal(z['ids'][loc],rows)
        assert np.array_equal(values[:,:3],z['original'][loc])
        old=z['baseline'][loc]
        gold=z['gold'][loc]
    regions=np.full((len(rows),2,4),-1,dtype=np.int32)
    for i,row in enumerate(rows):
        if values[i,0]==1 and values[i,8]==original[i].count('\n')+1:
            regions[i]=mapped(original[i],text[row],values[i,3:5])
    indexes={p:np.isin(rows,list(v)) for p,v in fit.items()}
    notes={}
    for p,where in indexes.items():
        positive=where & (values[:,0]==1)
        hidden=positive & (old[:,0]==0)
        located=regions[:,0,0]>=0
        notes[p]=dict(rows=int(where.sum()),positive=int(positive.sum()),located=int((positive&located).sum()),
                      hidden=int(hidden.sum()),hidden_located=int((hidden&located).sum()),
                      hidden_by_rule={int(k):int(v) for k,v in Counter(values[hidden&located,1]).items()})
    file=DEST/'targets.npz'
    np.savez_compressed(file,ids=rows,exact_source=values,visible_scanner=old,archival_w=gold,
                        spans=regions,**{p+'_fit': mask for p,mask in indexes.items()})
    write_json(DEST/'receipt.json',dict(pairs_sha256=checksum(CAUSAL/'pairs.json'),scanner_sha256=checksum(PROGRAM),
               oracle_sha256=checksum(oracle),code_sha256=checksum(Path(__file__)),file_sha256=checksum(file),
               aux_shape=regions.shape,scan_seconds=elapsed,parts=notes, total_rows=len(rows)))
    print(notes,flush=True)


if __name__=='__main__':
    main()
