"""Prepare independent localization targets for held-out reporting."""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from span55 import HERE, ROOT, PARTS, checksum, f, json_at, write_json
sys.path.insert(0,str(ROOT/'round54'))
from spans54 import mapped, scan, PROGRAM
from causal54 import DEST as CAUSAL, stored_pairs

DEST=HERE/'held_spans'


def collect():
    DEST.mkdir(exist_ok=True)
    pair,card=stored_pairs()
    roster={f'{part}/{phase}':set(card['cohorts'][f'{part}/{phase}'])&pair.keys()
            for part in PARTS for phase in ('cal','held')}
    for part in PARTS:
        assert not (roster[part+'/cal']|roster[part+'/held']) & set(card['cohorts'][part+'/fit'])
    oracle=CAUSAL/'oracle.npz'
    assert json_at(CAUSAL/'oracle.json')['oracle_sha256']==checksum(oracle)
    with np.load(oracle) as z:
        raw={int(i):b for i,b in zip(z['ids'],z['original'],strict=True) if b[0]==1}
    rows=np.array(sorted(set.union(*roster.values())&raw.keys()),dtype=np.int64)
    before=[pair[i]['original'] for i in rows]
    t=perf_counter()
    result=scan(before)
    cost=perf_counter()-t
    assert np.array_equal(result[:,:3],np.array([raw[i] for i in rows]))
    current=f.inputs()[0]
    target=np.full((len(rows),2,2),-1,dtype=np.int32)
    for j,i in enumerate(rows):
        if result[j,8]==before[j].count('\n')+1:
            target[j]=np.array(mapped(before[j],current[i],result[j,3:5]))[:,2:4]
    positions={k.replace('/','_'):np.isin(rows,list(v)) for k,v in roster.items()}
    name=DEST/'ground.npz'
    np.savez_compressed(name,ids=rows,targets=target,original=result,**positions)
    status={k:dict(rows=int(v.sum()),mapped=int((v&(target[:,0,0]>=0)).sum()),
                   initially_missed=int(sum(pair[i]['baseline']==0 for i in rows[v]))) for k,v in positions.items()}
    write_json(DEST/'receipt.json',dict(data_sha256=checksum(name),code_sha256=checksum(Path(__file__)),
               pairing_sha256=checksum(CAUSAL/'pairs.json'), scanner_sha256=checksum(PROGRAM),
               funds=status,seconds=cost,total=len(rows)))
    print(status,flush=True)


if __name__=='__main__':
    collect()
