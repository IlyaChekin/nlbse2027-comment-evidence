"""Measure workload balancing for the existing scanner with identical outputs."""
from concurrent.futures import ThreadPoolExecutor
from heapq import heapify, heappop, heappush
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from study50 import HERE, ROOT, checksum, json_at, write_json, f
from first_hit32 import SCRIPT, scan
from runtime import scan_chunk

DEST=HERE/'schedule'


def balanced(texts, only=None, program=None):
    workers=max(1,min(int(os.environ.get('NLBSE_SCANNER_WORKERS','6')),len(os.sched_getaffinity(0)),len(texts)//512))
    if workers==1:
        return scan_chunk(texts,only,program)
    count=max(workers,(len(texts)+1599)//1600)
    groups=[[] for _ in range(count)]
    heap=[(0,i) for i in range(count)]
    heapify(heap)
    costs=[len(s)+64*(s.count('\n')+1) for s in texts]
    for i in sorted(range(len(texts)),key=costs.__getitem__,reverse=True):
        owed,slot=heappop(heap)
        groups[slot].append(i)
        if len(groups[slot])<1600:
            heappush(heap,(owed+costs[i],slot))
    for roster in groups:
        roster.sort()

    def task(roster):
        return scan_chunk([texts[k] for k in roster],only,program)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        result=list(pool.map(task,groups))
    columns=result[0][0]
    values=np.empty((len(texts),len(columns)),np.int32)
    for k,(key,z) in zip(groups,result,strict=True):
        assert key==columns
        values[k]=z
    return columns,values


def main():
    DEST.mkdir(exist_ok=True)
    loc=ROOT/'round39/locations/training.npz'
    proof=json_at(loc.with_name('receipt.json'))
    assert proof['predictions_sha256']==checksum(loc)
    assert proof['original_scanner_sha256']==checksum(SCRIPT)
    corpus=f.inputs()[0]
    with np.load(loc) as source:
        ids,values=source['ids'],source['values']
    seed=5099
    allowed=np.random.default_rng(seed).choice(len(ids),15000,replace=False)
    rosters=[np.sort(a) for a in np.split(allowed,2)]
    report={}
    for i,idx in enumerate(rosters):
        chosen=[corpus[j] for j in ids[idx]]
        expected=values[idx,:3]
        for test in (False,True):
            z=balanced(chosen[:768],program=SCRIPT)[1] if test else scan(chosen[:768],True)[0]
            assert np.array_equal(z,expected[:768])
        speed={'reference':[],'balanced':[]}
        wins=[]
        for k in range(4):
            measured={}
            for test in ((False,True) if k%2==0 else (True,False)):
                t=perf_counter()
                if test:
                    _,score=balanced(chosen,program=SCRIPT)
                else:
                    score,_=scan(chosen,True)
                elapsed=perf_counter()-t
                assert np.array_equal(score,expected)
                name='balanced' if test else 'reference'
                speed[name].append(elapsed)
                measured[name]=elapsed
            wins.append(measured['reference']-measured['balanced'])
            print(i,k,wins[-1],flush=True)
        report[str(i)]=dict(times=speed,pairs=wins,median_saving=median(wins),
                            wins=sum(t>0 for t in wins),rows=len(idx))
    p=DEST/'sets.npz'
    np.savez_compressed(p,ids=ids[np.asarray(rosters)])
    incumbent=ROOT/'round38/release_summary.json'
    record=dict(study=report,qualified=all(v['wins']>=3 and v['median_saving']>.10 for v in report.values()),
                row_sha256=checksum(p),code_sha256=checksum(Path(__file__)),scanner_sha256=checksum(SCRIPT),
                source_sha256=checksum(loc),seed=seed,workers=min(6,len(os.sched_getaffinity(0))),
                anchor_sha256=checksum(incumbent))
    write_json(DEST/'results.json',record)
    print(record,flush=True)


if __name__=='__main__':
    main()
