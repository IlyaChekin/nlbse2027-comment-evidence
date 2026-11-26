"""Compare the accelerated exact scanner with every verified training count."""
import argparse
import time

import numpy as np

from evaluate_rules import HERE,WEAK,base,inputs,json_at,raw_rule_counts,scanner,write_json
from runtime import scan_chunk


def profile():
    texts,_,_,_,split=inputs()
    rng=np.random.default_rng(812)
    rows=np.sort(rng.choice(split["fit"],1200,replace=False))
    sample=[texts[i] for i in rows]
    profiles={}
    truth=raw_rule_counts()[rows]
    for name,program in (("original",WEAK/"scan.mjs"),("fast",HERE/"scan.mjs")):
        T=[]
        for _ in range(3):
            start=time.perf_counter()
            columns,data=scan_chunk(sample,program=program)
            T.append(time.perf_counter()-start)
            if not np.array_equal(data,truth):
                raise ValueError(f"Profiler sample disagrees with verified rule counts: {name}")
        profiles[name]=T
        print(name,T,flush=True)
    write_json(HERE/"scan_timing.json",dict(rows=rows.tolist(),timings=profiles,compared=len(rows)*19,
              fast_sha256=base.checksum(HERE/"scan.mjs"),original_sha256=base.checksum(WEAK/"scan.mjs")))


def verify():
    texts,_,_,_,_=inputs()
    target=raw_rule_counts()
    expected=json_at(WEAK/"train_parallel.json")
    changed=np.zeros(target.shape[1],dtype=np.int64)
    start=time.perf_counter()
    for start_row in range(0,len(texts),16000):
        chunk=texts[start_row:start_row+16000]
        keys,compiled=scanner(chunk,program=HERE/"scan.mjs")
        if keys!=expected["columns"]:
            raise ValueError("The scanner returned different rule names")
        prior=target[start_row:start_row+len(chunk)]
        changed+=np.count_nonzero(prior!=compiled,axis=0)
        print(start_row+len(chunk),"rows",int(changed.sum()),"differences",flush=True)
    write_json(HERE/"scan_equivalence.json",dict(rows=len(texts),cells=target.size,changes=int(changed.sum()),
                by_rule=dict(zip(keys,changed.tolist(),strict=True)),seconds=time.perf_counter()-start,
                data_sha256=expected["sha256"],program_sha256=base.checksum(HERE/"scan.mjs")))
    if changed.any():
        raise ValueError("Optimized scanner is not interchangeable with its reference")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("profile","verify"))
    a=p.parse_args()
    {"profile":profile,"verify":verify}[a.stage]()
