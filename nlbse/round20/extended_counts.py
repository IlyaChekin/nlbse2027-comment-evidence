"""Cache the additional passive rule with the bounded numeric-literal interpreter."""
import csv
import time

import numpy as np

from evaluate_rules import HERE,WEAK,base,inputs,json_at,scanner,write_json


if __name__=="__main__":
    texts,_,_,_,_=inputs()
    start=time.perf_counter()
    portions=[]
    for offset in range(0,len(texts),16000):
        keys,features=scanner(texts[offset:offset+16000],only=("195",),program=HERE/"scan_extended.mjs")
        if keys!=["rule_195"] or features.shape[1]!=1:
            raise ValueError("The added rule has unexpected columns")
        portions.append(features[:,0])
        print(offset+len(features),"rows",int((features>0).sum()),"hits in chunk",flush=True)
    rows=np.concatenate(portions)
    out=HERE/"extended_counts.csv"
    with out.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("row","rule_195"))
        w.writerows(enumerate(rows))
    write_json(HERE/"extended_counts.json",dict(rows=len(rows),hits=int((rows>0).sum()),
        failures=int((rows<0).sum()),secs=time.perf_counter()-start,sha256=base.checksum(out),
        implementation_sha256=base.checksum(HERE/"scan.mjs"),program_sha256=base.checksum(HERE/"scan_extended.mjs"),
        original_counts_sha256=json_at(WEAK/"train_parallel.json")["sha256"]))
