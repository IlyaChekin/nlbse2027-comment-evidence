"""Recover the availability mask for external annotation on the official train."""
from collections import defaultdict
import csv
import io
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from train51 import HERE, ROOT, COMPONENTS, f, checksum, json_at, write_json
from components49 import SOURCE, key, signature, COLS, transform

DEST=HERE/'support'
COLLECTIONS={'devign':1,'big-vul':2}


def main():
    DEST.mkdir(exist_ok=True)
    mark=COMPONENTS/'receipt.json'
    plan=json_at(mark)
    assert checksum(mark.with_name('labels.npz'))==plan['predictions_sha256']
    assert checksum(SOURCE)==plan['archive_sha256']
    with np.load(mark.with_name('labels.npz')) as z:
        components,known=z['component'],z['selected']
    texts,labels,project,files,_=f.inputs()
    assert len(texts)==plan['rows']==214586
    context=defaultdict(dict)
    for i in np.flatnonzero(known):
        place=key(project[i],files[i])
        assert all(place)
        context[place].setdefault(signature(texts[i]),[]).append(i)
    supplied=np.zeros(len(texts),np.uint8)
    note={}
    t0=perf_counter()
    csv.field_size_limit(32*1024*1024)
    with ZipFile(SOURCE) as store:
        for corpus,bit in COLLECTIONS.items():
            matched=0
            count=0
            entry=store.getinfo(f'MADE-WIC/Dataset/{corpus}/complete.csv')
            with io.TextIOWrapper(store.open(entry),encoding='utf-8-sig',errors='surrogateescape',newline='') as source:
                rows=csv.DictReader(source)
                fields=tuple(c for c in COLS if c in rows.fieldnames)
                assert len(fields)==2 and 'W' in fields
                for r in rows:
                    count+=1
                    local=context.get(key(r['Projectname'],r['Filepath']))
                    if not local:
                        continue
                    link=local.get(signature(r['Function']))
                    if link:
                        code=transform(r,fields)
                        assert (components[link]==code).all()
                        supplied[link]|=bit
                        matched+=1
            note[corpus]=dict(source_rows=count,matching_matches=matched,crc32=f'{entry.CRC:08x}')
    active=supplied>0
    assert (active&~known).sum()==0
    external=(components&6)>0
    usual=(components&1)>0
    # Only the absence of an explicit column is marked unavailable.
    assert not (known&external&~active).any()
    output=DEST/'mask.npz'
    np.savez_compressed(output,source=supplied,external_available=active,component=components,available=known)
    receipt=dict(rows=len(texts),matches=note,coverage=int(active.sum()),
                 absent=int((known&~active).sum()),external_positives=int((active&external).sum()),
                 external_only=int((active&external&~usual).sum()),
                 conflict_vulnerability=int((known&((usual|external)!=(labels[:,0]>0))).sum()),
                 source_sha256=checksum(SOURCE),source_labels_sha256=checksum(mark),
                 mask_sha256=checksum(output),code_sha256=checksum(Path(__file__)),seconds=perf_counter()-t0)
    write_json(DEST/'receipt.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
