"""Recover separate annotation components on the signed official training rows."""
from collections import defaultdict
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from layout49 import HERE, ROOT, f, checksum, json_at, write_json
from project_split import TOKEN, family

DEST = HERE/'components'
SOURCE = ROOT/'round48/madewic/v3.zip'
ALIGNED = ROOT/'round48/madewic/results.json'
COLS = ('W','Big-Vul','Devign')
COLLECTIONS = ('devign','big-vul','OSPR')


def key(project,path):
    return family(project or ''),str(PurePosixPath(path.replace('\\','/'))) if path else ''


def signature(code):
    terms = (m[0] for m in TOKEN.finditer(code) if m['comment'] is None)
    payload = '\0'.join(terms).encode('utf-8','surrogateescape')
    return hashlib.blake2b(payload,digest_size=24).digest()


def transform(row, names):
    return sum((row[name]=='1')<<COLS.index(name) for name in names)


def annotations():
    receipt=json_at(ALIGNED)
    assert checksum(SOURCE)==receipt['sources']['v3']['archive_sha256']
    manifest=json_at(ROOT/'data/manifest.json')['files'][0]
    texts, labels, project, files, _=f.inputs()
    assert len(texts)==manifest['rows']==214586
    assert checksum(ROOT/'data/raw/train-00000-of-00001.parquet')==manifest['sha256']
    signatures=defaultdict(dict)
    for i,(text,p,path) in enumerate(zip(texts,project,files,strict=True)):
        location=key(p,path)
        if all(location):
            signatures[location].setdefault(signature(text),[]).append(i)
    print('indexed training',len(signatures),flush=True)
    evidence=np.zeros(len(texts),np.uint16)
    matches=np.zeros(len(texts),np.int32)
    sets={}
    start=perf_counter()
    csv.field_size_limit(32*1024*1024)
    with ZipFile(SOURCE) as archive:
        for name in COLLECTIONS:
            records=0
            accepted=0
            file=archive.getinfo(f'MADE-WIC/Dataset/{name}/complete.csv')
            with io.TextIOWrapper(archive.open(file),encoding='utf-8-sig',errors='surrogateescape',newline='') as raw:
                rows=csv.DictReader(raw)
                active=tuple(k for k in COLS if k in rows.fieldnames)
                assert 'W' in active
                for item in rows:
                    records+=1
                    location=key(item['Projectname'],item['Filepath'])
                    known=signatures.get(location)
                    if not known:
                        continue
                    ids=known.get(signature(item['Function']))
                    if not ids:
                        continue
                    assert all(item[k] in ('0','1') for k in active)
                    component=transform(item,active)
                    evidence[ids] |= 1<<component
                    matches[ids] += 1
                    accepted+=1
            sets[name]=dict(archive_rows=records,source_matches=accepted,crc32=f'{file.CRC:08x}')
            print(name,sets[name],round(perf_counter()-start,2),flush=True)
    trusted=(evidence>0)&((evidence & (evidence-1))==0)
    signature_value = np.full(len(texts),-1,np.int8)
    for k in range(8):
        signature_value[trusted&(evidence==1<<k)]=k
    different = trusted & ((signature_value>0)!=labels[:,0].astype(bool))
    unique_and_agrees = trusted & ~different
    ready=DEST/'labels.npz'
    np.savez_compressed(ready,mask=evidence,count=matches,component=signature_value,
                        selected=unique_and_agrees,truth=labels[:,0].astype(bool))
    counts={str(i):int((unique_and_agrees&(signature_value==i)).sum()) for i in range(8)}
    report=dict(rows=len(texts),columns=COLS,matched=int((evidence>0).sum()),unique_components=int(trusted.sum()),
                label_conflicts=int(different.sum()),usable=int(unique_and_agrees.sum()),counts=counts,
                sets=sets,training_sha256=manifest['sha256'],archive_sha256=checksum(SOURCE),
                previous_source_sha256=checksum(ALIGNED),predictions_sha256=checksum(ready),
                code_sha256=checksum(Path(__file__)),seconds=perf_counter()-start)
    write_json(DEST/'receipt.json',report)
    print(report,flush=True)


if __name__=='__main__':
    DEST.mkdir(exist_ok=True)
    annotations()
