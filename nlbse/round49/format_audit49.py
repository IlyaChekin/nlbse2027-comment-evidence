"""Measure the sensitivity of weak annotation replay to source formatting."""
from collections import defaultdict
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from components49 import COLS, COLLECTIONS, DEST as SOURCE_DATA, SOURCE, key, signature, transform
from route49 import HERE, f, checksum, json_at, load, write_json
from first_hit32 import SCRIPT, scan
from project_split import TOKEN

DEST = HERE/'format_audit'
MAX_GROUP = 200
SEED = 4919


def unannotated(text):
    fragments = []
    last = 0
    for span in TOKEN.finditer(text):
        if span['comment'] is None:
            continue
        fragments.extend((text[last:span.start()],''.join(c if c in '\r\n' else ' ' for c in span[0])))
        last = span.end()
    return ''.join((*fragments,text[last:]))


def sample():
    state,_,stamp = load('official')
    training=state['oof']
    corpus,labels,project,files,_=f.inputs()
    origin=SOURCE_DATA/'labels.npz'
    card=json_at(origin.with_name('receipt.json'))
    assert card['predictions_sha256']==checksum(origin)
    with np.load(origin) as raw:
        ids=training['ids']
        source=raw['component'][ids]
        permitted=raw['selected'][ids]
        assert np.array_equal(raw['truth'][ids],labels[ids,0])
    flag = training['flag']
    known=training['old'] & (flag>=0) & permitted
    w=(source&1)>0
    rng = np.random.default_rng(SEED)
    files_wanted=defaultdict(set)
    plan=[]
    counts={}
    for name, scope in (
        ('historical_positive_only',w & (flag==0)),('scanner_positive_only',~w & (flag==1)),
        ('agree_positive',w & (flag==1)),('agree_negative',~w & (flag==0))):
        at=np.flatnonzero(scope&known)
        chosen=rng.choice(at,min(MAX_GROUP,len(at)),replace=False)
        counts[name]=int(len(at))
        for i in np.sort(chosen):
            row=int(ids[i])
            location=key(project[row],files[row])
            tok=signature(corpus[row])
            files_wanted[location].add(tok)
            plan.append((row,location,tok,int(source[i]),int(flag[i]),name))
    return plan,files_wanted,corpus,counts,stamp


def from_archive(files_wanted):
    original=defaultdict(dict)
    csv.field_size_limit(32*1024*1024)
    with ZipFile(SOURCE) as archive:
        for group in COLLECTIONS:
            path=f'MADE-WIC/Dataset/{group}/complete.csv'
            with io.TextIOWrapper(archive.open(path),encoding='utf-8-sig',errors='surrogateescape',newline='') as f:
                table=csv.DictReader(f)
                fields=tuple(c for c in COLS if c in table.fieldnames)
                for row in table:
                    location=key(row['Projectname'],row['Filepath'])
                    options=files_wanted.get(location)
                    if not options:
                        continue
                    text=row['Function']
                    matches=signature(text)
                    if matches in options:
                        original[location,matches][text] = transform(row,fields)
    return original


def main():
    DEST.mkdir(exist_ok=True)
    selected, keys, corpus, population, stamp=sample()
    card=json_at(SOURCE_DATA/'receipt.json')
    assert card['archive_sha256']==checksum(SOURCE)
    source_rows=from_archive(keys)
    chosen=[]
    texts={'contest':[],'original':[],'contest_masked':[],'original_masked':[]}
    records=[]
    for row,location,match,known,flag,group in selected:
        source=source_rows[location,match]
        if len(source)!=1:
            continue
        (before,v),=source.items()
        assert v==known and signature(corpus[row])==signature(before)
        if any(0xdc80 <= ord(c) <= 0xdcff for c in before):
            continue
        after=corpus[row]
        chosen.append((row,known&1,flag,group))
        texts['contest'].append(after)
        texts['original'].append(before)
        texts['contest_masked'].append(unannotated(after))
        texts['original_masked'].append(unannotated(before))
        records.append(dict(row=row,original_sha256=hashlib.sha256(before.encode()).hexdigest(),
                            contest_sha256=hashlib.sha256(after.encode()).hexdigest(),
                            same_function_text=before==after))
    rows,w,flag,names=map(np.asarray,zip(*chosen,strict=True))
    outputs,times={},{}
    for label,data in texts.items():
        outputs[label],times[label]=scan(data,True)
        print(label,len(data),times[label],flush=True)
    assert np.array_equal(outputs['contest'][:,0],flag)
    panels={}
    for name,size in population.items():
        mask=names==name
        checked={k:dict(valid=int(((z[:,0]>=0)&mask).sum()),
                        agreement=int(((z[:,0]==w)&mask).sum()),
                        changed_from_contest=int(((z[:,0]!=flag)&mask).sum())) for k,z in outputs.items()}
        panels[name]=dict(population=size,requested=min(size,MAX_GROUP),unique=int(mask.sum()),scores=checked)
    file=DEST/'predictions.npz'
    np.savez_compressed(file,ids=rows,annotation=w,group=names,**outputs)
    receipt=dict(matrix_sha256=stamp,archive_sha256=checksum(SOURCE),inputs=checksum(SOURCE_DATA/'receipt.json'),
                 scanner_sha256=checksum(SCRIPT),predictions_sha256=checksum(file),
                 code_sha256=checksum(Path(__file__)),selection_seed=SEED,
                 grouped=panels,wall_seconds=times,record_hashes=records)
    write_json(DEST/'results.json',receipt)
    print(panels,flush=True)


if __name__=='__main__':
    main()
