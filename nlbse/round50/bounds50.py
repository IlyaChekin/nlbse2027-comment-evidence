"""Measure how much published error falls in the archived W disagreement cohort."""
from collections import defaultdict
import csv
import io
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from study50 import HERE, ROOT, f, checksum, json_at, write_json
from components49 import SOURCE, COLS, COLLECTIONS, key, signature, transform
from pipeline import PUBLIC, load
from live40 import anchor

DEST=HERE/'public_audit'
ENDPOINTS=ROOT/'round49/public_ceiling/endpoints.npz'


def archive_lookup(texts,meta):
    groups=defaultdict(dict)
    for i,(text,p,path) in enumerate(zip(texts,meta['Projectname'],meta['Filepath'],strict=True)):
        location=key(p,path)
        if all(location):
            groups[location].setdefault(signature(text),[]).append(i)
    evidence=np.zeros(len(texts),np.uint16)
    csv.field_size_limit(32*1024*1024)
    with ZipFile(SOURCE) as archive:
        for name in COLLECTIONS:
            path=f'MADE-WIC/Dataset/{name}/complete.csv'
            with io.TextIOWrapper(archive.open(path),encoding='utf-8-sig',errors='surrogateescape',newline='') as raw:
                table=csv.DictReader(raw)
                fields=tuple(k for k in COLS if k in table.fieldnames)
                for row in table:
                    site=key(row['Projectname'],row['Filepath'])
                    cache=groups.get(site)
                    if not cache:
                        continue
                    found=cache.get(signature(row['Function']))
                    if found:
                        assert all(row[k] in ('0','1') for k in fields)
                        evidence[found] |= 1<<transform(row,fields)
    nonempty=evidence!=0
    trusted=nonempty & ((evidence & (evidence-1))==0)
    component=np.full(len(texts),-1,np.int8)
    for code in range(8):
        component[trusted & (evidence==1<<code)]=code
    return evidence,component,trusted


def main():
    DEST.mkdir(exist_ok=True)
    started=perf_counter()
    old=ROOT/'round49/components/receipt.json'
    assert checksum(SOURCE)==json_at(old)['archive_sha256']
    archive, signed,y_ref,baseline=anchor()
    texts,truth,meta=load(PUBLIC,audit=True)
    assert np.array_equal(truth,y_ref)
    endpoint=ENDPOINTS.with_name('results.json')
    header=json_at(endpoint)
    assert header['predictions_sha256']==checksum(ENDPOINTS)
    assert header['archive_sha256']==checksum(archive)
    assert header['release_predictions_sha256']==checksum(signed) and header['dataset_sha256']==checksum(PUBLIC)
    with np.load(ENDPOINTS) as z:
        assert np.array_equal(z['truth'],truth) and np.array_equal(z['signed']>0.,baseline[:,0])
        scanner,known=z['flag'][:,0],z['old']
    evidence,component,trusted=archive_lookup(texts,meta)
    y=truth[:,0].astype(bool)
    consistent=trusted & ((component>0)==y)
    w=(component&1)>0
    external=(component&6)>0
    disagree=consistent & (scanner>=0)&((scanner==1)!=w)
    ref=baseline[:,0]
    settings={
        'archived_W_disagrees_observed':disagree & known,
        'archived_W_disagrees_unqueried':disagree & ~known,
        'archived_W_disagrees_any':disagree,
        'observed_W_concordant':consistent & known & ~disagree,
        'archived_external_without_W':consistent & external & ~w,
        'archived_W_only':consistent & w & ~external,
        'archived_both':consistent & w & external,
        'unknown_annotation':~consistent,
        'all_observed':known,
    }
    # Every result is diagnostic, with labels used only to establish an upper bound.
    release=json_at(ROOT/'round38/release_summary.json')
    values=release['cost']
    mat=release['quality']['MAT']['f1']
    report={}
    for k,s in settings.items():
        correct=np.where(s,y,ref)
        delta=f.comparison(y,ref,correct)
        p=float(.5*(mat+delta['f1_after']))
        report[k]=dict(rows=int(s.sum()),f1=delta['f1_after'],additional_tp=delta['delta_tp'],
                       removed_fp=-delta['delta_fp'],
                       unchanged_cost_score=f.competition_formula(p,values['forward_seconds'],values['analytic_gflops']),
                       zero_cost_score=f.competition_formula(p,0.,0.))
    file=DEST/'attributions.npz'
    np.savez_compressed(file,component=component,evidence=evidence,consistent=consistent,gold=truth,
                        scanner=scanner,observed=known,before=ref,disagree=disagree)
    receipt=dict(rows=len(y),matched=int(consistent.sum()),conflicts=int((trusted&~consistent).sum()),
                 ambiguous=int(((evidence>0)&~trusted).sum()),
                 scenarios=report,source_sha256=checksum(SOURCE),dataset_sha256=checksum(PUBLIC),
                 release_predictions_sha256=checksum(signed),endpoints_sha256=checksum(ENDPOINTS),
                 archive_sha256=checksum(archive),predictions_sha256=checksum(file),
                 code_sha256=checksum(Path(__file__)),runtime_seconds=perf_counter()-started,
                 access='diagnostic only; public metadata and source labels do not enter any fitted model')
    write_json(DEST/'results.json',receipt)
    print(int(consistent.sum()),receipt['runtime_seconds'],report,flush=True)


if __name__=='__main__':
    main()
