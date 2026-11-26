"""Classify the provenance of natural MAT misses inside the project fit split."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys


from transfer import HERE, ROOT, canonical_mat, checksum, dataset, split, write_json

R18 = ROOT/'round18/mat_residual'
sys.path.insert(0,str(R18))
from source_audit import SOURCES, fetch, locate, placement, stream

FAMILY = 'linux'
CONTROLS = 8


def audit(job):
    path, cohort, texts, gold = job
    source, receipt = fetch(FAMILY,path)
    scopes = {'match_without_adjacent_marker','marker_inside_matched_span','marker_in_adjacent_preceding_comment'}
    if source is not None:
        tokens, comments = stream(source)
        sequence = tuple(row[0] for row in tokens)
        starts = defaultdict(list)
        for i,token in enumerate(sequence):
            starts[token].append(i)
    records = []
    for row in cohort:
        query, visible = stream(texts[row])
        outcome = 'source_unavailable'
        if source is not None:
            matches = locate(tokens,sequence,starts,query)
            outcome = placement(source,comments,matches)
        records.append(dict(row=int(row),gold=bool(gold[row]),visible_comments=len(visible),
                            matched=outcome in scopes,result=outcome))
    return dict(path=path,**receipt),records


def select_rows(ids,texts,y,files,base):
    by_file = defaultdict(list)
    for row, pred in zip(ids,base,strict=True):
        if not pred:
            by_file[files[row]].append(row)
    for path,rows in sorted(by_file.items()):
        pos = [row for row in rows if y[row]]
        if not pos:
            continue
        neg = [row for row in rows if not y[row]]
        control = set()
        for row in pos:
            # Use a deterministic proximity match by code length from the same training file.
            key = lambda i:(abs(len(texts[i])-len(texts[row])),i)
            control.update(sorted(neg,key=key)[:CONTROLS])
        yield path,sorted(pos+list(control))


def main():
    texts,labels,project,files = dataset()
    ids = split()['fit']
    if set(project[ids]) != {FAMILY}:
        raise ValueError('Unexpected project in the MAT fit role')
    text = [texts[i] for i in ids]
    old = canonical_mat(text)
    y = labels[:,1].astype(bool)
    eligible = list(select_rows(ids,texts,y,files,old))
    summary = defaultdict(Counter)
    sources, details = [], []
    print('fit',len(ids),'misses',int((y[ids] & ~old).sum()),'files',len(eligible),flush=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        jobs = ((p,r,texts,y) for p,r in eligible)
        for receipt,records in pool.map(audit,jobs):
            sources.append(receipt)
            details.extend(records)
            for row in records:
                key = 'miss' if row['gold'] else 'same_file_control'
                summary[key][row['result']] += 1
                placement_key = key+'/visible' if row['visible_comments'] else key+'/without_comments'
                summary[placement_key][row['result']] += 1
    assert len({i['row'] for i in details})==len(details)
    assert sum(c['gold'] for c in details)==int((y[ids] & ~old).sum())
    print(dict(summary),flush=True)
    write_json(HERE/'fit_provenance.json',dict(project=FAMILY,files=sources,summary=summary,observations=details,
               fit_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),controls_per_miss=CONTROLS,
               partition_sha256=checksum(ROOT/'round13/partition.npz'),
               source_audit_code_sha256=checksum(R18/'source_audit.py'),
               data_manifest_sha256=checksum(ROOT/'data/manifest.json'),code_sha256=checksum(Path(__file__)),
               pinned_sources=SOURCES[FAMILY]))


if __name__ == '__main__':
    main()
