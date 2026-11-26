"""Expose the first successful rule from the existing short-circuit scanner."""
import argparse
from statistics import median
import time

import numpy as np

from context32 import HERE, ROOT, f, checksum, json_at, write_json, PROGRAM, CONFIG
from pipeline import PUBLIC, load

SCRIPT = HERE/'scan_first.mjs'
COLUMNS = ('rule_union','first_rule','line_count')


def build():
    cfg = json_at(CONFIG)
    if checksum(PROGRAM)!=cfg['program_sha256']:
        raise ValueError('The existing scanner source changed')
    src = PROGRAM.read_text()
    replacements = {
        '  for (const check of rules) {':'  for (const [j, check] of rules.entries()) {',
        'else if (lines.length > 0) return [row, 1];':'else if (lines.length > 0) return [row, 1, j + 1, lines.length];',
        'return [row, available ? 0 : -1];':'return [row, available ? 0 : -1, 0, 0];',
        'return [[rows[0][0], -1]];':'return [[rows[0][0], -1, 0, 0]];',
        '["row", "rule_union"]':'["row", "rule_union", "first_rule", "line_count"]',
    }
    for old,new in replacements.items():
        if src.count(old)!=1:
            raise ValueError('The scanner source stopped matching the versioned change')
        src = src.replace(old,new)
    SCRIPT.write_text(src)


def scan(texts, first):
    started = time.perf_counter()
    names, values = f.scanner(texts,program=SCRIPT if first else PROGRAM)
    expected = COLUMNS if first else COLUMNS[:1]
    if tuple(names)!=expected or not np.isin(values[:,0],(-1,0,1)).all():
        raise ValueError('Unexpected scanner result')
    if first:
        if not np.array_equal(values[:,1]>0,values[:,0]==1) or not np.array_equal(values[:,2]>0,values[:,0]==1):
            raise ValueError('First rule and counts disagree with the union')
    return values,time.perf_counter()-started


def verify():
    scanner = ROOT/'round22/union_public'
    path = scanner/'scanner_predictions.npz'
    src = json_at(scanner/'results.json')
    assert checksum(path)==src['scanner_predictions_sha256']
    texts,truth,_ = load(PUBLIC)
    audit = ROOT/'round31/reference_audit.npz'
    manifest = json_at(ROOT/'round31/reference_audit.json')
    assert checksum(audit)==manifest['readout_sha256'] and checksum(PUBLIC)==manifest['public_sha256']
    with np.load(path) as z, np.load(audit) as prior:
        asked, expected = z['gate'], z['flag']
        assert np.array_equal(prior['asked'],asked)
        archived = prior['bits'][asked,0]
    selected = [s for s,enabled in zip(texts,asked,strict=True) if enabled]
    scans = [[],[]]
    for turn in range(5):
        for toggle in ((False,True) if turn%2==0 else (True,False)):
            raw, seconds = scan(selected,toggle)
            assert np.array_equal(raw[:,0],expected)
            scans[toggle].append(seconds)
    extended,_ = scan(selected,True)
    cols = json_at(CONFIG)['order']
    known = truth[asked,0].astype(bool)
    counts = {}
    for i,label in enumerate(cols,1):
        got = extended[:,1]==i
        counts[label] = dict(rows=int(got.sum()),tp=int((got & archived & known).sum()),
                             fp=int((got & archived & ~known).sum()),fn=int((got & ~archived & known).sum()))
    assert sum(rule['fp'] for rule in counts.values())==274
    prediction = HERE/'first_hit_public.npz'
    np.savez_compressed(prediction,ids=np.flatnonzero(asked),results=extended)
    receipt = dict(rows=len(selected),same_ternary=len(expected),labels_used='error description only',
                   order=cols,by_first_rule=counts,incremental_cost_seconds=median(b-a for a,b in zip(*scans)),
                   scan_only_seconds={'reference':scans[0],'first_hit':scans[1]},
                   predictions_sha256=checksum(prediction),source_scanner_sha256=checksum(PROGRAM),
                   program_sha256=checksum(SCRIPT),code_sha256=checksum(HERE/'first_hit32.py'))
    write_json(HERE/'first_hit_public.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('build','verify'))
    {'build':build,'verify':verify}[parser.parse_args().stage]()
