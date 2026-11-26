"""Separate width, indentation, and line-position effects on archived W."""
from collections import Counter
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from causal54 import (DEST as CAUSAL, HERE, PARTS, aligned, changes, checksum,
                      f, json_at, render, scan, stored_pairs, write_json)

DEST = HERE / 'spaces'
TYPES = ('horizontal', 'indent', 'horizontal_indent', 'exact_break', 'strict_break',
         'horizontal1', 'horizontal_tab', 'indent4', 'indent_tab', 'horizontal1_indent4')


def variants(text, paired):
    full, token, safe, line, _ = paired
    bucket = {k: {} for k in TYPES}
    opening = 0
    depths = []
    for s in token:
        opening += int(s[0] == '{') - int(s[0] == '}')
        depths.append(max(0, opening))
    for i, (current, source) in safe.items():
        a, b = '\n' in current, '\n' in source
        if a != b:
            bucket['exact_break'][i] = source
            bucket['strict_break'][i] = line[i]
        if not a:
            bucket['horizontal1'][i] = ' '
            bucket['horizontal_tab'][i] = '\t'
        else:
            old = current[:current.rfind('\n')+1]
            depth = max(0, depths[i] - int(token[i+1][0] == '}'))
            bucket['indent4'][i] = old + ' ' * (4 * depth)
            bucket['indent_tab'][i] = old + '\t' * depth
        if a == b and current != source:
            bucket['indent' if b else 'horizontal'][i] = source
    bucket['horizontal_indent'] = bucket['horizontal'] | bucket['indent']
    bucket['horizontal1_indent4'] = bucket['horizontal1'] | bucket['indent4']
    return {name: render(text, full, token, {i:d for i,d in edits.items() if d != safe[i][0]})
            for name, edits in bucket.items()}, bucket, safe


def main():
    DEST.mkdir(exist_ok=True)
    pair, plan = stored_pairs()
    origin = CAUSAL / 'oracle.npz'
    assert checksum(origin) == json_at(CAUSAL / 'oracle.json')['oracle_sha256']
    with np.load(origin) as z:
        roster, baseline, gold = (z[k] for k in ('ids','baseline','gold'))
    # Measure diagnostic oracles on the entire held/calibration set. Fixed recipes
    # are applied only wherever a strictly aligned pair allows this comparison.
    phases = {k: set(v) for k,v in plan['cohorts'].items() if not k.endswith('/fit')}
    active = np.isin(roster, list(set.union(*phases.values())))
    ids, baseline, gold = roster[active], baseline[active], gold[active]
    text = f.inputs()[0]
    cards = {name: {} for name in TYPES}
    changed = Counter()
    pattern = Counter()
    ready = perf_counter()
    for j, i in enumerate(ids):
        raw = text[i]
        view = aligned(raw, pair[i]['original'])
        assert view is not None
        alternatives, edits, safe = variants(raw, view)
        for mode, q in alternatives.items():
            if q != raw:
                cards[mode][j] = q
                changed[mode] += len(edits[mode])
        if baseline[j, 0] != gold[j]:
            for loc, (a,b) in safe.items():
                if a != b:
                    typ = ''.join('T' if '\t' in w else 'L' if '\n' in w else 'S' for w in (a,b))
                    pattern[typ] += 1
    built = perf_counter()-ready
    output = dict(ids=ids, baseline=baseline, gold=gold)
    runs = {}
    for mode, items in cards.items():
        result = baseline.copy()
        indices = np.fromiter(items, np.int64)
        result[indices], seconds = scan(list(items.values()), True)
        output[mode] = result
        runs[mode] = dict(rows=len(items), changes=changed[mode], scanner_seconds=seconds)
        print(mode, runs[mode], flush=True)
    report = {role: changes(np.flatnonzero(np.isin(ids, list(rows))), output, TYPES)
              for role, rows in phases.items()}
    file = DEST / 'predictions.npz'
    np.savez_compressed(file, **output)
    target = file.with_name('comparison.csv')
    with target.open('w',newline='') as handle:
        out = csv.writer(handle)
        out.writerow(('rotation','role','variant','rows','repaired','hurt','w_tp','w_fp','w_f1'))
        for name, variants_ in report.items():
            for k, y in variants_.items():
                m = y['raw_w']
                out.writerow((*name.split('/'),k,y['rows'],y['repaired'],y['hurt'],m['delta_tp'],m['delta_fp'],m['f1_after']))
    print({x:{k:(p['repaired'],p['hurt']) for k,p in r.items()} for x,r in report.items()},flush=True)
    write_json(DEST/'results.json',dict(predictions_sha256=checksum(file), code_sha256=checksum(Path(__file__)),
               oracle_sha256=checksum(origin), comparison_sha256=checksum(target), phases=report,
               recipes=runs, patterns=pattern, preparation_seconds=built))


if __name__ == '__main__':
    main()
