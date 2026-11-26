"""Capture numeric first-hit locations without changing the released scanner decision."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'round38'))
from data38 import PARTS, f, checksum, json_at, write_json, phases
from context32 import early_out, raw_values
from first_hit32 import SCRIPT
from points33 import OUT
from experiment33 import read_phase

PROGRAM = HERE/'scan_location.mjs'
OUTPUT = HERE/'locations'
COLUMNS = ('rule_union','first_rule','line_count','first_line','last_line','distinct_lines',
           'min_gap','valid_lines','source_lines')

LOCATION = """
        const valid = lines.filter(line => Number.isSafeInteger(line) && line >= 1 && line <= body.length);
        const unique = [...new Set(valid)].sort((a, b) => a - b);
        let gap = 0;
        for (let k = 1; k < unique.length; k++) {
          const d = unique[k] - unique[k - 1];
          gap = gap ? Math.min(gap, d) : d;
        }
        return [row, 1, j + 1, lines.length, unique[0] || 0, unique[unique.length - 1] || 0,
                unique.length, gap, valid.length, body.length];
"""


def build():
    receipt = json_at(ROOT/'round32/first_hit_public.json')
    if checksum(SCRIPT)!=receipt['program_sha256']:
        raise ValueError('Released scanner changed')
    src = SCRIPT.read_text()
    edits = (
        ('else if (lines.length > 0) return [row, 1, j + 1, lines.length];',
         'else if (lines.length > 0) {'+LOCATION+'      }'),
        ('[row, available ? 0 : -1, 0, 0]','[row, available ? 0 : -1, 0, 0, 0, 0, 0, 0, 0, body.length]'),
        ('[[rows[0][0], -1, 0, 0]]','[[rows[0][0], -1, 0, 0, 0, 0, 0, 0, 0, 0]]'),
        ('["row", "rule_union", "first_rule", "line_count"]',
         json_array(('row',*COLUMNS))),
    )
    for old,new in edits:
        if src.count(old)!=1:
            raise ValueError('Scanner signature changed')
        src=src.replace(old,new)
    PROGRAM.write_text(src)


def json_array(labels):
    import json
    return json.dumps(labels)


def scan(texts):
    names,values=f.scanner(texts,program=PROGRAM)
    if tuple(names)!=COLUMNS:
        raise ValueError('Wrong scanner column order')
    if not np.isin(values[:,0],(-1,0,1)).all():
        raise ValueError('Unknown scanner answer')
    positive=values[:,0]==1
    assert np.array_equal(positive,values[:,1]>0) and np.array_equal(positive,values[:,2]>0)
    assert (values[:,4]>=values[:,3]).all() and (values[:,4]<=values[:,8]).all()
    assert (values[:,5]<=values[:,7]).all() and (values[:,7]<=values[:,2]).all()
    return values


def collection():
    OUTPUT.mkdir(exist_ok=True)
    rosters={}
    signed={}
    for part in PARTS:
        path=OUT/part
        p=path/'receipt.json'
        used=json_at(p)
        signed[part]=checksum(p)
        for role in used['phases']:
            x=read_phase(role,path,used)
            rosters[part,role]=x['ids'][x['old']]
    ids=np.unique(np.concatenate(list(rosters.values())))
    texts=f.inputs()[0]
    rows=[texts[i] for i in ids]
    established=early_out(raw_values(ids))
    start=perf_counter()
    replay=scan(rows)
    elapsed=perf_counter()-start
    old_start=perf_counter()
    names,old=f.scanner(rows,program=SCRIPT)
    old_elapsed=perf_counter()-old_start
    assert tuple(names)==COLUMNS[:3]
    assert np.array_equal(replay[:,:3],old)
    got=replay[:,0]==1
    assert np.array_equal(replay[got,1:3],established[got])
    expected=np.fromiter((s.count('\n')+1 for s in rows),dtype=np.int32)
    assert np.array_equal(expected[replay[:,0]>=0],replay[replay[:,0]>=0,8])
    for (rotation,role), plan in rosters.items():
        index=np.searchsorted(ids,plan)
        assert np.array_equal(ids[index],plan)
        snapshot=read_phase(role,OUT/rotation,json_at(OUT/rotation/'receipt.json'))
        active=snapshot['old']
        assert np.array_equal(replay[index,0],snapshot['flag'][active])
    path=OUTPUT/'training.npz'
    np.savez_compressed(path,ids=ids,values=replay)
    write_json(OUTPUT/'receipt.json',dict(rows=len(ids),positive=int(got.sum()),location=int((replay[:,3]>0).sum()),
                  positive_without_valid_line=int((got&(replay[:,3]==0)).sum()),
                  original_scanner_sha256=checksum(SCRIPT),extended_scanner_sha256=checksum(PROGRAM),
                  predictions_sha256=checksum(path),points=signed,columns=COLUMNS,seconds=elapsed,
                  original_seconds=old_elapsed,code_sha256=checksum(Path(__file__))))
    print(len(ids),int(got.sum()),int((replay[:,3]>0).sum()),elapsed,old_elapsed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('build','collect'))
    if parser.parse_args().stage=='build':
        build()
    else:
        collection()
