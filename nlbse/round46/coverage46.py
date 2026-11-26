"""Measure complementarity in comparable source-length cohorts at the old FP count."""
from pathlib import Path

import numpy as np

from fit46 import HERE,ROOT,PARTS,DEST as FIT,f,checksum,json_at,load,parts,write_json,negative
from views46 import DEST as WINDOWS
from stack40 import diagnostic_order,per_project

BANK = ('unix96','unix384','qwen96','qwen384','qwen12','transfer_qwen96','transfer_qwen384','compact')


def measure(rotation):
    matrices,_,stamp = load(rotation)
    _,frame = parts(matrices)
    receipt = json_at(FIT/rotation/'receipt.json')
    file = FIT/rotation/'scores.npz'
    assert receipt['predictions_sha256']==checksum(file) and receipt['matrix_sha256']==stamp
    plan = json_at(WINDOWS/'receipt.json')
    assert checksum(WINDOWS/'rows.npz')==plan['roster_sha256']
    roster = np.load(WINDOWS/'rows.npz')['ids']
    eligible = negative(frame)
    rows = frame['ids'][eligible]
    positions = np.searchsorted(roster,rows)
    assert np.array_equal(roster[positions],rows)
    short,full = [np.load(WINDOWS/f'qwen_{limit}.npz')['full'][positions] for limit in (96,384)]
    assert not (short&~full).any()
    packs = {'all':np.ones(len(rows),bool),'complete_96':short,
             'extra_at_384':~short&full,'cut_at_384':~full}
    baseline = frame['signed']>0.
    y = frame['y']
    results = {}
    with np.load(file) as pred:
        assert np.array_equal(pred['held_ids'],rows)
        for name,z in packs.items():
            where = eligible.copy()
            where[eligible] &= z
            q = frame['signed'][where]
            if not len(q):
                continue
            decisions,op = {},{}
            for arm in BANK:
                estimate = .3*q+.7*pred['held_'+arm][z]
                rank,_ = diagnostic_order(estimate,q,frame['ids'][where])
                note,opts = per_project(y,frame['project'],baseline,where,rank,('fp',))
                decisions[arm] = opts['fp']
                op[arm] = note['fp']['change']
            footprints = {}
            for high,low in (('qwen96','unix96'),('qwen384','unix384'),('qwen12','unix96'),
                             ('qwen384','qwen96'),('transfer_qwen384','compact')):
                a,b = decisions[high],decisions[low]
                focus = y&~baseline
                footprints[high+'_vs_'+low] = dict(first_only=int((focus&a&~b).sum()),
                                                   second_only=int((focus&~a&b).sum()),
                                                   both=int((focus&a&b).sum()))
            results[name] = dict(rows=int(where.sum()),baseline_fp=int((~y&baseline&where).sum()),
                     baseline_fn=int((y&~baseline&where).sum()),matched=op,footprints=footprints)
    print(rotation,{k:{a:x['delta_tp'] for a,x in d['matched'].items()}
                         for k,d in results.items()},flush=True)
    return dict(results=results,predictions_sha256=checksum(file),windows_sha256=checksum(WINDOWS/'receipt.json'))


def main():
    result = {r:measure(r) for r in PARTS}
    write_json(HERE/'coverage.json',dict(rotations=result,mix=.7,mode='diagnostic_at_known_project_fp',
               code_sha256=checksum(Path(__file__))))


if __name__=='__main__':
    main()
