"""Verify held-out MAT predictions and count the FP cost of one rank-based recovery."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def by_project(values, truth, base, project):
    counts = {}
    for p in np.unique(project):
        use = (project==p) & ~base
        pos, neg = values[use & truth], values[use & ~truth]
        if not len(pos):
            counts[p] = None
        else:
            counts[p] = int((neg>=pos.max()).sum())
    return counts


def validate(split):
    baseline = ROOT/'round26/mat'/f'{split}_predictions.npz'
    checked = {}
    with np.load(baseline) as original:
        base, truth, ids, project = (original[k] for k in ('canonical','y','ids','project'))
    for area in ('placement','paired_provenance'):
        folder = HERE/area
        if not folder.is_dir():
            continue
        output = {}
        with np.load(folder/f'{split}_predictions.npz') as row:
            for reference, actual in ((base,row['base']),(truth,row['y']),(ids,row['ids']),(project,row['group'])):
                if not np.array_equal(reference,actual):
                    raise ValueError(f'Canonical replay differs for {area} and {split}')
            for key in row.files:
                if not key.endswith('_score'):
                    continue
                arm = key.removesuffix('_score')
                counts = by_project(row[key],truth,base,project)
                positive = truth & ~base
                result = [int((row[arm+'_'+mode] & positive).sum()) for mode in ('frozen','matched')]
                if np.count_nonzero(row[arm+'_matched'] & ~truth) != np.count_nonzero(base & ~truth):
                    raise ValueError('Unmatched project false-positive budget')
                output[arm] = dict(minimum_additional_fp_per_project=counts,
                                   minimum_over_any_project=min(v for v in counts.values() if v is not None),
                                   frozen_recovered=result[0],matched_recovered=result[1])
        checked[area] = output
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'margins.json')
    args = p.parse_args()
    result = {split:validate(split) for split in ('test_a','test_b')}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
