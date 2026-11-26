"""Measure frozen low-FP MAT trade-offs around the zero-FP operating point."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUDGETS = (0,1,2,4,8,16)
PARTS = ('dev','test_a','test_b')


def thresholds(scores, y, base):
    eligible = ~base
    score, truth = scores[eligible], y[eligible]
    order = np.argsort(-score,kind='stable')
    score, truth = score[order], truth[order]
    at = np.flatnonzero(np.r_[score[:-1]!=score[1:], True])
    tp = np.r_[0,np.cumsum(truth)[at]]
    fp = np.r_[0,np.cumsum(~truth)[at]]
    cut = np.r_[score.max(),np.nextafter(score[at],-np.inf)]
    positives, found = y.sum(), (y & base).sum()
    base_fp = (base & ~y).sum()
    f1 = 2*(found+tp)/(positives+found+tp+base_fp+fp)
    out = {}
    for budget in BUDGETS:
        feasible = np.flatnonzero(fp<=budget)
        chosen = feasible[np.argmax(f1[feasible])]
        out[budget] = float(cut[chosen])
    return out


def metrics(data, arm, cutoff):
    y, base = data['y'], data['base']
    score = data[arm+'_score']
    new = ~base & (score>cutoff)
    tp, fp = int((new & y).sum()),int((new & ~y).sum())
    prev_tp, prev_fp = int((base & y).sum()),int((base & ~y).sum())
    count = int(y.sum())
    before = 2*prev_tp/(count+prev_tp+prev_fp)
    after = 2*(prev_tp+tp)/(count+prev_tp+tp+prev_fp+fp)
    return dict(delta_tp=tp,delta_fp=fp,delta_f1=after-before)


def main():
    rows = []
    for family in ('placement','paired_provenance'):
        basedir = HERE/family
        data = {}
        for part in PARTS:
            with np.load(basedir/f'{part}_predictions.npz') as loaded:
                data[part] = dict(loaded)
        dev = data['dev']
        arms = [k.removesuffix('_score') for k in dev if k.endswith('_score')]
        for arm, budget in itertools.product(arms,BUDGETS):
            cutoff = thresholds(dev[arm+'_score'],dev['y'],dev['base'])[budget]
            cases = {part:metrics(data[part],arm,cutoff) for part in PARTS}
            rows.append(dict(family=family,arm=arm,budget=budget,cutoff=cutoff,
                             **{part+'_'+key:value for part, record in cases.items() for key,value in record.items()}))
    out = HERE/'frontier.csv'
    with out.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    promising = [row for row in rows if min(row[part+'_delta_f1'] for part in PARTS)>0]
    print(json.dumps(dict(configurations=len(rows),positive_on_all_three=promising,
                           best_by_development=max(rows,key=lambda r:(r['dev_delta_f1'],-r['budget']))),
                     indent=2))


if __name__=='__main__':
    main()
