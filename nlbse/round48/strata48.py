"""Calibrate MAT recovery separately by observable comment coverage and function length."""
import csv
from pathlib import Path

import numpy as np

from audit48 import count, load_shape, provenance
from shape48 import DEST, HERE, PARTS, checksum, write_json

OUT = HERE/'strata'
MIN_NEGATIVES = 500
LENGTH = 200
SCOPES = ('comment', 'comment_and_length')
ARMS = ('layout_paired', 'joint_paired', 'stub_paired', 'previous_joint')


def group(texts, source, scope):
    comment = source['comment'].astype(np.int8)
    if scope == 'comment':
        return comment
    short = np.fromiter((len(texts[i]) <= LENGTH for i in source['ids']), bool)
    return 2*comment + short


def conditional_cut(score, groups, neg, levels):
    general = float(score[neg].max())
    return [float(score[neg & (groups == i)].max())
            if (neg & (groups == i)).sum() >= MIN_NEGATIVES else general for i in range(levels)]


def run():
    OUT.mkdir(exist_ok=True)
    (texts, _, _, _), data, verified = load_shape()
    # The only holdout data eligible to enter the policy is the supplied text.
    # Projection of ids into a char length does not use project metadata or labels.
    sources,_ = provenance(data)
    policy = dict(arms=list(ARMS),scopes=list(SCOPES),maximum_short_chars=LENGTH,
                  minimum_development_negatives=MIN_NEGATIVES,cutoffs={},
                  predictions_input={k:verified[k]['predictions_sha256'] for k in PARTS},
                  initial_policy_sha256=checksum(DEST/'policy.json'),code_sha256=checksum(Path(__file__)))
    dev = data['dev']
    neg = ~dev['y'] & ~dev['base']
    for scope in SCOPES:
        g = group(texts,dev,scope)
        levels = 2 if scope == 'comment' else 4
        policy['cutoffs'][scope] = {arm:conditional_cut(dev[arm+'_score'],g,neg,levels) for arm in ARMS}
    write_json(OUT/'policy.json',policy)

    records = []
    evidence = {}
    for part,d in data.items():
        base, y = d['base'],d['y']
        project = d['group']
        binary = {'ids':d['ids'],'base':base,'gold':y,'project':project}
        evidence[part] = {}
        for scope,arms in policy['cutoffs'].items():
            g = group(texts,d,scope)
            for arm,cuts in arms.items():
                scores = d[arm+'_score'] - np.take(cuts,g)
                frozen = base | (scores>0)
                exact = base.copy()
                for p in np.unique(project):
                    panel = p==project
                    threshold = scores[panel & ~y & ~base].max()
                    exact[panel] |= scores[panel] > threshold
                    assert count(y[panel],base[panel],exact[panel])['fp'] == int((base[panel] & ~y[panel]).sum())
                for variant,forecast in [('frozen',frozen),('matched',exact)]:
                    key = f'{scope}_{arm}_{variant}'
                    binary[key] = forecast
                    row = count(y,base,forecast)
                    records.append(dict(part=part,scope=scope,arm=arm,variant=variant,**row))
                    ids = d['ids'][y & ~base & forecast]
                    evidence[part][key] = dict(recovered_rows=ids.tolist(),
                                              recovered_with_known_omission=np.intersect1d(
                                                  ids,sources.get(part,{}).get('rows',[])).tolist())
        p = OUT/f'{part}_predictions.npz'
        np.savez_compressed(p,**binary)
        evidence[part]['predictions_sha256'] = checksum(p)
    path = OUT/'comparison.csv'
    with path.open('w',newline='') as out:
        table = csv.DictWriter(out,fieldnames=list(records[0]))
        table.writeheader()
        table.writerows(records)
    write_json(OUT/'results.json',dict(protocol='diagnostic follow-up; strata chosen after first heldout analysis',
               summary=records,evidence=evidence,csv_sha256=checksum(path),
               policy_sha256=checksum(OUT/'policy.json')))
    for k in PARTS:
        print(k,[(r['scope'],r['arm'],r['variant'],r['recovered'],r['fp'])
                  for r in records if r['part']==k and (r['recovered'] or r['fp'])],flush=True)


if __name__=='__main__':
    run()
