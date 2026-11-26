"""Train a comment-provenance ranker from positives and controls in the same fit files."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression

from audit_fit import FAMILY
from placement import build_observations, cutoff, table
from transfer import HERE, ROOT, checksum, dataset, split, write_json

DEST = HERE/'paired_provenance'
MARKER = 'marker_in_adjacent_preceding_comment'
CLEAN = 'match_without_adjacent_marker'


def paired_examples(records, files, texts, possible):
    by_file = defaultdict(list)
    for row, info in records.items():
        if not info['gold'] and info['result']==CLEAN:
            by_file[files[row]].append(row)
    wins, losses = [], []
    for row in possible:
        controls = by_file[files[row]]
        if not controls:
            continue
        closest = sorted(controls,key=lambda i:(abs(len(texts[row])-len(texts[i])),i))[:8]
        wins.extend([row]*len(closest))
        losses.extend(closest)
    return np.asarray(wins,dtype=np.int64),np.asarray(losses,dtype=np.int64)


def learn(x, wins, losses, ids):
    difference = x[np.searchsorted(ids,wins)]-x[np.searchsorted(ids,losses)]
    pair = vstack((difference,-difference),format='csr')
    labels = np.r_[np.ones(len(wins),dtype=bool),np.zeros(len(wins),dtype=bool)]
    counts = np.unique(wins,return_inverse=True,return_counts=True)
    weight = 1./counts[2][counts[1]]
    linear = LogisticRegression(C=2.,solver='liblinear',fit_intercept=False,random_state=42)
    return linear.fit(pair,labels,sample_weight=np.tile(weight,2))


def train(path):
    DEST.mkdir(exist_ok=True)
    texts, labels, projects, files = dataset()
    gold = labels[:,1].astype(bool)
    split_ids = split()
    audit = json.loads(path.read_text())
    if audit['project']!=FAMILY or audit['fit_ids_sha256']!=hashlib.sha256(split_ids['fit'].tobytes()).hexdigest():
        raise ValueError('Training source audit does not match fit rows')
    provenance = {row['row']:row for row in audit['observations']}
    if not np.isin(list(provenance),split_ids['fit']).all():
        raise ValueError('Source provenance includes non-fit records')
    if any(row['gold']!=gold[i] for i,row in provenance.items()):
        raise ValueError('Source provenance labels differ from the signed dataset')
    matched = [i for i, item in provenance.items() if item['gold'] and item['matched']]
    censored = [i for i in matched if provenance[i]['result']==MARKER]
    options = {'all_located':matched,'preceding':censored}
    ids = np.array(sorted(provenance))
    x, _, _, _ = build_observations(texts,ids)
    heads, fit = {}, {}
    for name,cases in options.items():
        win,loss = paired_examples(provenance,files,texts,cases)
        if not len(win):
            continue
        heads[name] = learn(x,win,loss,ids)
        fit[name] = dict(cases=len(np.unique(win)),controls=len(np.unique(loss)),pairs=len(win))
        print(name,fit[name],flush=True)
    if not heads:
        raise ValueError('The fit slice provides no valid provenance pairs')
    weights=DEST/'weights.joblib'
    joblib.dump(heads,weights,compress=3)
    reports = {}
    frozen = {}
    old_residual = HERE/'placement/heads.joblib'
    residual = joblib.load(old_residual)['residual']
    for part in ('dev','test_a','test_b'):
        rows = split_ids[part]
        features, _, comments, base = build_observations(texts,rows)
        ref = residual.decision_function(features)
        rank = {'residual':ref}
        for name, classifier in heads.items():
            rank[name] = classifier.decision_function(features)
            rank[name+'_blend'] = ref + rank[name]
        if part=='dev':
            frozen = {key:cutoff(gold[rows],base,val) for key,val in rank.items()}
            write_json(DEST/'policy.json',dict(training=fit,cutoffs=frozen,audit_sha256=checksum(path),
                       code_sha256=checksum(Path(__file__)),partition_sha256=audit['partition_sha256'],
                       weights_sha256=checksum(weights),residual_sha256=checksum(old_residual)))
        # Reuse the exact FP accounting; untagged evaluation inputs are from the parquet only.
        reports[part] = table(part,rows,gold[rows],projects[rows],rank,comments,base,frozen,dest=DEST)
        print(part,{k:[v['frozen']['recovered'],v['frozen']['delta_fp'],v['matched']['recovered']]
                    for k,v in reports[part]['results'].items()},flush=True)
    write_json(DEST/'results.json',reports)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    train(parser.parse_args().audit)
