"""Predict a missing MAT marker using supervision matched to comment placement."""
from collections import Counter
import csv
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from transfer import HERE, ROOT, checksum, dataset, split, write_json, canonical_mat, TOKEN, upstream_tag, record

DEST = HERE/'placement'
SYNTHETIC_WEIGHT = .30
MATRIX = HashingVectorizer(n_features=2**19, analyzer='word', tokenizer=str.split,
                           lowercase=False, token_pattern=None, ngram_range=(1,3), dtype=np.float32)


def observe(text):
    """Remove tagged spans while keeping observable comment positions and other code."""
    terms = []
    ncode = 0
    body = False
    has_comment = False
    placement = []
    for m in TOKEN.finditer(text):
        token = m[0]
        if m['comment'] is not None:
            if upstream_tag(token):
                placement.append('inside' if body else ('leading' if ncode == 0 else 'signature'))
                continue
            has_comment = True
            token = 'mat_visible_inside' if body else 'mat_visible_before_body'
        else:
            ncode += 1
            body |= token == '{'
        terms.append(token)
    return ' '.join(terms), placement, has_comment


def build_observations(texts, ids):
    start = time.perf_counter()
    obs = [observe(texts[i]) for i in ids]
    x, locations, comments = zip(*obs,strict=True)
    raw = [texts[i] for i in ids]
    base = canonical_mat(raw)
    print('observations',len(ids),round(time.perf_counter()-start,2),flush=True)
    return MATRIX.transform(x), locations, np.asarray(comments), base


def cutoff(y, base, score):
    negatives = score[~y & ~base]
    if not len(negatives):
        raise ValueError('Cannot estimate a cutoff without negative functions')
    return float(negatives.max())


def table(which, ids, y, group, scores, comments, base, frozen, dest=DEST):
    predictions = {}
    result = {}
    for name, value in scores.items():
        fixed = base | (value>frozen[name])
        match = base.copy()
        by_project = {}
        for p in np.unique(group):
            local = group==p
            t = cutoff(y[local],base[local],value[local])
            match[local] |= value[local]>t
            item = record(y[local],match[local],base[local],comments[local])
            if item['delta_fp'] != 0:
                raise ValueError('False positive budget changed')
            by_project[p] = item | {'cut':t}
        result[name] = dict(frozen=record(y,fixed,base,comments),matched=record(y,match,base,comments),
                            average_precision=float(average_precision_score(y[~base],value[~base])),
                            projects=by_project)
        predictions.update({name+'_score':value,name+'_frozen':fixed,name+'_matched':match})
    path = dest/f'{which}_predictions.npz'
    np.savez_compressed(path,ids=ids,y=y,base=base,group=group,comment=comments,**predictions)
    return dict(baseline=record(y,base,base,comments),results=result,predictions_sha256=checksum(path))


def main():
    DEST.mkdir(exist_ok=True)
    texts, labels, projects, _ = dataset()
    y = labels[:,1].astype(bool)
    rows = split()
    ids = rows['fit']
    x,locations,_,base = build_observations(texts,ids)
    has_tag = np.asarray([bool(row) for row in locations])
    leading = np.asarray([bool(row) and all(x=='leading' for x in row) for row in locations])
    # On natural negatives, preserve annotation semantics. Synthetics must be positives
    # with a genuine removed span and a preceding-only tag for erase_leading.
    natural = ~base & ~has_tag
    allowed = {'residual':np.zeros(len(ids),bool),'erase_any':base & has_tag & y[ids],
               'erase_leading':base & leading & y[ids]}
    models = {}
    stats = {}
    for name, augmented in allowed.items():
        use = natural | augmented
        weight = np.where(y[ids][use],8.,1.) * np.where(augmented[use],SYNTHETIC_WEIGHT,1.)
        model = LogisticRegression(solver='liblinear',C=2.,max_iter=200,random_state=42)
        model.fit(x[use],y[ids][use],sample_weight=weight)
        models[name] = model
        stats[name] = dict(synthetic=int(augmented.sum()),natural_positives=int((natural & y[ids]).sum()),
                           natural_negatives=int((natural & ~y[ids]).sum()))
        print(name,stats[name],flush=True)
    path = DEST/'heads.joblib'
    joblib.dump(models,path,compress=3)
    body_counts = Counter(tag for i,line in enumerate(locations) if y[ids[i]] for tag in line)
    output = {}
    frozen = {}
    for which in ('dev','test_a','test_b'):
        selected = rows[which]
        x, _, comments, base = build_observations(texts,selected)
        scores = {name:model.decision_function(x) for name,model in models.items()}
        if which == 'dev':
            frozen = {k:cutoff(y[selected],base,v) for k,v in scores.items()}
            for key,value in frozen.items():
                if not np.isfinite(value):
                    raise ValueError(f'Nonfinite calibration cutoff for {key}')
            # Freeze all policies before computing held-out results.
            write_json(DEST/'policy.json',dict(cutoffs=frozen,synthetic_weight=SYNTHETIC_WEIGHT,training=stats,
                       gold_tagged_span_locations=body_counts,code_sha256=checksum(Path(__file__)),
                       heads_sha256=checksum(path),partition_sha256=checksum(ROOT/'round13/partition.npz'),
                       data_manifest_sha256=checksum(ROOT/'data/manifest.json')))
        output[which] = table(which,selected,y[selected],projects[selected],scores,comments,base,frozen)
        print(which,output[which]['baseline'],[(k,v['frozen'],v['matched'])
                                               for k,v in output[which]['results'].items()],flush=True)
    write_json(DEST/'results.json',output)
    with (DEST/'comparison.csv').open('w',newline='') as handle:
        out = csv.writer(handle)
        out.writerow(('partition','arm','protocol','tp','fp','fn','f1','recovered','recovered_without_comment'))
        fields = ('tp','fp','fn','f1','recovered','recovered_without_comment')
        for which, row in output.items():
            out.writerow((which,'canonical','released',*[row['baseline'][key] for key in fields]))
            for arm, res in row['results'].items():
                for mode in ('frozen','matched'):
                    out.writerow((which,arm,mode,*[res[mode][key] for key in fields]))


if __name__ == '__main__':
    main()
