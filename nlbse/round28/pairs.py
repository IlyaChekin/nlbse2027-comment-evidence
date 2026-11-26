"""Mine paired labels from within files using two out-of-file prediction sources."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'round26'))
from disagreement import LIMIT, f, checksum, json_at, write_json, estimate_training, clip_logit
from compact_oof import signed_source

PARTS = ('test_a', 'test_b', 'official')
OUT = HERE/'pairs'


def mining(rotation):
    expert, sig = signed_source(rotation)
    location = ROOT/'round26/compact'/rotation
    file = location/'scores.npz'
    manifest = json_at(location/'manifest.json')
    if checksum(file) != manifest['sha256'] or manifest['sparse_oof_sha256'] != sig:
        raise ValueError('Out-of-file predictions changed')
    choose = expert['weak'] == 0
    ids, gold, folds = (expert[k][choose] for k in ('ids', 'y', 'fold'))
    with np.load(file) as z:
        if not all(np.array_equal(z[k], expert[k]) for k in ('ids', 'y', 'fold')):
            raise ValueError('The two outside-file sources disagree')
        p = z['values'][choose, 2].astype(np.float64)
    sparse = expert['logits'][choose, 0]
    probs = estimate_training(gold, p, sparse, folds)
    a, b, joint = probs.T
    gold = gold.astype(bool)
    miss = gold & (np.maximum(a,b) < LIMIT)
    false = ~gold & (np.minimum(a,b) >= LIMIT)
    # A control has the opposite label with a score near the hard example.
    control_pos = gold & (joint >= LIMIT)
    control_neg = ~gold & (joint < LIMIT)
    text, labels, projects, paths, _ = f.inputs()
    if not np.array_equal(labels[ids, 0], gold):
        raise ValueError('Mismatched main-label order')
    lengths = np.fromiter((len(text[i]) for i in ids), dtype=np.int32)
    loglength = np.log1p(lengths)
    confidence = clip_logit(joint)
    groups = defaultdict(list)
    for j, i in enumerate(ids):
        groups[projects[i], paths[i]].append(j)

    selected = {}
    kinds = Counter()
    covered_miss, covered_false = set(), set()
    for members in groups.values():
        ix = np.asarray(members)
        if np.unique(folds[ix]).size != 1:
            raise ValueError('A training file was split across folds')
        groups_for_pairs = ((ix[miss[ix]], ix[control_neg[ix]], 0),
                            (ix[false[ix]], ix[control_pos[ix]], 1),
                            (ix[miss[ix]], ix[false[ix]], 2))
        for anchors, controls, kind in groups_for_pairs:
            if not len(anchors) or not len(controls):
                continue
            for anchor in anchors:
                score = abs(confidence[controls]-confidence[anchor])
                size = abs(loglength[controls]-loglength[anchor])
                eligible = np.flatnonzero(size <= np.log(3.))
                if not len(eligible):
                    continue
                order = eligible[np.argsort(score[eligible]+1.5*size[eligible], kind='stable')][:2]
                for n in controls[order]:
                    j, k = (anchor, n) if gold[anchor] else (n, anchor)
                    if text[ids[j]] == text[ids[k]] or (j, k) in selected:
                        continue
                    selected[j, k] = kind
                    kinds[kind] += 1
                    if miss[j]:
                        covered_miss.add(j)
                    if false[k]:
                        covered_false.add(k)
    pairs = np.array(list(selected), dtype=np.int32)
    flags = np.fromiter(selected.values(), dtype=np.int8)
    if len(pairs) < 100 or not (gold[pairs[:,0]].all() and not gold[pairs[:,1]].any()):
        raise ValueError('The paired experiment has too few valid opposite-label contrasts')
    to = OUT/rotation
    to.mkdir(parents=True, exist_ok=True)
    path = to/'pairs.npz'
    np.savez_compressed(path, rows=ids, label=gold, fold=folds, pairs=pairs, role=flags,
                        oof=probs.astype(np.float32), length=lengths)
    result = dict(rows=len(ids), positive=int(gold.sum()), shared_misses=int(miss.sum()),
                  shared_false_positives=int(false.sum()), anchored_misses=len(covered_miss),
                  anchored_false_positives=len(covered_false), pairs=len(pairs), types=dict(kinds),
                  max_length_ratio=3, source_sha256={'sparse': sig, 'compact': checksum(file)},
                  predictions_sha256=checksum(path), code_sha256=checksum(Path(__file__)))
    write_json(to/'manifest.json', result)
    print(rotation, result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=PARTS)
    mining(p.parse_args().part)
