"""Replay the MAT decisions and quantify irreducible collisions of visible input."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from mat52 import HERE, ROOT, DEST, PARTS, checksum, dataset, split, write_json
from transfer import canonical_mat


def read(path):
    return json.loads(path.read_text())


def verify(destination, code, artifact_key, artifact_name):
    _, labels, projects, _ = dataset()
    parts = split()
    policy = read(destination / 'policy.json')
    results = read(destination / 'results.json')
    for key, path in {'code_sha256': HERE / code, artifact_key: destination / artifact_name,
                      'train_manifest_sha256': ROOT / 'data/manifest.json',
                      'partition_sha256': ROOT / 'round13/partition.npz'}.items():
        assert policy[key] == checksum(path)
    checked = {}
    for part in PARTS:
        path = destination / f'{part}_predictions.npz'
        assert checksum(path) == results[part]['predictions_sha256']
        with np.load(path) as z:
            ids = z['ids']
            y, base, project = (z[k] for k in ('y', 'base', 'group'))
            assert np.array_equal(ids, parts[part]) and np.array_equal(y, labels[ids, 1])
            assert np.array_equal(project, projects[ids])
            claims = {}
            for name, cut in policy['cutoffs'].items():
                scores = z[f'{name}_score']
                assert np.isfinite(scores).all()
                if part == 'dev':
                    assert cut == scores[~y & ~base].max()
                frozen = base | (scores > cut)
                assert np.array_equal(frozen, z[f'{name}_frozen'])
                matched = z[f'{name}_matched']
                counts = {}
                for p in np.unique(project):
                    local = project == p
                    threshold = scores[local & ~base & ~y].max()
                    assert np.array_equal(matched[local], base[local] | (scores[local] > threshold))
                    added = matched & ~base & local
                    assert not (added & ~y).any()
                    counts[p] = int((added & y).sum())
                recorded = results[part]['results'][name]
                added = frozen & ~base
                totals = {'frozen_tp': int((added & y).sum()), 'frozen_fp': int((added & ~y).sum()),
                          'matched_tp': sum(counts.values()), 'matched_tp_by_project': counts}
                assert totals['frozen_tp'] == recorded['frozen']['recovered']
                assert totals['frozen_fp'] == recorded['frozen']['delta_fp']
                assert totals['matched_tp'] == recorded['matched']['recovered']
                claims[name] = totals
            checked[part] = dict(arms=claims, predictions_sha256=checksum(path))
    return checked


def visible_collisions(texts, y, project, rows, base):
    # A shared input with opposite gold labels forces any deterministic predictor
    # using Function alone to trade a false positive for the missed positive.
    miss = rows[y[rows] & ~base]
    negative = defaultdict(Counter)
    digest = lambda text: sha256(text.encode()).hexdigest()
    targets = {digest(texts[i]): i for i in miss}
    identical = 0
    example = []
    for i in rows[~y[rows]]:
        key = digest(texts[i])
        if key in targets:
            negative[key][project[i]] += 1
    within_project = 0
    for i in miss:
        key = digest(texts[i])
        if key in negative:
            identical += 1
            within_project += int(project[i] in negative[key])
            example.append(dict(row=int(i), function_sha256=key, negative_projects=dict(negative[key])))
    return dict(misses=len(miss), identical_negative=identical, same_project_negative=within_project,
                pairs=example)


def main():
    proof = verify(DEST, 'mat52.py', 'weights_sha256', 'weights.joblib')
    references = verify(HERE / 'references', 'reference52.py', 'index_sha256', 'index.json')
    texts, label, project, _ = dataset()
    groups = split()
    collisions = {}
    for part in PARTS:
        with np.load(DEST / f'{part}_predictions.npz') as z:
            collisions[part] = visible_collisions(texts, label[:, 1].astype(bool), project, groups[part], z['base'])
    texts, label, project, _ = dataset(public=True)
    baseline = canonical_mat(texts)
    release = read(ROOT / 'round38/release_summary.json')
    assert int((label[:, 1].astype(bool) & ~baseline).sum()) == release['MAT']['fn']
    collisions['public'] = visible_collisions(texts, label[:, 1].astype(bool), project, np.arange(len(texts)), baseline)
    evidence = ROOT / 'round48/madewic/unmatched.json'
    source = ROOT / 'round48/madewic/results.json'
    quality = release['quality']['mean']['f1']
    write_json(HERE / 'audit.json', dict(verification=proof, reference_verification=references,
               visible_function_collisions=collisions, score_at_zero_cost=.6 * quality + .4,
               release_score=release['score'], target_gap=release['target_gap'],
               required_public_mat_at_zero_fp=release['changes_to_target_at_fixed_cost']['tp_m'],
               provenance={'source_sha256': checksum(source), 'unmatched_sha256': checksum(evidence)},
               code_sha256=checksum(Path(__file__))))
    print({k: {name: n for name, n in v.items() if name != 'pairs'} for k, v in collisions.items()}, flush=True)


if __name__ == '__main__':
    main()
