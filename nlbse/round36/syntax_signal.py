"""Test anonymised syntax for genuinely omitted MAT comments."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

from comment_signal import HERE, ROOT, PARTS, canonical_mat, checksum, dataset, split, write_json, cutoff, table, coverage

sys.path.insert(0, str(HERE/'vendor'))
from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp
from paired_provenance import paired_examples, learn, MARKER, CLEAN

DEST = HERE/'syntax'
C = Parser(Language(tree_sitter_c.language()))
CPP = Parser(Language(tree_sitter_cpp.language()))
HASHER = FeatureHasher(n_features=2**17, input_type='dict', dtype=np.float32)
ANCHORS = {'function_definition', 'lambda_expression'}
IDENTIFIERS = {'identifier', 'type_identifier', 'field_identifier', 'namespace_identifier'}
NODE_LIMIT = 2400
BYTE_LIMIT = 24_000


def error_weight(root):
    if not root.has_error:
        return 0
    pending = [root]
    weight = 0
    while pending:
        node = pending.pop()
        if node.is_error:
            weight += max(1, node.end_byte-node.start_byte)
        elif node.is_missing:
            weight += 1
        elif node.has_error:
            pending.extend(node.children)
    return weight


def parse(text):
    raw = text.encode('utf-8', 'replace')[:BYTE_LIMIT]
    root = C.parse(raw).root_node
    first = error_weight(root)
    if first:
        other = CPP.parse(raw).root_node
        if error_weight(other) < first:
            root = other
    return root


def identity(node):
    if node.type in IDENTIFIERS:
        return 'identifier'
    return node.type


def grammar(text):
    root = parse(text)
    declaration, body = Counter(), Counter()
    queue = [(root, '', '', False)]
    visited = 0
    while queue and visited < NODE_LIMIT:
        node, parent, ancestor, inside = queue.pop()
        if node.type == 'comment':
            continue
        counts = body if inside else declaration
        current = identity(node)
        visited += 1
        counts['node/'+current] += 1
        if parent:
            counts['edge/'+parent+'/'+current] += 1
        if ancestor:
            counts['path/'+ancestor+'/'+parent+'/'+current] += 1
        children = [(i,c) for i,c in enumerate(node.children) if c.is_named and c.type != 'comment']
        for left, right in zip(children, children[1:]):
            counts['siblings/'+current+'/'+identity(left[1])+'/'+identity(right[1])] += 1
        for i, child in reversed(children):
            field = node.field_name_for_child(i)
            subtree = inside or (node.type in ANCHORS and field == 'body')
            if field:
                counts['field/'+current+'/'+field+'/'+identity(child)] += 1
            queue.append((child, current, parent, subtree))
    for counts in (declaration, body):
        for key in counts:
            counts[key] = np.log1p(counts[key])
    return declaration, body


def features(texts):
    declaration, body = zip(*(grammar(text) for text in texts), strict=True)
    prefix, implementation = HASHER.transform(declaration), HASHER.transform(body)
    return dict(declaration=prefix, body=implementation,
                joined=hstack((prefix, implementation), format='csr'))


def score(texts, models):
    design = features(texts)
    return {key:head.decision_function(design[key.split('_')[0]]) for key, head in models.items()}


def main():
    DEST.mkdir(exist_ok=True)
    texts, labels, projects, files = dataset()
    gold = labels[:, 1].astype(bool)
    ids = split()
    audit_file = ROOT/'round31/mat/fit_provenance.json'
    audit = json.loads(audit_file.read_text())
    assert audit['fit_ids_sha256'] == hashlib.sha256(ids['fit'].tobytes()).hexdigest()
    provenance = {item['row']:item for item in audit['observations']
                  if item['result'] in (MARKER, CLEAN)}
    eligible = np.array(sorted(provenance))
    assert np.isin(eligible, ids['fit']).all()
    cases = [r for r in eligible if provenance[r]['gold']]
    # A match without a marker can still be a natural positive of unknown origin.
    fit = np.array([r for r in eligible if not provenance[r]['gold'] or provenance[r]['result']==MARKER])
    assert np.array_equal(gold[fit], [provenance[r]['gold'] for r in fit])
    assert not canonical_mat([texts[r] for r in fit]).any()
    wins, losses = paired_examples(provenance, files, texts, [r for r in cases if provenance[r]['result']==MARKER])
    heads = {}
    for key,x in features([texts[r] for r in fit]).items():
        point = LogisticRegression(solver='liblinear', C=1., class_weight='balanced', max_iter=160,
                                   random_state=42)
        heads[key+'_point'] = point.fit(x, gold[fit])
        heads[key+'_pair'] = learn(x, wins, losses, fit)
    weight_file = DEST/'heads.joblib'
    joblib.dump(heads, weight_file, compress=3)
    fit_counts = dict(positives=int(gold[fit].sum()), negatives=int((~gold[fit]).sum()),
                      pairs=int(wins.size), paired_positives=int(np.unique(wins).size),
                      paired_controls=int(np.unique(losses).size))
    print('fit',fit_counts,flush=True)
    thresholds, results = {}, {}
    for part in PARTS:
        start = perf_counter()
        rows = ids[part]
        local = [texts[i] for i in rows]
        base = canonical_mat(local)
        reference = ROOT/'round31/mat/paired_provenance'/f'{part}_predictions.npz'
        with np.load(reference, allow_pickle=True) as z:
            assert np.array_equal(z['ids'], rows) and np.array_equal(z['base'], base)
            assert np.array_equal(z['y'], gold[rows])
            comments = z['comment']
        active = np.flatnonzero(~base)
        rank = {}
        for key,value in score([local[i] for i in active], heads).items():
            rank[key] = np.full(len(rows), -100.)
            rank[key][active] = value
        if part == 'dev':
            thresholds = {key:cutoff(gold[rows], base, value) for key,value in rank.items()}
            write_json(DEST/'policy.json', dict(
                fit=fit_counts, cutoffs=thresholds, provenance_sha256=checksum(audit_file),
                partition_sha256=checksum(ROOT/'round13/partition.npz'), code_sha256=checksum(Path(__file__)),
                dependencies_sha256=checksum(HERE/'syntax_requirements.txt'), model_sha256=checksum(weight_file),
                data_manifest_sha256=checksum(ROOT/'data/manifest.json'), bytes=BYTE_LIMIT, nodes=NODE_LIMIT,
                inference_uses_function_only=True))
        result = table(part,rows,gold[rows],projects[rows],rank,comments,base,thresholds,dest=DEST)
        result['coverage'] = coverage(local, gold[rows], base, comments, rank, projects[rows])
        result['seconds'] = perf_counter()-start
        result['reference_sha256'] = checksum(reference)
        results[part] = result
        print(part,result['baseline'],{key:(r['frozen']['recovered'],r['frozen']['delta_fp'],r['matched']['recovered'])
                                      for key,r in result['results'].items()},flush=True)
    write_json(DEST/'results.json',results)
    with (DEST/'comparison.csv').open('w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('partition','arm','frozen_tp','frozen_fp','matched_tp','matched_fp'))
        for part,result in results.items():
            for key,row in result['results'].items():
                writer.writerow((part,key,row['frozen']['recovered'],row['frozen']['delta_fp'],
                                 row['matched']['recovered'],row['matched']['delta_fp']))


if __name__ == '__main__':
    main()
