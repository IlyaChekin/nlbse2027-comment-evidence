"""Test physical comment-boundary and stub evidence using only Function."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'round31/mat'))
from transfer import TOKEN, canonical_mat, checksum, dataset, split, write_json
from placement import cutoff, table
from paired_provenance import MARKER, paired_examples

DEST = HERE/'shape'
AUDIT = ROOT/'round31/mat/fit_provenance.json'
PARTS = ('dev', 'test_a', 'test_b')
SIZE = 128
EMPTY = frozenset(['0', '1', 'false', 'true', 'nullptr', 'NULL', 'null', 'undefined'])
FLOW = frozenset(['if', 'else', 'switch', 'for', 'while', 'catch', 'goto', 'case', 'throw'])
GAP = re.compile(r'\n[ \t]*\r?\n')
NOTE = re.compile(r'\b(?:unimplemented|unsupported|unreachable|placeholder|stub|workaround|not (?:yet )?implemented)\b', re.I)
DIAGNOSTIC = re.compile(r'\b(?:abort|panic|BUG|BUG_ON|assert|static_assert|g_assert_not_reached|unreachable|'
                        r'__builtin_unreachable|NOTIMPLEMENTED|NOTREACHED|MOZ_CRASH|UNREACHABLE)\b')
ERROR = re.compile(r'\b(?:ENOSYS|ENOTSUP|EOPNOTSUPP|E_NOTIMPL|NS_ERROR_NOT_IMPLEMENTED)\b')
DEFER = re.compile(r'\b(?:todo|fixme|hack|xxx)\b', re.I)
IDENT = re.compile(r'[A-Za-z_]\w*')
DOC = re.compile(r'/\*[*!]|//[/!]')

def shape(text):
    layout = {}
    stub = {}
    ncode = 0
    first = None
    last_leading = None
    pos_after_header = None
    n_body = 0
    comments = []
    body = []
    signature = []
    literal_evidence = 0
    for span in TOKEN.finditer(text):
        value = span[0]
        if span['comment'] is not None:
            comments.append((span, ncode, pos_after_header is not None, n_body))
            if ncode == 0:
                last_leading = span
            continue
        if first is None:
            first = span
        ncode += 1
        if pos_after_header is None:
            signature.append(value)
            if value == '{':
                pos_after_header = span.end()
        else:
            body.append(value)
            n_body += 1
        if value.startswith(('"', 'R"')) and NOTE.search(value):
            literal_evidence += 1

    start = first.start() if first is not None else len(text)
    before = text[:start]
    header_gap = text[last_leading.end():start] if last_leading is not None else before
    layout.update(initial_newlines=before.count('\n'), initial_padding=len(before),
                  indent_after_last_break=len(before.rsplit('\n', 1)[-1]),
                  first_code_gap_newlines=header_gap.count('\n'),
                  initial_blank_lines=len(GAP.findall(before)),
                  crlf=text.count('\r\n'), tabs=text.count('\t'),
                  spaces_before_newline=len(re.findall(r'[ \t]+\r?\n', text)),
                  blank_pairs=len(GAP.findall(text)), line_splice=text.count('\\\n'),
                  lines=text.count('\n')+1, characters=len(text),
                  header_tokens=len(signature), body_tokens=len(body))
    for section in ('prefix', 'header', 'early_body', 'later_body'):
        c = [span for span, n, inner, position in comments
             if section == ('prefix' if n == 0 else 'header' if not inner else
                             'early_body' if position <= 4 else 'later_body')]
        layout.update({section+'_count': len(c),
                       section+'_doc': sum(DOC.match(v[0]) is not None for v in c),
                       section+'_max_lines': max((v[0].count('\n')+1 for v in c), default=0),
                       section+'_note': sum(NOTE.search(v[0]) is not None for v in c),
                       section+'_tag_like': sum(DEFER.search(v[0]) is not None for v in c),
                       section+'_standalone': sum(not text[text.rfind('\n', 0, v.start())+1:v.start()].strip()
                                                 for v in c)})
    layout['first_body_gap_newlines'] = 0
    if body and pos_after_header is not None:
        match = re.match(r'[\s]*', text[pos_after_header:])
        layout['first_body_gap_newlines'] = match[0].count('\n')
    structure = Counter(body)
    return_indices = [i for i,v in enumerate(body) if v == 'return']
    terminals = []
    for i in return_indices:
        end = i+1
        while end < min(len(body), i+12) and body[end] != ';':
            end += 1
        terms = body[i+1:end]
        terminals.append(' '.join(terms) if len(terms) <= 4 else None)
    stub.update(statements=structure[';'], flow=sum(structure[k] for k in FLOW),
                braces=structure['{'], returns=len(return_indices),
                singleton_return=bool(len(terminals)==1 and structure[';']==1),
                void_return=terminals.count(''), sentinel_return=sum(t in EMPTY for t in terminals),
                negative_return=sum(t is not None and t.startswith('- ') for t in terminals),
                return_error=sum(bool(t and ERROR.search(t)) for t in terminals),
                failure_literal=literal_evidence,
                failure_call=sum(bool(DIAGNOSTIC.fullmatch(v)) and body[i+1]=='('
                                 for i,v in enumerate(body[:-1])),
                error_identifier=sum(bool(ERROR.fullmatch(v)) for v in body),
                void_cast=sum(body[i:i+3]==['(', 'void', ')'] for i in range(len(body)-2)),
                assignments=structure['='], nullptr_mentions=structure['NULL']+structure['nullptr'],
                calls=sum(bool(IDENT.fullmatch(v)) and body[i+1]=='(' for i,v in enumerate(body[:-1])),
                no_body=pos_after_header is None, stub_empty=body==['}'])
    return layout, stub, bool(comments)


def observe(texts, ids):
    started = perf_counter()
    layouts, stubs, visible = zip(*(shape(texts[i]) for i in ids), strict=True)
    lx = np.log1p(np.array([list(row.values()) for row in layouts], dtype=np.float32))
    sx = np.log1p(np.array([list(row.values()) for row in stubs], dtype=np.float32))
    names = {'layout': list(layouts[0]), 'stub': list(stubs[0])}
    z = {'layout': lx, 'stub': sx, 'joint': np.column_stack((lx, sx))}
    base = canonical_mat([texts[i] for i in ids])
    print('features', len(ids), round(perf_counter()-started, 3), flush=True)
    return z, base, np.array(visible), names


def fn_report(y, base, projects, visible, provenance):
    misses = y & ~base
    masks = {k: projects == k for k in np.unique(projects)} | {'all': np.ones(len(y), bool)}
    return {key: dict(fn=int((misses & m).sum()), fn_without_comment=int((misses & m & ~visible).sum()),
                      fn_with_comment=int((misses & m & visible).sum()),
                      provenance=Counter(str(v) for v in provenance[misses & m]))
            for key,m in masks.items()}


def first_false(y, base, project, scores):
    out = {}
    for group in np.unique(project):
        residual = (project == group) & ~base
        positive = scores[residual & y]
        negatives = scores[residual & ~y]
        if len(positive):
            out[group] = dict(first_tp_fp=int((negatives >= positive.max()).sum()),
                              fn=int(len(positive)), negatives=int(len(negatives)))
    return out


def main():
    DEST.mkdir(exist_ok=True)
    texts, labels, projects, files = dataset()
    gold = labels[:, 1].astype(bool)
    indices = split()
    train = indices['fit']
    assert not np.isin(projects[train], np.unique(projects[np.r_[indices['test_a'],indices['test_b']]])).any()
    audit = json.loads(AUDIT.read_text())
    assert audit['fit_ids_sha256'] == hashlib.sha256(train.tobytes()).hexdigest()
    prior = {v['row']: v for v in audit['observations']}
    assert np.isin(list(prior), train).all() and all(gold[i] == v['gold'] for i,v in prior.items())
    confirmed = [i for i,v in prior.items() if v['gold'] and v['result'] == MARKER]
    train_pos, train_neg = paired_examples(prior, files, texts, confirmed)
    holdout = np.r_[indices['dev'],indices['test_a'],indices['test_b']]
    # Exact copies are ineligible for training. The envelope is derived before scoring.
    other = {hashlib.blake2b(texts[i].encode(),digest_size=16).digest() for i in holdout}
    eligible = np.fromiter((hashlib.blake2b(texts[i].encode(),digest_size=16).digest() not in other
                            for i in train), bool)
    joint, base, _, names = observe(texts, train)
    keep = eligible & ~base
    y_fit = gold[train]
    a,b = (np.searchsorted(train, x) for x in (train_pos, train_neg))
    allowed = keep[a] & keep[b]
    a,b = a[allowed],b[allowed]
    pos_counts = np.bincount(a, minlength=len(train))
    pair_w = 1./pos_counts[a]
    classifiers = {}
    for name,x in joint.items():
        scaler = StandardScaler().fit(x[keep])
        difference = scaler.transform(x[a])-scaler.transform(x[b])
        paired = LogisticRegression(solver='liblinear',fit_intercept=False,C=.05,random_state=48)
        paired.fit(np.r_[difference, -difference], np.r_[np.ones(len(a)),np.zeros(len(a))],
                   sample_weight=np.tile(pair_w,2))
        point = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, learning_rate=.06,
                                               min_samples_leaf=80, l2_regularization=10, random_state=48)
        point.fit(x[keep], y_fit[keep], sample_weight=np.where(y_fit[keep], 8., 1.))
        classifiers[name] = (scaler, paired, point)
    weights = DEST/'weights.joblib'
    joblib.dump(classifiers, weights, compress=3)
    refs = ROOT/'round41/mat/text'
    receipt = ROOT/'round41/mat/source_context/receipt.json'
    old_provenance = {}
    if receipt.exists():
        for mode in ('test_a','test_b'):
            with np.load(receipt.parent/f'{mode}_predictions.npz') as old:
                old_provenance[mode] = {int(i): 'tag_attached_in_source' for i,rank in
                                        zip(old['ids'],old['owner_aligned_score'],strict=True) if rank > 0}
    stats = {}
    frozen = {}
    selected = 'canonical'
    for part in PARTS:
        ids = indices[part]
        m, base, visible, _ = observe(texts, ids)
        y = gold[ids]
        with np.load(refs/f'{part}_predictions.npz') as control:
            assert np.array_equal(ids, control['ids']) and np.array_equal(base, control['base'])
            scores = {'previous_joint': control['joint_score']}
        for name,(scaler,paired,point) in classifiers.items():
            scores[name+'_paired'] = paired.decision_function(scaler.transform(m[name]))
            scores[name+'_point'] = point.decision_function(m[name])
        scores['layout_stub_consensus'] = np.minimum(scores['layout_paired'],scores['stub_paired'])
        if part == 'dev':
            frozen = {name:cutoff(y,base,value) for name,value in scores.items()}
            recovery = {name:int((y & ~base & (value>frozen[name])).sum()) for name,value in scores.items()
                        if name != 'previous_joint'}
            best = max(recovery, key=recovery.get)
            selected = best if recovery[best] else 'canonical'
            write_json(DEST/'policy.json', dict(selected=selected, cutoffs=frozen, dev_recovered=recovery,
                       fit_rows=len(train), fit_residual_rows=int(keep.sum()), fit_misses=int((keep & y_fit).sum()),
                       excluded_exact_copies=int((~eligible).sum()), confirmed_pairs=len(a),
                       confirmed_cases=len(np.unique(a)), confirmed_controls=len(np.unique(b)),
                       features=names, input='Function', data_sha256=checksum(ROOT/'data/manifest.json'),
                       audit_sha256=checksum(AUDIT), partition_sha256=checksum(ROOT/'round13/partition.npz'),
                       weights_sha256=checksum(weights), code_sha256=checksum(Path(__file__))))
        known = old_provenance.get(part, {})
        provenance = np.array([known.get(i,'unverified_in_source') for i in ids])
        stats[part] = table(part,ids,y,projects[ids],scores,visible,base,frozen,dest=DEST)
        stats[part]['misses'] = fn_report(y,base,projects[ids],visible,provenance)
        stats[part]['first_fp'] = {name:first_false(y,base,projects[ids],values) for name,values in scores.items()}
        print(part,stats[part]['baseline'],{k:[v['matched']['recovered'],v['frozen']['recovered'],
              v['frozen']['delta_fp']] for k,v in stats[part]['results'].items()},flush=True)
    write_json(DEST/'results.json', stats)
    with (DEST/'comparison.csv').open('w',newline='') as handle:
        out = csv.writer(handle)
        out.writerow(('part','arm','baseline_fn','baseline_fp','recovered_at_same_fp',
                      'frozen_recovered','frozen_delta_fp','frozen_f1','min_project_fp_for_first_tp'))
        for part,item in stats.items():
            for name,arm in item['results'].items():
                first = min((x['first_tp_fp'] for x in item['first_fp'][name].values()),default=None)
                out.writerow((part,name,item['baseline']['fn'],item['baseline']['fp'],arm['matched']['recovered'],
                             arm['frozen']['recovered'],arm['frozen']['delta_fp'],arm['frozen']['f1'],first))
    print('selected', selected, flush=True)


if __name__ == '__main__':
    main()
