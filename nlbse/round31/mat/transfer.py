"""Test whether tagged documentation transfers across code and comment templates."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'round26/mat'), str(ROOT / 'round21/mat'), str(ROOT / 'round14')]
from audit import checksum, dataset, split, write_json
from marker import LETTERS, TAGS, upstream_tag
from mat_runtime import SOURCE_COMMENT, canonical_mat
from project_split import TOKEN

WORD = re.compile(r'[a-z]{2,}|\d+', re.I | re.ASCII)
IDENT = re.compile(r'[a-z_]\w*', re.I | re.ASCII)
KEYWORDS = frozenset("""
alignas alignof asm auto bool break case catch char class const constexpr consteval constinit
continue default delete do double else enum explicit extern false final float for friend goto
if inline int long mutable namespace new noexcept nullptr operator override private protected
public register restrict return short signed sizeof static static_assert struct switch template
this throw true try typedef typename union unsigned using virtual void volatile while
""".split())
MODES = ('body', 'body_alpha', 'documentation', 'code_or_doc', 'code_and_doc')
FEATURES = ('body', 'body_alpha', 'whole_comment', 'comment_line')


def digest(terms):
    return hashlib.blake2b('\0'.join(terms).encode(), digest_size=16).digest()


def marker_word(raw):
    word = LETTERS.sub('', raw).lower()
    return 2 < len(word) < 20 and (
        (word.startswith(TAGS) or word.endswith(TAGS)) and ('xxx' not in word or word == 'xxx')
    )


def comment_terms(text):
    # Use the same normalization in marked and unmarked comments.
    return [term for word in text.split() if not marker_word(word)
            for term in WORD.findall(word.casefold())]


def evidence(text):
    tokens = []
    docs = []
    for token in TOKEN.finditer(text):
        if token['comment'] is None:
            tokens.append(token[0])
            continue
        doc = token[0].strip('/*')
        marked = upstream_tag(token[0])
        terms = comment_terms(doc)
        if 6 <= len(terms) <= 100 and len(set(terms)) >= 5:
            docs.append(('whole_comment', digest(terms), marked))
        for line in doc.splitlines():
            terms = comment_terms(line)
            if 7 <= len(terms) <= 45 and len(set(terms)) >= 6:
                # A tagged line offers stronger evidence than a different line in the same block.
                docs.append(('comment_line', digest(terms), marked and upstream_tag(line)))
    body = []
    alpha = []
    if '{' in tokens and tokens[-1] == '}' and 24 <= len(tokens) <= 3000:
        start = tokens.index('{')
        body = tokens[start:]
        if len(body) >= 24:
            names = {}
            for i, term in enumerate(body):
                if IDENT.fullmatch(term) and term not in KEYWORDS and not term.isupper():
                    call = i+1 < len(body) and body[i+1] == '('
                    member = i > 0 and body[i-1] in ('.', ':')
                    if not call and not member:
                        term = 'name:' + str(names.setdefault(term, len(names)))
                alpha.append(term)
    code = [('body', digest(body), False), ('body_alpha', digest(alpha), False)] if alpha else []
    return code + list(set(docs))


def indexes(texts, y, projects, files, rows):
    store = defaultdict(dict)
    for i in rows:
        positive = bool(y[i])
        for kind, key, tagged in evidence(texts[i]):
            by_file = store[kind, key]
            file_id = (projects[i], files[i])
            pair = (kind.startswith('body') and positive) or (tagged and positive)
            negative = not positive
            # Keep each source file at unit weight and retain conflicting evidence.
            by_file[file_id] = by_file.get(file_id, 0) | int(pair) | (int(negative) << 1)
    return {key: {loc: state for loc, state in by_file.items()}
            for key, by_file in store.items() if any(state & 1 for state in by_file.values())}


def rank(keys, project, store):
    counts = np.zeros((len(FEATURES), 2), dtype=np.int32)
    for kind, token, _ in keys:
        hits = store.get((kind, token))
        if hits is None:
            continue
        cohort = [state for (owner, _), state in hits.items() if owner != project]
        positive = sum(bool(state & 1) for state in cohort)
        negative = sum(bool(state & 2) for state in cohort)
        j = FEATURES.index(kind)
        previous, opposition = counts[j]
        if (positive/(2.+negative), positive, -negative) > (previous/(2.+opposition), previous, -opposition):
            counts[j] = [positive, negative]
    return counts


def proposals(x):
    positive, negative = x.transpose(2, 1, 0)
    # Reject conflicting matches for the precision-first operation.
    score = np.where(negative == 0, np.log1p(positive), 0.)
    exact, alpha, full, line = score
    doc = np.maximum(full, line)
    code = np.maximum(exact, alpha)
    return dict(body=exact, body_alpha=alpha, documentation=doc,
                code_or_doc=np.maximum(code, doc), code_and_doc=np.minimum(code, doc))


def record(y, pred, base, comment):
    def tally(p):
        tp = int((y & p).sum())
        fp = int((~y & p).sum())
        fn = int((y & ~p).sum())
        return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(1, 2*tp+fp+fn))
    before, after = tally(base), tally(pred)
    new = y & pred & ~base
    return dict(**after, recovered=int(new.sum()), delta_fp=after['fp']-before['fp'],
                recovered_without_comment=int((new & ~comment).sum()))


def cut(y, base, score):
    # Cut at the greatest negative score; ties are negative. Zero provides no evidence.
    negative = score[~y & ~base]
    return max(0., float(negative.max(initial=0.)))


def inputs(rows, texts, y, projects, store):
    group = projects[rows]
    local = [texts[i] for i in rows]
    start = time.perf_counter()
    base = canonical_mat(local)
    comment = np.asarray([any(token['comment'] is not None for token in TOKEN.finditer(t)) for t in local])
    x = np.zeros((len(rows), len(FEATURES), 2), dtype=np.int32)
    for j in np.flatnonzero(~base):
        x[j] = rank(evidence(local[j]), group[j], store)
    return dict(ids=rows, y=y[rows], project=group, base=base, comment=comment, features=x,
                seconds=time.perf_counter()-start)


def report(which, data, frozen):
    y, base, groups, comment = (data[key] for key in ('y', 'base', 'project', 'comment'))
    scores = proposals(data['features'])
    checked = {}
    predictions = {}
    for arm, score in scores.items():
        fixed = base | (score > frozen[arm])
        raw = base | (score > 0.)
        per = {}
        matched = base.copy()
        for project in np.unique(groups):
            mask = groups == project
            local_cut = cut(y[mask], base[mask], score[mask])
            matched[mask] |= score[mask] > local_cut
            per[project] = dict(cut=local_cut, **record(y[mask], matched[mask], base[mask], comment[mask]))
            if per[project]['delta_fp'] != 0:
                raise ValueError('Changed fixed project false positive count')
        misses = y & ~base
        model_matched = record(y, matched, base, comment)
        checked[arm] = dict(cut=frozen[arm], frozen=record(y, fixed, base, comment),
                            matched=model_matched, any_evidence=record(y, raw, base, comment), projects=per,
                            matched_miss_rows=data['ids'][misses & matched].tolist())
        predictions.update({arm+'_frozen': fixed, arm+'_matched': matched, arm+'_score': score})
    path = HERE/f'{which}_predictions.npz'
    np.savez_compressed(path, **data, **predictions)
    return dict(rows=len(y), elapsed_seconds=data['seconds'], baseline=record(y, base, base, comment),
                missed_without_comment=int((y & ~base & ~comment).sum()), metrics=checked,
                candidate_coverage={k:dict(miss_positive=int(((data['features'][:,i,0]>0) & y & ~base).sum()),
                                          negative=int(((data['features'][:,i,0]>0) & ~y & ~base).sum()))
                                    for i,k in enumerate(FEATURES)},
                predictions_sha256=checksum(path))


def main():
    texts, labels, projects, files = dataset()
    y = labels[:,1].astype(bool)
    rows = split()
    start = time.perf_counter()
    store = indexes(texts, y, projects, files, rows['fit'])
    print('fit', len(rows['fit']), len(store), round(time.perf_counter()-start, 2), flush=True)
    dev = inputs(rows['dev'], texts, y, projects, store)
    frozen = {k:cut(dev['y'], dev['base'], val) for k,val in proposals(dev['features']).items()}
    dev_result = report('dev', dev, frozen)
    # Preserve the released rule on ties; frozen selection uses development files only.
    selection = max(MODES, key=lambda k:dev_result['metrics'][k]['frozen']['recovered'])
    if dev_result['metrics'][selection]['frozen']['recovered'] == 0:
        selection = 'canonical'
    config = dict(selection=selection, frozen=frozen, fit_rows=len(rows['fit']),
                  features=FEATURES, partition_sha256=checksum(ROOT/'round13/partition.npz'),
                  data_manifest_sha256=checksum(ROOT/'data/manifest.json'),
                  code_sha256=checksum(Path(__file__)), prediction_field='Function',
                  fit_ids_sha256=hashlib.sha256(rows['fit'].tobytes()).hexdigest(),
                  project_metadata_usage='exclude same-project fit hits for validation; never serves as predictor',
                  marker_sha256=checksum(ROOT/'round21/mat/marker.py'))
    write_json(HERE/'policy.json', config)
    results = {'dev':dev_result}
    for which in ('test_a','test_b'):
        item = inputs(rows[which], texts, y, projects, store)
        results[which] = report(which, item, frozen)
        print(which, results[which]['baseline'], results[which]['candidate_coverage'], flush=True)
    write_json(HERE/'results.json', results)
    with (HERE/'comparison.csv').open('w', newline='') as handle:
        out = csv.writer(handle)
        out.writerow(('partition','candidate','scenario','misses','recovered','recovered_without_comment','fp','f1'))
        for which, result in results.items():
            base = result['baseline']
            out.writerow((which,'canonical','released',base['fn'],0,0,base['fp'],base['f1']))
            for arm, values in result['metrics'].items():
                for name in ('frozen','matched','any_evidence'):
                    entry = values[name]
                    out.writerow((which,arm,name,base['fn'],entry['recovered'],
                                  entry['recovered_without_comment'],entry['fp'],entry['f1']))
    print(json.dumps(dict(selection=selection, frozen=frozen)), flush=True)


if __name__ == '__main__':
    main()
