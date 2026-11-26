"""Test whether disagreement between a function's declaration and body recovers MAT."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEST = HERE / 'contract'
sys.path.insert(0, str(ROOT / 'round31/mat'))
from transfer import KEYWORDS, TOKEN, canonical_mat, checksum, dataset, digest, split, write_json
from placement import cutoff, table

PARTS = ('dev', 'test_a', 'test_b')
IDENT = re.compile(r'[A-Za-z_]\w*', re.ASCII)
PIECES = re.compile(r'[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+')
RETURN = re.compile(r'\breturn\s+([^;]{1,120})\s*;')
LEXEME = re.compile(TOKEN.pattern.removesuffix(r'|\w+|[^\s\w]') +
                    r'|(?:<<=|>>=|<=>|==|!=|<=|>=|\+\+|--|[+*/%&|^\-]=)|\w+|[^\s\w]',
                    TOKEN.flags)
MUTATIONS = ('=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=', '++', '--')
CALLABLE_NAMES = KEYWORDS | {'__attribute__', '__declspec', 'assert', 'typeof', '__typeof__'}
NULL = {'0', 'false', 'NULL', 'nullptr', 'null', '( void * ) 0'}
STUB_CODES = {'ENOSYS', 'ENOTSUP', 'EOPNOTSUPP', 'E_NOTIMPL', 'NOTREACHED', 'NOTIMPLEMENTED',
              'NS_ERROR_NOT_IMPLEMENTED', 'unimplemented', 'unsupported', 'NotImplementedError'}
HARD_FAIL = {'abort', 'panic', 'BUG', 'BUG_ON', 'MOZ_CRASH', 'UNREACHABLE', '__builtin_unreachable'}
ACTIONS = tuple(map(frozenset, (
    ('get', 'find', 'lookup', 'read', 'fetch', 'load', 'query', 'resolve', 'select'),
    ('set', 'write', 'store', 'save', 'put', 'update', 'replace', 'reset', 'clear'),
    ('new', 'alloc', 'allocate', 'create', 'init', 'setup', 'open', 'start', 'enable', 'register'),
    ('free', 'destroy', 'release', 'deinit', 'dispose', 'close', 'stop', 'disable', 'unregister'),
    ('add', 'insert', 'append', 'push', 'attach', 'link', 'enqueue'),
    ('remove', 'delete', 'pop', 'detach', 'unlink', 'dequeue'),
    ('is', 'has', 'can', 'valid', 'check', 'verify', 'test', 'compare', 'match'),
    ('parse', 'convert', 'format', 'encode', 'decode', 'copy', 'clone', 'compute', 'build'),
    ('send', 'recv', 'receive', 'connect', 'accept', 'bind', 'listen', 'transmit'),
    ('lock', 'unlock', 'acquire', 'wait', 'wake', 'schedule'),
)))
BY_WORD = {word: i for i, bank in enumerate(ACTIONS) for word in bank}
PLACEHOLDER = re.compile(r'\b(?:not (?:yet )?implemented|left unimplemented|implementation pending|stub for)\b', re.I)


@dataclass(frozen=True)
class Evidence:
    keys: tuple
    effects: tuple
    shape: tuple
    roles: tuple
    visible_comment: bool
    code_hash: bytes


def words(name):
    return [v.lower() for v in PIECES.findall(name)]


def declaration(header):
    depth = 0
    name, start, stop = '', 0, 0
    for i, token in enumerate(header):
        if token == '(':
            previous = header[i-1] if i else ''
            if depth == 0 and IDENT.fullmatch(previous) and previous not in CALLABLE_NAMES:
                name, start, stop = previous, i + 1, i + 1
            depth += 1
        elif token == ')':
            depth -= 1
            if depth == 0 and name and stop == start:
                stop = i
    return name, header[start:stop]


def observe(text):
    tokens, comments = [], []
    for m in LEXEME.finditer(text):
        if m['comment'] is None:
            tokens.append(m[0])
        else:
            comments.append(m[0])
    if '{' in tokens:
        start = tokens.index('{')
        head, body = tokens[:start], tokens[start + 1:]
    else:
        head, body = tokens, []
    name, params = declaration(head)
    segments = words(name)
    verb = next((w for w in segments if w in BY_WORD), '')
    action = BY_WORD.get(verb, -1)
    stem = segments[0] if segments and segments[0] != verb else ''
    keys = ('global',) + ((f'a:{action}', 'v:' + verb) if verb else ())
    if verb and stem:
        keys += ('s:' + stem + '/' + verb,)

    body_counts = Counter(body)
    calls = {w for w, following in zip(body, body[1:]) if following == '(' and
             IDENT.fullmatch(w) and w not in KEYWORDS}
    verbs = {BY_WORD[v] for call in calls for v in words(call) if v in BY_WORD}
    returns = [value.strip() for value in RETURN.findall(' '.join(body))]
    only_const = bool(returns) and all(r in NULL or r.replace(' ', '').lstrip('-+').isdigit() for r in returns)
    placeholder = bool(STUB_CODES.intersection(body_counts)) or any(
        token.startswith(('"', 'R"')) and PLACEHOLDER.search(token) for token in body)
    hard_fail = bool(HARD_FAIL & calls)
    params_last = [chunk.split()[-1] for chunk in ' '.join(params).split(',') if chunk.strip()]
    args = {p for p in params_last if IDENT.fullmatch(p) and p not in KEYWORDS}
    absent = sum(p not in body_counts for p in args)
    assignments = sum(body_counts[t] for t in MUTATIONS)
    flow = sum(body_counts[t] for t in ('if', 'switch', 'for', 'while', 'catch'))
    delegate = body_counts[';'] == 1 and len(calls) == 1
    no_effect = not (calls or assignments or body_counts['throw'])
    effects = (not body or body == ['}'], only_const, placeholder, hard_fail, bool(calls),
               bool(assignments), bool(flow), delegate, bool(args) and absent == len(args),
               bool(returns), *(i in verbs for i in range(len(ACTIONS))))
    shape = (len(text), len(head), len(body), body_counts[';'], len(calls), assignments, flow,
             len(returns), len(args), len(comments), int(bool(name)))
    other = set(segments) - set(BY_WORD) - {stem}
    role_mentions = len(other.intersection(w for t in body_counts if IDENT.fullmatch(t) for w in words(t)))
    roles = (int(bool(verb)), int(action in verbs), len(verbs), absent, role_mentions,
             int(no_effect and action in (1, 2, 3, 4, 5, 8, 9)),
             int(only_const and action in (0, 2, 7, 8)),
             int(placeholder and not flow), int(hard_fail and not flow),
             int(any(PLACEHOLDER.search(c) for c in comments)))
    return Evidence(keys, effects, shape, roles, bool(comments), digest(tokens))


def counts(observations):
    bank = defaultdict(lambda: np.zeros(1 + len(observations[0].effects), dtype=np.int64))
    for row in observations:
        value = np.asarray((1, *row.effects), dtype=np.int64)
        for key in row.keys:
            bank[key] += value
    return dict(bank)


def expected(row, bank, excluded):
    """Smooth the frequency of body effects for declaration families against broader groups."""
    empty = np.zeros(len(row.effects)+1, dtype=np.int64)
    prior = np.full(len(row.effects), .5)
    support = 0
    for key in row.keys:
        v = bank.get(key, empty) - excluded.get(key, empty)
        if v[0] >= 16:
            prior = (v[1:] + 24 * prior) / (v[0] + 24)
            support = int(v[0])
    surprise = -np.log(np.where(row.effects, prior, 1 - prior))
    return np.r_[np.log1p(support), surprise]


def features(obs, bank, exclusion=None):
    if exclusion is None:
        exclusion = [{}] * len(obs)
    shape = np.log1p(np.asarray([r.shape for r in obs], dtype=np.float32))
    values = np.asarray([(*r.effects, *r.roles) for r in obs], dtype=np.float32)
    expectation = np.asarray([expected(r, bank, drop) for r, drop in zip(obs, exclusion, strict=True)], dtype=np.float32)
    roles = np.column_stack((shape, values))
    return {'shape': shape, 'roles': roles, 'contract': np.column_stack((roles, expectation))}


def diagnostics(obs, y, base, groups, prob):
    missed = y & ~base
    neg = ~y & ~base
    strata = {
        'no_visible_comment': [not r.visible_comment for r in obs],
        'no_named_action': [not r.roles[0] for r in obs],
        'any_body_effect_modelled': [any(r.effects[4:8]) for r in obs],
        'constant_only_returns': [r.effects[1] for r in obs],
        'explicit_placeholder': [r.effects[2] for r in obs],
        'declared_side_effect_but_no_body_effect': [r.roles[5] for r in obs],
        'declared_value_but_only_constant': [r.roles[6] for r in obs],
        'visible_placeholder_phrase': [r.roles[-1] for r in obs],
    }
    edge = {}
    for project in np.unique(groups[missed]):
        m = groups == project
        edge[project] = {k: int((s[m & neg] >= s[m & missed].max()).sum()) for k, s in prob.items()}
    return dict(strata={k: dict(misses=int((np.asarray(v, bool) & missed).sum()),
                                negatives=int((np.asarray(v, bool) & neg).sum())) for k, v in strata.items()},
                min_fp_for_first_tp_by_project=edge)


def main():
    DEST.mkdir(exist_ok=True)
    started = perf_counter()
    texts, label, project, files = dataset()
    rows = split()
    gold = label[:, 1].astype(bool)
    fit = rows['fit']
    held = np.r_[rows['test_a'], rows['test_b']]
    assert not set(project[fit]) & set(project[held])
    path = lambda i: (project[i], files[i])
    assert not {path(i) for i in fit} & {path(i) for i in np.r_[rows['dev'], held]}
    all_obs = {}
    for k, inds in rows.items():
        all_obs[k] = [observe(texts[i]) for i in inds]
        print('parsed', k, len(inds), round(perf_counter() - started, 1), flush=True)
    holdout_code = {r.code_hash for k in PARTS for r in all_obs[k]}
    novel = np.array([r.code_hash not in holdout_code for r in all_obs['fit']])
    fit = fit[novel]
    obs = [r for r, keep in zip(all_obs.pop('fit'), novel, strict=True) if keep]
    base_fit = canonical_mat([texts[i] for i in fit])
    # Fit-specific expectations exclude the entire deterministic source-file fold.
    fold = [int.from_bytes(hashlib.blake2b(str(path(i)).encode(), digest_size=4).digest(), 'little') % 5 for i in fit]
    bank = counts(obs)
    drop = [counts([r for r, f in zip(obs, fold, strict=True) if f == j]) for j in range(5)]
    x = features(obs, bank, [drop[f] for f in fold])
    y = gold[fit]
    models = {}
    for name, value in x.items():
        clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=9, learning_rate=.055,
                                             min_samples_leaf=80, l2_regularization=12, random_state=52)
        clf.fit(value[~base_fit], y[~base_fit], sample_weight=np.where(y[~base_fit], 8., 1.))
        models[name] = clf
    weights = DEST / 'weights.joblib'
    joblib.dump({'models': models, 'body_effect_counts': bank}, weights, compress=3)

    summaries, thresholds = {}, {}
    for part in PARTS:
        ids = rows[part]
        obs = all_obs[part]
        y = gold[ids]
        visible = np.array([r.visible_comment for r in obs])
        base = canonical_mat([texts[i] for i in ids])
        x = features(obs, bank)
        score = {k: model.decision_function(x[k]) for k, model in models.items()}
        score['explicit_failure'] = np.array([float(r.effects[2] and not r.effects[6]) for r in obs])
        control = ROOT / 'round48/shape' / f'{part}_predictions.npz'
        with np.load(control) as z:
            assert all(np.array_equal(z[k], v) for k, v in {'ids': ids, 'y': y, 'base': base, 'comment': visible}.items())
            score['prior_layout'] = z['layout_paired_score']
        if part == 'dev':
            thresholds = {k: cutoff(y, base, v) for k, v in score.items()}
            write_json(DEST / 'policy.json', dict(cutoffs=thresholds, code_sha256=checksum(Path(__file__)),
                       train_manifest_sha256=checksum(ROOT / 'data/manifest.json'),
                       partition_sha256=checksum(ROOT / 'round13/partition.npz'),
                       weights_sha256=checksum(weights), fit_novel_rows=len(fit), excluded_code_copies=int((~novel).sum()),
                       fit_rows_used=int((~base_fit).sum()), fit_misses=int((~base_fit & gold[fit]).sum()),
                       file_group_rows=Counter(fold), action_groups=[sorted(v) for v in ACTIONS], runtime_input='Function'))
        report = table(part, ids, y, project[ids], score, visible, base, thresholds, dest=DEST)
        report['diagnostics'] = diagnostics(obs, y, base, project[ids], score)
        report['prior_sha256'] = checksum(control)
        summaries[part] = report
        print(part, report['baseline'], {k: (v['matched']['recovered'], v['frozen']['recovered'],
              v['frozen']['delta_fp']) for k, v in report['results'].items()}, flush=True)
    write_json(DEST / 'results.json', summaries)
    with (DEST / 'comparison.csv').open('w', newline='') as handle:
        out = csv.writer(handle)
        out.writerow(('partition','arm','residual_ap','same_fp_tp','same_fp_fp','frozen_tp','frozen_fp','frozen_f1'))
        for part, report in summaries.items():
            for name, row in report['results'].items():
                out.writerow((part, name, row['average_precision'], row['matched']['recovered'], row['matched']['fp'],
                              row['frozen']['recovered'], row['frozen']['delta_fp'], row['frozen']['f1']))


if __name__ == '__main__':
    main()
