"""Probe individual token gaps and fit a before-scanner edit model."""
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from causal54 import (DEST as CAUSAL, FORBIDDEN, HERE, PARTS, ROOT, SHORT, SCRIPT, aligned, checksum,
                      f, gaps, json_at, line_gap, load, parts, render, scan, spans, stored_pairs, write_json)
from styles49 import DEST as STYLES
from source41 import crop

DEST = HERE / 'edit'
SEED = 5441
OP_GROUPS = (
    {'=', '+=', '-=', '*=', '/=', '%=', '|=', '&=', '^=', '<<=', '>>='},
    {'==', '!=', '<', '>', '<=', '>='},
    {'&&', '||', '?', ':'},
    {'+', '-', '*', '/', '%', '++', '--'},
    {'&', '|', '^', '~', '<<', '>>'},
)
TYPES = ('(', ')', '{', '}', '[', ']', ';', ',', 'return', 'if', 'else', 'while',
         'for', 'switch', 'case', 'new', 'delete', 'const', 'operator', '.', '->', '::')
TYPE = {s: i + len(OP_GROUPS) + 1 for i, s in enumerate(TYPES)}
for group, ops in enumerate(OP_GROUPS, 1):
    for symbol in ops:
        TYPE[symbol] = group
CONTEXT_WIDTH = 3
TOK_IDENT = re.compile(r'[a-zA-Z_]\w*')
TOK_NUM = re.compile(r'[0-9]')
MODEL_ARGS = dict(max_iter=74, max_leaf_nodes=15, min_samples_leaf=45, learning_rate=.071,
                  max_bins=64, l2_regularization=7., early_stopping=False, random_state=SEED)


@dataclass
class Source:
    full: tuple
    token: tuple
    gap: tuple
    sites: np.ndarray
    x: np.ndarray


def descriptor(text):
    full = spans(text)
    token = tuple(s for s in full if s['comment'] is None)
    terms = tuple(s[0] for s in token)
    if len(terms) < 6 or len(text) > SHORT or any(s in FORBIDDEN for s in terms) or '\\\n' in text:
        return None
    gap = gaps(text, token)
    selected = [i for i, s in enumerate(gap) if s.isspace()]
    if not selected:
        return None
    sites = np.asarray(selected, dtype=np.int32)
    ids = np.asarray([TYPE.get(t, 31 if TOK_IDENT.fullmatch(t) else 32 if TOK_NUM.match(t) else 0)
                      for t in terms], dtype=np.int8)
    lines = np.fromiter((0, *(m.end() for m in re.finditer('\n', text)), len(text) + 1), np.int32)
    positions = np.fromiter((s.start() for s in token), np.int32)
    lineno = np.searchsorted(lines, positions, side='right') - 1
    l, r = lineno[sites], lineno[sites + 1]
    start, end = lines[l], lines[r + 1]
    break_here = l != r
    count = np.zeros((len(lines), len(OP_GROUPS)), dtype=np.int32)
    for k in range(1, len(OP_GROUPS) + 1):
        p = ids == k
        count[:, k - 1] = np.bincount(lineno[p], minlength=len(lines))
    prefix = np.vstack((np.zeros(len(OP_GROUPS), np.int32), np.cumsum(count, axis=0)))
    # Count operator groups separately on either side of the candidate boundary.
    by_token = np.column_stack([np.cumsum(ids == k) for k in range(1, len(OP_GROUPS) + 1)])
    by_token = np.vstack((np.zeros(len(OP_GROUPS), np.int32), by_token))
    a = np.searchsorted(lineno, l, side='left')
    b = np.searchsorted(lineno, r, side='right')
    left_ops, right_ops = by_token[sites + 1] - by_token[a], by_token[b] - by_token[sites + 1]
    padded = np.pad(ids, CONTEXT_WIDTH)
    around = np.stack([padded[sites + CONTEXT_WIDTH + offset] for offset in range(-2, 4)], axis=1)
    depth = np.asarray(list(accumulate((int(t == '(') - int(t == ')') for t in terms))))
    blocks = np.asarray(list(accumulate((int(t == '{') - int(t == '}') for t in terms))))
    widths = np.asarray([len(gap[i]) for i in sites])
    tabs = np.asarray([gap[i].count('\t') for i in sites])
    x = np.column_stack((break_here, widths, tabs, depth[sites], blocks[sites], positions[sites] - start,
                         end - positions[sites + 1], (positions[sites] - start) + end - positions[sites + 1],
                         sites / len(token), l / len(lines), np.full(len(sites), len(terms)),
                         r - l, around, left_ops, right_ops, prefix[r + 1] - prefix[l])).astype(np.float32)
    return Source(full, token, gap, sites, x)


def edit_one(text, src, j):
    i = int(src.sites[j])
    gap = ' ' if '\n' in src.gap[i] else line_gap(src.token, text, i)
    return render(text, src.full, src.token, {i: gap})


def gen_design():
    DEST.mkdir(exist_ok=True)
    pair, sources = stored_pairs()
    oracle = CAUSAL / 'oracle.npz'
    assert json_at(CAUSAL / 'oracle.json')['oracle_sha256'] == checksum(oracle)
    with np.load(oracle) as z:
        baseline = dict(zip(z['ids'], z['baseline'][:, 0], strict=True))
    texts = f.inputs()[0]
    rng = np.random.default_rng(SEED)
    fit = {rotation: set(sources['cohorts'][rotation+'/fit']) & pair.keys() for rotation in PARTS}
    ids = sorted(set.union(*fit.values()))
    all_x, all_row, all_j, target = [], [], [], []
    pending, test_links = [], []
    # Test source-edited gaps and equally many unchanged controls. For agreement rows,
    # test a fixed sample to learn when a plausible edit would introduce an error.
    good_controls = set()
    for form in fit.values():
        clean = [i for i in sorted(form) if baseline[i] == pair[i]['archived_w']]
        good_controls.update(rng.choice(clean, size=min(1800, len(clean)), replace=False).tolist())
    for row in ids:
        src = descriptor(texts[row])
        if src is None:
            continue
        fitted = aligned(texts[row], pair[row]['original'])
        assert fitted is not None
        _, _, safe, true_edit, _ = fitted
        xx = np.array([j for j,v in enumerate(src.sites) if v in true_edit], dtype=np.int32)
        nn = np.array([j for j,v in enumerate(src.sites) if v in safe and v not in true_edit], dtype=np.int32)
        positives = rng.choice(xx, min(36, len(xx)), replace=False)
        negatives = rng.choice(nn, min(42, len(nn)), replace=False)
        start = len(all_row)
        picked = np.r_[positives, negatives]
        if not len(picked):
            continue
        all_x.append(src.x[picked])
        all_row.extend([row] * len(picked))
        all_j.extend(src.sites[picked].tolist())
        target.extend((j in true_edit for j in src.sites[picked]))
        if baseline[row] != pair[row]['archived_w'] or row in good_controls:
            limit = 10 if baseline[row] != pair[row]['archived_w'] else 3
            a = rng.choice(len(positives), min(limit, len(positives)), replace=False)
            b = rng.choice(len(negatives), min(limit, len(negatives)), replace=False) + len(positives)
            for at in np.r_[a, b]:
                pushed = edit_one(texts[row], src, int(picked[at]))
                if pushed != texts[row]:
                    test_links.append(start + int(at))
                    pending.append(pushed)
    if pending:
        q, seconds = scan(pending, True)
    else:
        raise ValueError('No edits to probe')
    rows = np.asarray(all_row, dtype=np.int32)
    probe = np.asarray(test_links, dtype=np.int64)
    actual = np.full(len(rows), -2, np.int8)
    history = np.array([pair[i]['archived_w'] for i in rows])
    initial = np.fromiter((baseline[i] for i in rows), dtype=np.int8)
    actual[probe] = q[:, 0]
    file = DEST / 'design.npz'
    np.savez_compressed(file, row=rows, gap=all_j, x=np.concatenate(all_x), source_edit=target,
                        initial=initial, historical=history, single=actual, probed=probe, state=q)
    card = dict(design_sha256=checksum(file), pairs_sha256=checksum(CAUSAL / 'pairs.json'),
                oracle_sha256=checksum(oracle), code_sha256=checksum(Path(__file__)), scanner_sha256=checksum(SCRIPT),
                lined_examples=len(rows), variants=len(probe), seconds=seconds,
                recoverable_single=int(((actual >= 0) & (initial != history) & (actual == history)).sum()),
                introducing_error=int(((actual >= 0) & (initial == history) & (actual != history)).sum()),
                fit={k: len(y) for k, y in fit.items()})
    write_json(DEST / 'design.json', card)
    print(card, flush=True)


def fit(rotation, pairs):
    path = DEST / 'design.npz'
    proof = json_at(DEST / 'design.json')
    assert checksum(path) == proof['design_sha256']
    with np.load(path) as z:
        d = dict(z)
    x = d['x']
    fit_rows = np.asarray(pairs['cohorts'][rotation+'/fit'])
    use = np.isin(d['row'], fit_rows)
    # Never use calibration or held-out code or archived targets to fit a fold.
    assert not np.intersect1d(d['row'][use], np.r_[pairs['cohorts'][rotation+'/cal'], pairs['cohorts'][rotation+'/held']]).size
    fitted = {}
    side = d['source_edit']
    prior = np.where(side[use], 2., 1.)
    fitted['source'] = HistGradientBoostingClassifier(**MODEL_ARGS).fit(x[use], side[use], sample_weight=prior)
    tested = use & (d['single'] >= 0)
    flip = d['single'][tested] != d['initial'][tested]
    improvement = flip & (d['single'][tested] == d['historical'][tested])
    target = np.select((improvement, flip), (1, 2), default=0)
    assert np.array_equal(np.unique(target), [0, 1, 2])
    fitted['effect'] = HistGradientBoostingClassifier(**MODEL_ARGS).fit(x[tested], target,
                        sample_weight=np.where(flip, 10., 1.))
    return fitted, dict(pairs=len(np.unique(d['row'][use])), examples=int(use.sum()), gap_changes=int(side[use].sum()),
                        probed=int(tested.sum()), helpful=int(improvement.sum()), harmful=int((target == 2).sum()))


MODES = ('source', 'effect', 'joint')
LIMITS = (1, 3)
FRACTIONS = (.035, .15, .40)


def estimates(texts, bank):
    rows = [descriptor(t) for t in texts]
    pieces = [(i, d) for i, d in enumerate(rows) if d is not None]
    if not pieces:
        return rows, {k: {} for k in MODES}
    x = np.concatenate([d.x for _, d in pieces])
    p = bank['source'].predict_proba(x)[:, 1]
    effect = bank['effect'].predict_proba(x)
    priority = {'source': p, 'effect': effect[:, 1] - effect[:, 2],
                'joint': p * (effect[:, 1] - effect[:, 2])}
    scores = {k: {} for k in MODES}
    j = 0
    for i, d in pieces:
        n = len(d.sites)
        for k, values in priority.items():
            scores[k][i] = values[j:j+n]
        j += n
    return rows, scores


def alterations(text, data, value, count):
    by_score = np.argsort(-value, kind='stable')[:count]
    edits = {}
    for placed in by_score:
        i = int(data.sites[placed])
        edits[i] = ' ' if '\n' in data.gap[i] else line_gap(data.token, text, i)
    return render(text, data.full, data.token, edits)


def experiment(rotation):
    _, paired = stored_pairs()
    bank, counts = fit(rotation, paired)
    root = DEST / rotation
    root.mkdir(exist_ok=True, parents=True)
    w = root / 'models.joblib'
    joblib.dump(bank, w, compress=3)
    view, _, stamp = load(rotation)
    frames = parts(view)
    texts = f.inputs()[0]
    # W and packed labels are loaded only after fitting.
    labels = ROOT / 'round49/components/labels.npz'
    with np.load(labels) as z:
        gold, known = (z['component'] & 1) > 0, z['selected']
    style_proof = json_at(STYLES / 'receipt.json')
    assert style_proof['predictions_sha256'] == checksum(STYLES / 'predictions.npz')
    with np.load(STYLES / 'predictions.npz') as z:
        styles = dict(z)
    exceptions = {'source': None, 'effect': 0., 'joint': 0.}
    result, audit, config, threshold = {}, {}, {}, {}
    for phase, trial in zip(('cal', 'held'), frames, strict=True):
        phase_start = perf_counter()
        frame = crop(trial, trial['old'] & (trial['flag'] >= 0))
        ids = frame['ids']
        reading = [texts[i] for i in ids]
        source, scores = estimates(reading, bank)
        old = frame['flag'] == 1
        desired = gold[ids]
        labeled = known[ids]
        audit.update({phase+'_ids': ids, phase+'_expected':desired, phase+'_known':labeled, phase+'_old':old})
        out = {}
        for name, ranks in scores.items():
            for count in LIMITS:
                built, active = [], []
                for k, row in ranks.items():
                    value = float(row.max())
                    if exceptions[name] is None or value > exceptions[name]:
                        active.append(k)
                        built.append(alterations(reading[k], source[k], row, count))
                flags, elapsed = scan(built, True) if built else (np.empty((0, 3),np.int32), 0.)
                actual = frame['flag'].copy()
                actual[np.asarray(active, dtype=np.int64)] = flags[:, 0]
                exposure = np.full(len(ids), -np.inf)
                exposure[active] = [ranks[k].max() for k in active]
                if count == 1:
                    out[name+'_candidates'] = int(len(active))
                for fraction in FRACTIONS:
                    key = f'{name}/{count}/{fraction:g}'
                    if phase == 'cal':
                        finite = exposure[np.isfinite(exposure)]
                        allowance = min(len(finite), max(1,int(fraction*len(ids))))
                        threshold[key] = (float(np.sort(finite)[-allowance]) if allowance else None)
                    touched = np.isfinite(exposure) & (exposure >= threshold[key]) if threshold[key] is not None else np.zeros(len(ids),bool)
                    candidate = np.where(touched, actual, frame['flag'])
                    estimated = np.where(candidate >= 0, candidate == 1, old)
                    assert np.array_equal(estimated[~touched], old[~touched])
                    audit[phase+'_'+key] = candidate
                    audit[phase+'_'+key+'_touched'] = touched
                    common = labeled & np.isin(ids, styles['ids'])
                    pos = np.searchsorted(styles['ids'], ids[common])
                    fixed_styles = {k:f.comparison(desired[common],old[common],styles[k][pos,0]==1)
                                    for k in ('LLVM','GNU','Chromium','WebKit','Microsoft')}
                    out[key] = dict(W=f.comparison(desired[labeled], old[labeled], estimated[labeled]),
                                    common_style_rows=int(common.sum()),
                                    common_W=f.comparison(desired[common],old[common],estimated[common]),
                                    style_controls=fixed_styles, scanner_seconds=elapsed,
                                    touched=int(touched.sum()), unavailable=int((candidate < 0).sum()),
                                    cases_changed=int((estimated != old).sum()))
        result[phase] = out
        config[phase] = dict(rows=len(ids), known=int(labeled.sum()), seconds=perf_counter()-phase_start)
        print(rotation,phase,config[phase],{k:(v['W']['delta_tp'],v['W']['delta_fp'],v['unavailable'])
                  for k,v in out.items() if isinstance(v,dict)},flush=True)
    predictions = root / 'predictions.npz'
    np.savez_compressed(predictions, **audit)
    write_json(root / 'results.json', dict(rotation=rotation, proof=dict(matrix_sha256=stamp, models_sha256=checksum(w),
               predictions_sha256=checksum(predictions), code_sha256=checksum(Path(__file__)),
               design_sha256=checksum(DEST/'design.json'), styles_sha256=checksum(STYLES/'receipt.json')),
               training=counts, phases=config, policy=threshold, comparison=result))


if __name__ == '__main__':
    stage = sys.argv[1]
    assert stage in ('probe', *PARTS)
    if stage == 'probe':
        gen_design()
    else:
        experiment(stage)
