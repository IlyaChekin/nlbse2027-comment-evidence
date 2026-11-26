"""Isolate scanner changes caused by line breaks on paired official training strings."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'round50'))
from study50 import PARTS, LABELS, checksum, f, json_at, load, parts, write_json
from format_audit49 import from_archive
from components49 import SOURCE, key, signature
from first_hit32 import SCRIPT, scan
from project_split import TOKEN, fingerprint

DEST = HERE / 'causal'
PER_CELL = 2000
SHORT = 4096
SEED = 5409
LEXEME = re.compile(TOKEN.pattern.removesuffix(r'|\w+|[^\s\w]') +
        r'|(?:%:%:|>>=|<<=|->\*|\.\.\.|<=>|\+\+|--|->|&&|\|\||<<|>>|::|\.\*|'
        r'==|!=|<=|>=|[+*/%&|^\-]=|\#\#|<:|:>|<%|%>|%:)|\w+|[^\s\w]', TOKEN.flags)
FORBIDDEN = frozenset(('#', '##', '\\', '%:', '%:%:'))


def spans(text):
    return tuple(m for m in LEXEME.finditer(text))


def code(spans_):
    return tuple(s[0] for s in spans_ if s['comment'] is None)


def gaps(text, tokens):
    return tuple(text[a.end():b.start()] for a, b in zip(tokens, tokens[1:]))


def line_gap(tokens, original, i):
    start = original.rfind('\n', 0, tokens[i].start()) + 1
    part = original[start:tokens[i].start()]
    return '\n' + part[:len(part) - len(part.lstrip(' \t'))]


def render(original, full, token, edits):
    if not edits:
        return original
    out, previous = [], 0
    for i, gap in sorted(edits.items()):
        left, right = token[i].end(), token[i + 1].start()
        old = original[left:right]
        assert old.isspace() and gap.isspace()
        out.extend((original[previous:left], gap))
        previous = right
    draft = ''.join((*out, original[previous:]))
    return draft if tuple(s[0] for s in spans(draft)) == tuple(s[0] for s in full) else original


def aligned(a, b):
    """Locate only gaps with existing whitespace on both sides and identical code lexemes."""
    full_a, full_b = spans(a), spans(b)
    token_a, token_b = (tuple(s for s in x if s['comment'] is None) for x in (full_a, full_b))
    aa, bb = (tuple(s[0] for s in tokens) for tokens in (token_a, token_b))
    if aa != bb or any(s in FORBIDDEN for s in aa) or '\\\n' in a or '\\\n' in b:
        return None
    ga, gb = gaps(a, token_a), gaps(b, token_b)
    safe = {i: (x, y) for i, (x, y) in enumerate(zip(ga, gb, strict=True)) if x.isspace() and y.isspace()}
    differing = {i: y for i, (x, y) in safe.items() if ('\n' in x) != ('\n' in y)}
    # Preserve every character except whitespace whose neighboring tokens align.
    lines = {i: line_gap(token_a, a, i) if '\n' in y else ' ' for i, y in differing.items()}
    whitespace = {i: y for i, (x, y) in safe.items() if x != y}
    return full_a, token_a, safe, lines, whitespace


def prepare():
    DEST.mkdir(exist_ok=True)
    stored = LABELS / 'labels.npz'
    proof = json_at(stored.with_name('receipt.json'))
    assert proof['archive_sha256'] == checksum(SOURCE) and proof['predictions_sha256'] == checksum(stored)
    with np.load(stored) as z:
        matched, component, gold = (z[k] for k in ('selected', 'component', 'truth'))
    text, labels, project, paths, _ = f.inputs()
    assert np.array_equal(labels[:, 0], gold)
    files = defaultdict(set)
    slot, cohorts, sources = {}, {}, {}
    for part, rotation in enumerate(PARTS):
        data, _, sources[rotation] = load(rotation)
        cal, held = parts(data)
        fit = data['oof']
        hold = np.unique(np.r_[data['dev']['ids'], data.get('test', data['dev'])['ids']])
        hold_paths = {key(project[i], paths[i]) for i in hold if paths[i]}
        hold_code = {fingerprint(text[i]) for i in hold}
        assert not np.intersect1d(fit['ids'], hold).size
        if rotation != 'official':
            assert not set(project[fit['ids']]) & set(project[held['ids']])
        available = matched[fit['ids']] & fit['old'] & (fit['flag'] >= 0) & (fit['length'] <= SHORT)
        rng = np.random.default_rng(SEED + part)
        eligible = np.ones(len(fit['ids']), bool)
        for j in np.flatnonzero(available):
            i = fit['ids'][j]
            eligible[j] = bool(paths[i]) and key(project[i], paths[i]) not in hold_paths and fingerprint(text[i]) not in hold_code
        old = fit['flag'] == 1
        w = (component[fit['ids']] & 1) > 0
        sample = np.zeros(len(old), bool)
        for a in (False, True):
            for b in (False, True):
                pos = np.flatnonzero(available & eligible & (old == a) & (w == b))
                sample[rng.choice(pos, min(len(pos), PER_CELL), replace=False)] = True
        for kind, d, active in [('fit', fit, sample), *[(name, z, z['old'] & (z['flag'] >= 0) &
                         matched[z['ids']] & (z['length'] <= SHORT)) for name, z in (('cal', cal), ('held', held))]]:
            cohort = []
            for i, flag in zip(d['ids'][active], d['flag'][active], strict=True):
                row = int(i)
                loc = key(project[row], paths[row])
                sha = signature(text[row])
                files[loc].add(sha)
                assert slot.get(row, (loc, sha, int(flag))) == (loc, sha, int(flag))
                slot[row] = (loc, sha, int(flag))
                cohort.append(row)
            cohorts[rotation + '/' + kind] = cohort
        print('sample', rotation, [len(cohorts[rotation + '/' + k]) for k in ('fit', 'cal', 'held')], flush=True)
    original = from_archive(files)
    artifact = DEST / 'pairs.jsonl.gz'
    seen = 0
    with gzip.open(artifact, 'wt', encoding='utf-8') as output:
        for row, (loc, sig, bit) in sorted(slot.items()):
            candidates = original[loc, sig]
            if len(candidates) != 1:
                continue
            ((before, label),) = candidates.items()
            assert label == component[row]
            if any(0xdc80 <= ord(c) <= 0xdcff for c in before):
                continue
            if aligned(text[row], before) is None:
                continue
            seen += 1
            print(json.dumps(dict(row=row, original=before, archived_w=bool(label & 1), baseline=bit),
                             ensure_ascii=False), file=output)
    metadata = dict(cohorts=cohorts, sources=sources, unique_requests=len(slot), aligned=seen,
                    raw_sha256=checksum(artifact), annotation_sha256=checksum(stored), archive_sha256=proof['archive_sha256'],
                    code_sha256=checksum(Path(__file__)))
    write_json(DEST / 'pairs.json', metadata)
    print('paired', seen, 'of', len(slot), flush=True)


def stored_pairs():
    p = DEST / 'pairs.jsonl.gz'
    metadata = json_at(DEST / 'pairs.json')
    assert metadata['raw_sha256'] == checksum(p)
    with gzip.open(p, 'rt', encoding='utf-8') as raw:
        entries = {i['row']: i for i in map(json.loads, raw)}
    assert metadata['aligned'] == len(entries)
    return entries, metadata


def changes(out, data, variants):
    keep = np.asarray(out, dtype=np.int64)
    baseline = data['baseline'][keep, 0]
    expected = data['gold'][keep]
    result = {}
    for name in variants:
        q = data[name][keep, 0]
        valid = (q >= 0) & (baseline >= 0)
        same = (q == expected) & valid
        benefit = same & (baseline != expected)
        harm = (q != expected) & (baseline == expected) & valid
        result[name] = dict(rows=len(q), eligible=int(valid.sum()), switched=int((valid & (q != baseline)).sum()),
                            repaired=int(benefit.sum()), hurt=int(harm.sum()), invalid=int((q < 0).sum()),
                            raw_w=f.comparison(expected[valid], baseline[valid] == 1, q[valid] == 1))
    return result


def audit():
    all_rows, selection = stored_pairs()
    text = f.inputs()[0]
    ids = np.array(list(all_rows), dtype=np.int64)
    old = [text[i] for i in ids]
    current, total_scan = scan(old, True)
    assert np.array_equal(current[:, 0], [all_rows[i]['baseline'] for i in ids])
    result = dict(ids=ids, baseline=current, gold=np.array([all_rows[i]['archived_w'] for i in ids]),
                  edits=np.zeros((len(ids), 2), dtype=np.int32))
    variants = ('lines', 'gaps', 'original')
    catalog = {v: {} for v in variants}
    meta = Counter()
    for j, i in enumerate(ids):
        before = all_rows[i]['original']
        full, token, safe, line, gap = aligned(old[j], before)
        result['edits'][j] = len(line), len(gap)
        meta['safe_gaps'] += len(safe)
        for v, t in zip(variants, (render(old[j], full, token, line), render(old[j], full, token, gap), before), strict=True):
            if t != old[j]:
                catalog[v][j] = t
    times = {'baseline': total_scan}
    for v in variants:
        q = current.copy()
        if catalog[v]:
            at = np.fromiter(catalog[v], np.int64)
            q[at], times[v] = scan(list(catalog[v].values()), True)
        result[v] = q
    plan = {name: changes([j for j, i in enumerate(ids) if i in set_], result, variants)
            for name, set_ in {k: set(v) for k, v in selection['cohorts'].items()}.items()}
    path = DEST / 'oracle.npz'
    np.savez_compressed(path, **result)
    with (DEST / 'oracle.csv').open('w', newline='') as stream:
        table = csv.writer(stream)
        table.writerow(('rotation', 'split', 'variant', 'rows', 'repaired', 'hurt', 'invalid', 'w_delta_tp', 'w_delta_fp'))
        for panel, outcomes in plan.items():
            for name, stat in outcomes.items():
                table.writerow((*panel.split('/'), name, *(stat[k] for k in ('rows', 'repaired','hurt','invalid')),
                                stat['raw_w']['delta_tp'], stat['raw_w']['delta_fp']))
    receipt = dict(pairs_sha256=checksum(DEST / 'pairs.json'), scanner_sha256=checksum(SCRIPT),
                   oracle_sha256=checksum(path), code_sha256=checksum(Path(__file__)), panels=plan,
                   replacement_rows={v: len(s) for v, s in catalog.items()}, scanner_seconds=times, coverage=meta)
    write_json(DEST / 'oracle.json', receipt)
    print({role: {v: (p['repaired'], p['hurt']) for v, p in table.items()} for role, table in plan.items()}, flush=True)


if __name__ == '__main__':
    stage = sys.argv[1]
    assert stage in ('prepare', 'oracle')
    {'prepare': prepare, 'oracle': audit}[stage]()
