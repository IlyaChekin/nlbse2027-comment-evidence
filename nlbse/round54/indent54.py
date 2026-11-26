"""Learn changes in the relative indentation of adjacent lines."""
from pathlib import Path
import re
import sys
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from causal54 import HERE, PARTS, ROOT, aligned, checksum, f, json_at, load, parts, render, scan, stored_pairs, spans, write_json
from effects54 import descriptor
from source41 import crop

DEST = HERE / 'indent'
SEED = 5421
LIMITS = (1, 3, 12)
CUTS = (.035, .2)
PAD = re.compile(r'[ \t]*')


def indent(text, at):
    start = text.rfind('\n', 0, at) + 1
    return PAD.match(text, start).end() - start


def observations(text, src):
    chosen = np.array([j for j,i in enumerate(src.sites) if '\n' in src.gap[i] and
                      src.gap[i].count('\n') == 1], dtype=np.int32)
    sites = src.sites[chosen]
    if not len(sites):
        return None
    left = np.array([indent(text, src.token[i].start()) for i in sites], dtype=np.int32)
    right = np.array([indent(text, src.token[i + 1].start()) for i in sites], dtype=np.int32)
    difference = right - left
    design = np.column_stack((src.x[chosen], left, right, difference,
                              np.r_[0, difference[:-1]], np.r_[difference[1:], 0]))
    return sites, left, difference, design


def fit(rotation, texts, pairs, fit_roster):
    features, labels, before = [], [], []
    rng = np.random.default_rng(SEED)
    for i in fit_roster:
        if i not in pairs:
            continue
        src = descriptor(texts[i])
        if src is None:
            continue
        sample = observations(texts[i], src)
        if sample is None:
            continue
        sites, _, old, x = sample
        original = pairs[i]['original']
        full, _, safe, _, _ = aligned(texts[i], original)
        prior = tuple(s for s in spans(original) if s['comment'] is None)
        choose = [j for j, site in enumerate(sites) if site in safe and safe[site][1].count('\n') == 1]
        if not choose:
            continue
        choose = np.asarray(choose, dtype=np.int32)
        aligned_sites = sites[choose]
        target = np.array([indent(original, prior[k+1].start()) - indent(original, prior[k].start())
                           for k in aligned_sites])
        options = np.flatnonzero(np.sign(old[choose]) != np.sign(target))
        remain = np.flatnonzero(np.sign(old[choose]) == np.sign(target))
        at = np.r_[options, rng.choice(remain, min(36, len(remain)), replace=False)]
        features.append(x[choose[at]])
        labels.extend((np.sign(target[at]) + 1).tolist())
        before.extend((np.sign(old[choose[at]]) + 1).tolist())
    old = np.asarray(before)
    new = np.asarray(labels)
    x = np.concatenate(features).astype(np.float32)
    tree = HistGradientBoostingClassifier(max_iter=72, max_leaf_nodes=15, min_samples_leaf=45,
              max_bins=64, l2_regularization=7., early_stopping=False, random_state=SEED)
    tree.fit(x, new, sample_weight=np.where(old != new, 2.25, 1.))
    assert np.array_equal(tree.classes_, [0,1,2])
    return tree, dict(gaps=len(new), changed_relation=int((old != new).sum()), columns=x.shape[1])


def propose(texts, model):
    read, samples = [], []
    for j, text in enumerate(texts):
        src = descriptor(text)
        if src is None:
            continue
        observed = observations(text, src)
        if observed is None:
            continue
        sites, left, delta, x = observed
        read.append((j, src, sites, left, delta))
        samples.append(x)
    if not samples:
        return {}
    probability = model.predict_proba(np.concatenate(samples).astype(np.float32))
    options = {}
    start = 0
    for j, src, sites, left, delta in read:
        stop = start + len(sites)
        p = probability[start:stop]
        a = np.arange(len(p))
        chosen = p.argmax(1)
        q = p[a, chosen] - p[a, np.sign(delta) + 1]
        sign = chosen - 1
        valid = (q > 0.) & ((left > 0) | (sign >= 0))
        if valid.any():
            # The old difference determines the step where available; use two
            # columns for previously aligned lines. Keep the right-line pad positive.
            right = np.maximum(0, left[valid] + sign[valid] * np.maximum(2, abs(delta[valid])))
            options[j] = (src, sites[valid], right, q[valid])
        start = stop
    return options


def rewrite(text, packed, maximum):
    src, site, width, strength = packed
    take = np.argsort(-strength, kind='stable')[:maximum]
    replacements = {int(k): '\n' + ' ' * int(n) for k,n in zip(site[take],width[take],strict=True)}
    return render(text, src.full, src.token, replacements)


def experiment(rotation):
    dest = DEST / rotation
    dest.mkdir(exist_ok=True, parents=True)
    pairs, metadata = stored_pairs()
    texts = f.inputs()[0]
    src, _, stamp = load(rotation)
    cal, held = parts(src)
    fit_rows = metadata['cohorts'][rotation+'/fit']
    assert not np.intersect1d(fit_rows, np.r_[cal['ids'], held['ids']]).size
    model, fit_info = fit(rotation, texts, pairs, fit_rows)
    store = dest/'model.joblib'
    joblib.dump(model, store, compress=3)
    with np.load(ROOT/'round49/components/labels.npz') as label:
        gold, known = (label['component'] & 1)>0, label['selected']
    threshold, report, output = {}, {}, {}
    for phase, raw in (('cal',cal),('held',held)):
        start=perf_counter()
        frame = crop(raw,raw['old']&(raw['flag']>=0))
        ids = frame['ids']
        x = [texts[i] for i in ids]
        old = frame['flag']
        got = propose(x, model)
        gate = np.full(len(x), -np.inf)
        positions = np.fromiter(got, np.int64)
        gate[positions] = [got[j][3].max() for j in positions]
        if phase == 'cal':
            ranks = np.sort(gate[np.isfinite(gate)])
            for cut in CUTS:
                size = min(len(ranks), max(1, int(cut*len(x))))
                threshold[str(cut)] = float(ranks[-size]) if size else None
        output.update({phase+'_ids': ids, phase+'_expected':gold[ids], phase+'_known':known[ids], phase+'_old':old==1})
        where = known[ids]
        result = {}
        for maximum in LIMITS:
            values, seconds = scan([rewrite(x[j], got[j], maximum) for j in positions], True)
            shift = old.copy()
            shift[positions] = values[:,0]
            for fraction, edge in threshold.items():
                applied = np.isfinite(gate) & (gate >= edge) if edge is not None else np.zeros(len(x),bool)
                forecast = np.where(applied, shift, old)
                pred = np.where(forecast >= 0, forecast, old)
                name = f'indent/{maximum}/{fraction}'
                output[phase+'_'+name] = forecast
                output[phase+'_'+name+'_touched'] = applied
                result[name] = dict(W=f.comparison(gold[ids][where], (old==1)[where], (pred==1)[where]),
                                    touched=int(applied.sum()), unavailable=int((forecast<0).sum()),
                                    cases_changed=int((pred!=old).sum()), scanner_seconds=seconds)
        report[phase] = result
        print(rotation,phase,round(perf_counter()-start,2),
              {k:(v['W']['delta_tp'],v['W']['delta_fp'],v['touched']) for k,v in result.items()},flush=True)
    predictions = dest/'predictions.npz'
    np.savez_compressed(predictions, **output)
    write_json(dest/'results.json',dict(rotation=rotation, fit=fit_info, threshold=threshold, comparison=report,
               matrix_sha256=stamp, model_sha256=checksum(store), code_sha256=checksum(Path(__file__)),
               pairs_sha256=checksum(HERE/'causal/pairs.json'), predictions_sha256=checksum(predictions)))


if __name__ == '__main__':
    name = sys.argv[1]
    assert name in PARTS
    experiment(name)
