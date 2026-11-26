"""Rank a subset of functions with an observed positive auxiliary outcome."""
import time

import numpy as np

from conditional import features
from fuse_sparse import clip_logit
from sparse_model import vectorize


def scope(texts, state, cut, width):
    # Cast the cutoff as in the original float32 comparison before taking logs.
    cut = clip_logit(np.asarray(np.float32(cut), dtype=np.float64))
    score = clip_logit(state['p'].astype(np.float64)) - cut
    mask = state['known'] & state['flag']
    if width is not None:
        mask &= np.abs(score) < width
    return mask, score, cut


def signals(values, local, sparse):
    compact = features(values, 'positive').astype(np.float64)
    if sparse is None:
        return compact
    return np.column_stack((compact, np.clip(sparse, -18., 18.)))


def layout(x, local):
    size = np.fromiter((np.log1p(len(t)) for t in local), dtype=np.float64)
    shape = np.column_stack((np.clip(size-6., -4., 5.),
                             np.clip(size-6., -4., 5.)*np.clip(x[:, 1], -4., 4.)))
    return np.column_stack((x, shape))


def directional(x):
    q = np.clip(x, -8., 8.)
    return np.column_stack((x, np.abs(q[:, 1]-q[:, 2]), np.minimum(q[:, 1], q[:, 2]),
                            np.maximum(q[:, 1], q[:, 2])))


def proposed(x, local, score, cut, mode, head):
    if mode == 'compact':
        return head['joint'].decision_function(x) - cut
    if mode == 'shape':
        features = layout(x, local)
    elif mode == 'contrast':
        features = directional(x)
    else:
        features = x
    gain = head['joint'].decision_function(features)
    if 'anchor' in head:
        return score + gain - head['anchor'].decision_function(features[:, :2])
    return gain - cut


def ranking(score, pred, mix, direction):
    delta = pred-score
    if direction == 'veto':
        delta = np.where(score > 0., np.minimum(delta, 0.), 0.)
    return score + mix*delta


def apply(texts, values, state, cut, original, variant, package):
    start = time.perf_counter()
    mask, score, origin = scope(texts, state, cut, variant['width'])
    indices = np.flatnonzero(mask)
    local = [texts[i] for i in indices]
    nnz = 0
    if variant['mode'] != 'compact' and local:
        x = vectorize(local, variant['sparse'])
        feature = package['expert'].decision_function(x)
        nnz = 2*x.nnz
    else:
        feature = None
    result = original.copy()
    out = np.empty(len(local), dtype=np.float64)
    if local:
        x = signals(values[mask], local, feature)
        pp = proposed(x, local, score[mask], origin, variant['mode'], package['model'])
        out = ranking(score[mask], pp, variant['mix'], variant['direction'])
        result[mask] = out > variant['cut']
    return result, mask, out, dict(rows=len(local), nnz=nnz, seconds=time.perf_counter()-start)
