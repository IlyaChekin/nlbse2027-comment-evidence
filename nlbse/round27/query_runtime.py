"""Select and combine a second set of requests in the original scanner batch."""
import numpy as np

from fuse_sparse import clip_logit


def add_requests(text, p, no, yes, probability, asked, threshold, option, model):
    p = p.astype(np.float32)
    cut = clip_logit(np.asarray(np.float32(threshold), dtype=np.float64))
    margin = clip_logit(p.astype(np.float64))-cut
    original = p > threshold
    pos_flip, neg_flip = (yes > threshold) != original, (no > threshold) != original
    possible = ~asked & (pos_flip | neg_flip)
    local = np.flatnonzero(possible)
    length = np.clip(np.asarray([len(text[i]) for i in local], dtype=np.float64), 48., 2048.)
    method = option['method']
    if method == 'distance':
        score = 1./(.15+abs(margin[local]))
    elif method == 'expected_flip':
        score = (1.-probability[local])*neg_flip[local]+probability[local]*pos_flip[local]
    else:
        variables = np.column_stack((margin[local], probability[local], no[local]-p[local],
                                     yes[local]-p[local], np.log1p(length), pos_flip[local], neg_flip[local]))
        score = model.predict(variables) if len(local) else np.empty(0)
    requested = np.zeros(len(text), dtype=bool)
    requested[local] = score/np.sqrt(length) > option['route']
    return requested, margin, cut


def update(bits, extra, scanned, raw, no, yes, margin, origin, option):
    active = extra[scanned]
    observed = raw[active]
    q = np.where(observed == 1, yes[extra], no[extra]).astype(np.float32)
    relative = clip_logit(q.astype(np.float64))-origin
    counterpart = np.where(observed >= 0, relative, margin[extra])
    weight = option['mixture']
    corrected = (1.-weight)*margin[extra]+weight*counterpart
    adjusted = bits.copy()
    adjusted[extra] = corrected > option['cut']
    return adjusted, corrected
