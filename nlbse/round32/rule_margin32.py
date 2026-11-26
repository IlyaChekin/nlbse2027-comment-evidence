"""Apply the frozen first-rule adjustment using scores already in memory."""
import numpy as np

from positive_runtime import scope


def adjust(texts, state, threshold, original, corrected, positive_score, positive_cut, counts, settings):
    relevant, margin, _ = scope(texts,state,threshold,None)
    margin[corrected] = positive_score-positive_cut
    relevant &= abs(margin)<settings['width']
    if not relevant.any():
        return original
    selected = counts[relevant]
    code = selected[:,0,None]==np.arange(1,21)[None,:]
    score = margin[relevant].clip(-3.,3.)
    count = np.log1p(selected[:,1].clip(0,64))
    design = np.column_stack((np.ones(len(score)), score, code, code*score[:,None], code*count[:,None]))
    candidate = margin[relevant]+settings['strength']*(design@np.asarray(settings['weights']))
    result = original.copy()
    result[relevant] = candidate>settings['cut']
    return result
