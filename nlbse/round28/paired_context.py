"""Replay the signed Round 27 comparator before changing its negative-cohort decisions."""
from pathlib import Path
import sys

import joblib
import numpy as np

from pairs import HERE, ROOT, f, checksum, json_at
from disagreement import encoded_context
from length_eval import REFERENCE, R24, recipe, match, calibrate, load_sparse

R27 = ROOT/'round27'
sys.path.insert(0,str(R27))
from positive_runtime import scope, signals, proposed, ranking
from positive import strict_cut


def context(rotation):
    d, t, fit, cal, review, result = encoded_context(rotation)
    original = t
    t = d if t is None else t
    frozen = json_at(R27/'public/policy.json')
    winner = frozen['selected']
    key, kind = winner.rsplit(':',1)
    study_folder = R27/'joint'/rotation
    study = json_at(study_folder/'results.json')
    if checksum(study_folder/'results.json') != frozen['table_sha256'][rotation]:
        raise ValueError('Round 27 calibration table changed')
    row = study['options'][key]
    config = frozen['supported'][winner]
    pos = config['online'] | {'cut':row['cuts'][kind]['cut']}
    file = study_folder/'weights.joblib'
    if checksum(file) != study['weights_sha256']:
        raise ValueError('Frozen weights changed')
    w = joblib.load(file)[config['weights_key']]
    if review is not None:
        threshold = recipe()[0]['threshold']
    else:
        plan = json_at(f.WEAK/rotation/'policy.json')
        budget = json_at(f.HERE/'factor_numeric'/rotation/'study.json')['dev_fp']
        _, _, threshold, _, _ = calibrate({'dev':d.frame,'test':t.frame},REFERENCE,fit,cal,budget)
    old, p0 = result[0]
    held, p1 = result[-1]
    folder = R24/rotation
    sparse = json_at(folder/'fit.json')['models'][pos['sparse']]
    def adjust(c, current, phase):
        raw = load_sparse(folder,pos['sparse'],phase,c.frame,sparse)
        yes, margin, cut = scope(c.texts,c.state,threshold,None)
        where, _, _ = scope(c.texts,c.state,threshold,pos['width'])
        local = [text for text, inside in zip(c.texts,yes,strict=True) if inside]
        x = signals(c.frame[REFERENCE],None,raw[:,1])
        pre = proposed(x[yes],local,margin[yes],cut,pos['mode'],w['model'])
        rank = ranking(margin[where],pre[where[yes]],pos['mix'],pos['direction'])
        fixed = current.copy()
        fixed[where] = rank>pos['cut']
        if np.any(where & c.active):
            raise ValueError('A positive correction touched the negative cohort')
        return fixed
    d27 = adjust(d,old,'dev')
    t27 = d27 if t is d else adjust(t,held,'test')
    report = row['cuts'][kind]
    assert f.comparison(d.frame['y'][cal],old[cal],d27[cal]) == report['calibration']
    path = study_folder/'predictions.npz'
    assert checksum(path)==study['predictions_sha256']
    with np.load(path) as z:
        expected = t27 if original is not None else t27[review]
        if not np.array_equal(expected,z[f'{key}_{kind}']):
            raise ValueError('Round 27 held-out baseline did not reproduce')
    return d,t,fit,cal,review,d27,t27,p0,p1
