"""Reconstruct the frozen Round 37 ranks and recover the signed file pairs."""
from pathlib import Path
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R37 = ROOT/'round37'
sys.path.insert(0, str(R37))
from section37 import PARTS, f, checksum, json_at, write_json, inputs
from eval37 import unpack, features
from eval35 import crop
from points33 import OUT

TARGETS = ('negative_consensus','positive_retained')


def phases(rotation):
    rows,ref,base,cohorts = inputs(rotation)
    data,source = unpack(rotation,rows,base,cohorts,ref)
    compressed = data['oof']
    compressed['signed'] = compressed['replay'].copy()
    selected = json_at(R37/'selection.json')
    option = selected['best']['semantic']
    folder = R37/'comparison'/rotation
    report = folder/'results.json'
    assert checksum(report)==selected['source_table'][rotation]
    proof = json_at(report)
    weights = folder/'readouts.joblib'
    assert checksum(weights)==proof['weights_sha256']
    assert proof['section_receipt_sha256']==source
    model = joblib.load(weights)[option['source'],option['mode']]
    stored = folder/'predictions.npz'
    assert checksum(stored)==proof['predictions_sha256']
    cut = proof['options'][option['key']][option['rule']]['cut']
    for part,values in data.items():
        if part=='oof':
            continue
        old = values['latest'].copy()
        scope = values['support']&(abs(old)<option['width'])
        found = crop(values,scope)
        score = model.decision_function(features(found,ref['cutoff'],option['mode'],found['section']))
        w = option['strength']
        old[scope] = (1.-w)*old[scope]+w*score-cut
        values['latest'] = old
        values['signed'][scope] = old[scope]
    held = data.get('test',data['dev'])
    with np.load(stored) as z:
        assert np.array_equal(z['ids'],held['ids'][held['held']])
        assert np.array_equal(z[option['key']+'_'+option['rule']],held['signed'][held['held']]>0.)
    return data,ref,checksum(report)


def routes(data):
    known = data['old']
    scan = data['flag']
    return {'negative':known&(scan==0)&(data['unqueried']<=0.),
            'positive':known&(scan==1)}


def pairs(rotation, data, name):
    assert name in TARGETS
    folder = R37/'matched'
    meta = json_at(folder/'results.json')['analysis'][rotation]
    assert meta['points_sha256']==checksum(OUT/rotation/'receipt.json')
    source = folder/f'{rotation}.npz'
    assert meta['predictions_sha256']==checksum(source)
    offset = {row:i for i,row in enumerate(data['ids'])}
    with np.load(source) as z:
        a = np.asarray([offset[ix] for ix in z[name+'_anchor']],np.int64)
        b = np.asarray([offset[ix] for ix in z[name+'_control']],np.int64)
        for side,items in [('anchor',a),('control',b)]:
            assert np.array_equal(data['fold'][items],z[name+'_fold'])
            assert np.array_equal(data['duplicates'][items],z[name+'_'+side+'_dup_mask'])
    y=data['y']
    if name=='positive_retained':
        assert not y[a].any() and y[b].all()
        return b,a,checksum(source)
    assert y[a].all() and not y[b].any()
    return a,b,checksum(source)
