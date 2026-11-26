"""Reconstruct the released Round 29 operating point on all experiment rotations."""
import sys

import joblib
import numpy as np

from embed import HERE, ROOT, f, checksum, json_at
R28=ROOT/'round28'
sys.path.insert(0,str(R28))
from paired_context import context, match, strict_cut


def reference(rotation):
    dev,test,fit,cal,review,d0,t0,p0,p1=context(rotation)
    base=R28/'studies/sparse'/rotation
    trained=json_at(base/'results.json')
    neg_choice=json_at(R28/'public/release_selection.json')['selected']
    _, candidate, criterion=neg_choice.split(':')
    option=trained['options'][candidate]
    threshold_y=dev.active&cal&(abs(p0)<option['width'])
    y=dev.frame['y'][threshold_y]
    positives=int((y & d0[threshold_y]).sum())+option['cuts'][criterion]['cal']['delta_tp']
    values=np.sort(p0[threshold_y][y>0])[::-1]
    cutoff=float(np.nextafter(values[positives-1],-np.inf)) if positives else float(max(p0[threshold_y]))
    if not abs(cutoff)<=option['width']:
        raise ValueError('The archived threshold is outside its affected band')
    approved=R28/'recover_public'
    key=json_at(approved/'release_selection.json')['selected']
    configuration=json_at(approved/'policy.json')
    if key not in configuration['options'] or configuration['options']!=[key]:
        raise ValueError('Round 29 accepted a different policy')
    result_dir=R28/'recover_calibrated'/rotation
    results=json_at(result_dir/'results.json')
    if checksum(result_dir/'results.json')!=configuration['tables_sha256'][rotation]:
        raise ValueError('Round 29 calibration changed')
    name,crit=key.rsplit(':',1)
    record=results['options'][name]
    cfg=configuration['variants'][key]['setting']['recover']
    path=result_dir/'heads.joblib'
    if checksum(path)!=results['weights_sha256']:
        raise ValueError('Round 29 weight fingerprint changed')
    head=joblib.load(path)['compact']

    def adjust(c,b,p):
        pure=b.copy()
        margin=p.copy()
        margin[c.active]-=cutoff
        pure[c.active]=margin[c.active]>0.
        x=np.column_stack((p,margin,c.inputs[:,:3]))
        width=cfg['width']
        active=c.active&(abs(margin)<width)
        q=head.decision_function(x[active])
        w=cfg['mix']
        raw=(1.-w)*margin[active]+w*q
        if cfg['direction']=='recover':
            raw=np.where(pure[active],margin[active],np.maximum(raw,margin[active]))
        margin[active]=raw-record['cuts'][crit]['cut']
        new=pure.copy()
        new[active]=margin[active]>0.
        return new,margin,pure
    new0,m0,z0=adjust(dev,d0,p0)
    new1,m1,z1=(new0,m0,z0) if test is dev else adjust(test,t0,p1)
    assert f.comparison(dev.frame['y'][cal],z0[cal],new0[cal])==record['cuts'][crit]['cal']
    forecast=result_dir/'predictions.npz'
    if checksum(forecast)!=results['predictions_sha256']:
        raise ValueError('Round 29 forecast changed')
    with np.load(forecast) as z:
        actual=new1 if review is None else new1[review]
        if not np.array_equal(z[f'{name}_{crit}'],actual):
            raise ValueError('Round 29 decisions did not reproduce')
    return dev,test,fit,cal,review,new0,new1,m0,m1
