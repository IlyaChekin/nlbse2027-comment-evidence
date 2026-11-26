"""Verify live results and isolate the contribution of the learned ranking."""
import csv
from pathlib import Path
from statistics import median

import numpy as np

from pairs import HERE, ROOT, f, checksum, json_at, write_json, clip_logit
from paired_context import context
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes

OUT=HERE/'public'


def load_csv(path, sha):
    if checksum(path)!=sha:
        raise ValueError('A prediction fingerprint changed')
    with path.open(newline='') as src:
        values=list(csv.DictReader(src))
    assert [int(r['row']) for r in values]==list(range(len(values)))
    return values


def probability(folder, name=None):
    results=json_at(folder/'results.json')
    specified=results['selected'] if name is None else name
    file=folder/f'{specified.replace(":","_")}.csv'
    data=load_csv(file,results['results'][specified]['predictions_sha256'])
    features=np.asarray([float(r['probability']) for r in data], dtype=np.float32)
    labels=np.asarray([int(r['pred_vulnerability']) for r in data],dtype=bool)
    return features,labels


def source_margin():
    r24=ROOT/'round24'
    config=json_at(r24/'official/tuned/policy.json')
    base=config['candidates'][config['selected']]['threshold']
    old,_=probability(r24/'public')
    r25=ROOT/'round25/views/public'
    cfg=json_at(r25/'policy.json')
    extra=cfg['support'][cfg['selected']]
    current,read=probability(r25)
    prior=ROOT/'round22/union_public'
    with np.load(prior/'scanner_predictions.npz') as z:
        scope=z['gate'].copy()
        scope[scope]=z['flag']==0
    sliced=scope&(abs(old-base)<extra['distance'])
    original=clip_logit(old.astype(np.float64))-clip_logit(np.asarray(np.float32(base),dtype=np.float64))
    original[sliced]=clip_logit(current[sliced].astype(np.float64))-clip_logit(np.asarray(extra['threshold']))
    if not np.array_equal(original[scope]>0,read[scope]):
        raise ValueError('Stored input precision is insufficient to reproduce the negative scope')
    return original,scope


def matrices(rows):
    y=np.asarray([[int(r[k]) for k in ('label_vulnerability','label_mat')] for r in rows],dtype=np.int8)
    p=np.asarray([[int(r[k]) for k in ('pred_vulnerability','pred_MAT')] for r in rows],dtype=bool)
    scope=np.asarray([int(r['scoped']) for r in rows],dtype=bool)
    return y,p,scope


def main():
    profile=json_at(OUT/'results.json')
    policy=json_at(OUT/'policy.json')
    assert checksum(OUT/'policy.json')==profile['policy_sha256']
    receipt=json_at(OUT/'release_selection.json')
    assert checksum(OUT/'results.json')==receipt['comparison_sha256']
    _,true,meta=load(PUBLIC,audit=True)
    assert checksum(PUBLIC)==profile['public_sha256']
    group=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    previous=ROOT/'round27/public'
    baseline=json_at(previous/'results.json')
    current=load_csv(previous/'candidate.csv',baseline['results'][baseline['selected']]['predictions_sha256'])
    _,old,_=matrices(current)
    d,_,_,cal,_,d27,_,p0,_=context('official')
    margin,eligible=source_margin()
    ablation={}
    for key, record in profile['results'].items():
        rows=load_csv(OUT/record['csv'],record['predictions_sha256'])
        y,z,s=matrices(rows)
        assert np.array_equal(y,true) and np.array_equal(z[~s],old[~s]) and np.array_equal(z[:,1],old[:,1])
        assert all(v==median(x[k] for x in record['trials']) for k,v in record['cost'].items())
        metric=f.metric(y,z)
        assert metric==record['quality']
        assert f.competition_formula(metric['mean']['f1'],record['cost']['forward_seconds'],
                                     record['cost']['analytic_gflops'])==record['score']
        if key=='reference':
            continue
        assert np.all(s<=eligible)
        cfg=policy['variants'][key]
        width=cfg['setting']['width']
        same=d.active & cal
        if width is not None:
            same &= abs(p0)<width
            assert np.array_equal(s,eligible&(abs(margin)<width))
        truth=d.frame['y'][same]
        initial=int((truth & d27[same]).sum())
        target=initial+cfg['cal']['delta_tp']
        order=np.sort(p0[same][truth>0])[::-1]
        cut=float(np.nextafter(order[target-1],-np.inf)) if target else float(max(p0[same]))
        alternative=old[:,0].copy()
        alternative[s]=margin[s]>cut
        quality=f.comparison(y[:,0],old[:,0],z[:,0])
        assert quality==record['frozen']
        ablation[key]=dict(interval=bootstrap_delta(y,old,z,group),recalibration_only=f.comparison(y[:,0],old[:,0],alternative),
                           learned_vs_threshold=f.comparison(y[:,0],alternative,z[:,0]),
                           recalibration_cut=cut,calibration_tp=target,actual_positive=int((z[:,0]&s).sum()),
                           change=quality)
        path=OUT/f'threshold_{record["csv"].replace(".csv",".npz")}'
        np.savez_compressed(path,rows=np.arange(len(y)),scope=s,margin=margin,reference=old[:,0],
                            threshold_only=alternative,truth=y[:,0],learned=z[:,0])
        ablation[key]['predictions_sha256']=checksum(path)
    if receipt['selected']!='reference':
        signed=json_at(HERE/'release_verification/receipt.json')
        path=ROOT/'submission/NLBSE2027_round28_negative_tool.zip'
        assert checksum(path)==signed['archive_sha256'] and signed['identical_binary_decisions']==2*len(true)
    write_json(OUT/'audit.json',dict(options=ablation,source=checksum(OUT/'results.json'),selected=receipt['selected']))
    print(ablation,flush=True)


if __name__=='__main__':
    main()
