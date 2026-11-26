"""Verify controlled public decisions and compute a paired file-level interval."""
import csv

import numpy as np

from embed import HERE, ROOT, f, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes

OUT=HERE/'public'


def matrix(path,expected):
    if checksum(path)!=expected:
        raise ValueError('Compared predictions changed')
    with path.open(newline='') as src:
        rows=list(csv.DictReader(src))
    assert list(range(len(rows)))==[int(r['row']) for r in rows]
    forecasts=np.array([[int(r[k]) for k in ('pred_vulnerability','pred_MAT')] for r in rows],dtype=bool)
    target=np.array([[int(r[k]) for k in ('label_vulnerability','label_mat')] for r in rows],dtype=np.int8)
    scope=np.asarray([int(r['scoped']) for r in rows],dtype=bool)
    matched=np.asarray([int(r['matched']) for r in rows],dtype=bool)
    return forecasts,target,scope,matched


def main():
    r=json_at(OUT/'results.json')
    assert checksum(PUBLIC)==r['public_sha256'] and checksum(OUT/'policy.json')==r['policy_sha256']
    a,b=r['results']['reference'],r['results'][r['selected']]
    ref,y,_,_=matrix(OUT/'reference.csv',a['predictions_sha256'])
    pred,yt,s,paired=matrix(OUT/'candidate.csv',b['predictions_sha256'])
    _,official,meta=load(PUBLIC,audit=True)
    assert np.array_equal(y,official) and np.array_equal(y,yt)
    assert np.array_equal(ref[:,1],pred[:,1]) and np.array_equal(ref[~s],pred[~s])
    assert f.comparison(y[:,0],ref[:,0],paired)==b['matched']
    assert f.comparison(y[:,0],ref[:,0],pred[:,0])==b['frozen']
    for p,q in ((ref,a),(pred,b)):
        metrics=f.metric(y,p)
        assert metrics==q['quality']
        c=q['timing']
        assert f.competition_formula(metrics['mean']['f1'],c['forward_seconds'],c['analytic_gflops'])==q['score']
    origin=ROOT/'round28/recover_public'
    published=json_at(origin/'results.json')
    sel=json_at(origin/'release_selection.json')['selected']
    old=published['results'][sel]
    file=origin/old['csv']
    assert checksum(file)==old['predictions_sha256']
    with file.open(newline='') as src:
        expect=np.array([[int(r[k]) for k in ('pred_vulnerability','pred_MAT')] for r in csv.DictReader(src)],dtype=bool)
    assert np.array_equal(ref,expect)
    codes=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    interval=bootstrap_delta(y,ref,pred,codes)
    data=dict(interval_95=interval,scope=int(s.sum()),changed=int((ref[:,0]!=pred[:,0]).sum()),
              benchmark_sha256=checksum(OUT/'results.json'))
    write_json(OUT/'audit.json',data)
    print(data,flush=True)


if __name__=='__main__':
    main()
