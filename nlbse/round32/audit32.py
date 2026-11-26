"""Check public metrics and paired uncertainty against the exact binary decisions."""
import csv
from statistics import median

import numpy as np

from context32 import HERE, R28, f, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes

OUT=HERE/'public'


def main():
    results=json_at(OUT/'results.json')
    assert checksum(PUBLIC)==results['public_sha256']
    assert checksum(HERE/'selection.json')==results['receipt']['selection_sha256']
    assert checksum(HERE/'live32.py')==results['code_sha256']
    _,target,meta=load(PUBLIC,audit=True)
    forecasts=[]
    for name,info in results['results'].items():
        path=OUT/f'{name}.npz'
        assert checksum(path)==info['predictions_sha256']
        with np.load(path) as z:
            assert np.array_equal(target,z['truth'])
            forecasts.append((z['pred'],z['scope']))
        assert info['quality']==f.metric(target,forecasts[-1][0])
        cost=info['cost']
        assert all(median(row[key] for row in info['trials'])==value for key,value in cost.items())
        assert info['score']==f.competition_formula(info['quality']['mean']['f1'],
                                                  cost['forward_seconds'],cost['analytic_gflops'])
    (ref,_),(chosen,scope)=forecasts
    assert np.array_equal(ref[~scope],chosen[~scope]) and np.array_equal(ref[:,1],chosen[:,1])
    previous=R28/'recover_public'
    selected=json_at(previous/'release_selection.json')['selected']
    path=previous/json_at(previous/'results.json')['results'][selected]['csv']
    assert checksum(path)==results['baseline_reference_sha256']
    with path.open() as handle:
        old=np.array([[int(row[k]) for k in ('pred_vulnerability','pred_MAT')]
                      for row in csv.DictReader(handle)],dtype=bool)
    assert np.array_equal(ref,old)
    assert f.comparison(target[:,0],ref[:,0],chosen[:,0])==results['quality']['frozen']
    classes=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    record=dict(confidence_interval_95=bootstrap_delta(target,ref,chosen,classes),draws=2000,
                clusters=int(max(classes)+1),scope=int(scope.sum()),changed=int((ref!=chosen).sum()),
                benchmark_sha256=checksum(OUT/'results.json'))
    write_json(OUT/'audit.json',record)
    print(record,flush=True)


if __name__=='__main__':
    main()
