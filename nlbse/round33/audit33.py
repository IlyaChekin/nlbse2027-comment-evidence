"""Audit the potential and observed changes of scanner acquisition."""
import argparse
from statistics import median

import numpy as np

from points33 import HERE, R32, PARTS, OUT, f, checksum, json_at, write_json
from experiment33 import read_phase
from live33 import added_requests
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes

import joblib


def records(p):
    parent=OUT/p
    manifest=json_at(parent/'receipt.json')
    signed=read_phase('dev' if p=='official' else 'test',parent,manifest)
    active=signed['held']
    y,seen,old,flag,scan=(signed[k][active] for k in ('y','queried','unqueried','flag','old'))
    observed,base=seen>0.,old>0.
    result={}
    for direction,use,before,after in [('add',~scan,base,observed),('drop',scan,observed,base)]:
        change=after.astype(np.int8)-before.astype(np.int8)
        counters={}
        for cohort,sub in [('all',np.ones(len(y),bool)),('rule_negative',flag==0),('rule_positive',flag==1)]:
            mask=sub & use
            counters[cohort]=dict(rows=int(mask.sum()),tp_gained=int((mask & y & (change==1)).sum()),
                                  tp_lost=int((mask & y & (change==-1)).sum()),
                                  fp_added=int((mask & ~y & (change==1)).sum()),
                                  fp_removed=int((mask & ~y & (change==-1)).sum()))
        result[direction]=counters
    return result


def check_live_add():
    config=json_at(HERE/'joint_selection.json')['addition']
    for p in PARTS:
        target=HERE/'separate'/p
        comp=json_at(target/'results.json')
        archive=target/'models.joblib'
        assert comp['model_sha256']==checksum(archive)
        signed=joblib.load(archive)
        classifier=signed[config['source'],False]['new']
        parent=OUT/p
        header=json_at(parent/'receipt.json')
        sample=read_phase('dev' if p=='official' else 'test',parent,header)
        h=sample['held']
        query,_=added_requests(classifier,sample['pre'][h],sample['length'][h],sample['old'][h],header['cutoff'],config)
        with np.load(target/'predictions.npz') as z:
            if not np.array_equal(query,z[config['key']+'_requests']):
                raise ValueError(f'The runtime additional-call mask changed on {p}')


def public_check():
    root=HERE/'public'
    receipt=root/'results.json'
    results=json_at(receipt)
    if checksum(HERE/'joint_selection.json')!=results['source']['selection_sha256']:
        raise ValueError('The selected policy changed')
    assert checksum(HERE/'live33.py')==results['code_sha256']
    assert checksum(PUBLIC)==results['public_sha256']
    _,target,meta=load(PUBLIC,audit=True)
    forecasts=[]
    calls=[]
    for label,record in results['comparison'].items():
        file=root/f'{label}.npz'
        assert checksum(file)==record['predictions_sha256']
        with np.load(file) as data:
            assert np.array_equal(target,data['truth'])
            forecasts.append(data['pred'])
            calls.append(data['asked'])
        assert record['quality']==f.metric(target,forecasts[-1])
        cost=record['cost']
        assert all(median(run[k] for run in record['trials'])==v for k,v in cost.items())
        assert record['score']==f.competition_formula(record['quality']['mean']['f1'],
                                                     cost['forward_seconds'],cost['analytic_gflops'])
    reference,candidate=forecasts
    orig=R32/'public/candidate.npz'
    assert checksum(orig)==results['reference_sha256']
    with np.load(orig) as z:
        assert np.array_equal(reference,z['pred'])
    assert (calls[0]<=calls[1]).all() and np.array_equal(reference[:,1],candidate[:,1])
    assert np.array_equal(reference[~calls[1]],candidate[~calls[1]])
    effect={}
    y=target[:,0].astype(bool)
    for name,zone in [('addition',~calls[0]),('veto',calls[0])]:
        isolated=np.where(zone,candidate[:,0],reference[:,0])
        effect[name]=f.comparison(y,reference[:,0],isolated)
    final=json_at(root/'ceiling.json')
    path=root/'full_endpoints.npz'
    assert checksum(path)==final['endpoints_sha256']
    assert checksum(HERE/'ceiling33.py')==final['code_sha256']
    from ceiling33 import optimize_threshold
    with np.load(path) as z:
        assert np.array_equal(target,z['y'])
        assert np.array_equal(reference[:,0],np.where(z['old'],z['queried'],z['prior'])>0.)
        assert optimize_threshold(y,z['prior'],z['queried'],reference[:,0])==final['label_selected_threshold']
    assert results['quality']['frozen']==f.comparison(target[:,0],reference[:,0],candidate[:,0])
    clusters=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    record=dict(attempted_extra_queries=int((calls[1]&~calls[0]).sum()),
                changed_binary_decisions=int((reference!=candidate).sum()),
                confidence_interval_95=bootstrap_delta(target,reference,candidate,clusters),
                benchmark_sha256=checksum(receipt),isolated_effect=effect,oracle_sha256=checksum(root/'ceiling.json'))
    write_json(root/'audit.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public',action='store_true')
    args=p.parse_args()
    if args.public:
        print(public_check(),flush=True)
    else:
        check_live_add()
        result={phase:records(phase) for phase in PARTS}
        write_json(HERE/'outcomes.json',result)
        for phase,rec in result.items():
            print(phase,rec,flush=True)


if __name__=='__main__':
    main()
