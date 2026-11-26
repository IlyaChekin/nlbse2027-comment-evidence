"""Verify held-out decisions, fixed error budgets and query costs."""
import csv
from pathlib import Path
from statistics import median
import sys

import numpy as np

from policy45 import HERE,ROOT,PARTS,DEST as FIT,f,checksum,json_at,load,parts,write_json
sys.path.insert(0,str(ROOT/'round42'))
from audit42 import assert_equal


def source_receipt(folder,role,ticket):
    file = folder/role/'receipt.json'
    proof = json_at(file)
    assert proof['matrix_sha256']==ticket
    if folder==FIT:
        tmp = proof['files']
        assert proof['code_sha256']==checksum(HERE/'policy45.py')
        for k,m in tmp.items():
            assert m==checksum(folder/role/k)
    else:
        assert proof['code_sha256']==checksum(HERE/'switch45.py')
        for key in ('scores','weights'):
            suffix = '.npz' if key=='scores' else '.joblib'
            assert proof[key+'_sha256']==checksum(folder/role/(key+suffix))
        assert proof['fit_sha256']==checksum(FIT/role/'receipt.json')
    return checksum(file)


def verify(folder,role,ranks,receipt,alternate,archive_hash):
    p = folder/'selection.json'
    plan = json_at(p)
    stage = folder/role
    p = stage/'results.json'
    info = json_at(p)
    assert plan['sources'][role]==checksum(p)
    exchange = folder.name=='swap'
    assert info['code_sha256']==checksum(HERE/('swap45.py' if exchange else 'compare45.py'))
    assert plan['code_sha256']==info['code_sha256']
    if not exchange:
        assert plan['archive_sha256']==archive_hash
    assert info['fit_sha256']==receipt
    assert info['switch_sha256']==(None if folder.name=='comparison' else alternate)
    bits = stage/'predictions.npz'
    assert info['predictions_sha256']==checksum(bits)
    with np.load(bits) as predictions:
        old = ranks['signed']>0.
        gold = ranks['y']
        assert np.array_equal(predictions['ids'],ranks['ids'])
        assert np.array_equal(predictions['base'],old)
        assert np.array_equal(predictions['truth' if exchange else 'y'],gold)
        yes = np.ones(len(old),bool)
        project_masks = {v:ranks['project']==v for v in np.unique(ranks['project'])}
        for key,check in info['options'].items():
            if exchange:
                result = predictions[key]
                out = assert_equal(check['review'],key,yes,gold,old,result)
                assert out['delta_tp']==-out['delta_fp']
                for regime in np.unique(ranks['observed_regime']):
                    p = ranks['observed_regime']==regime
                    out = assert_equal(check['review_scanners'][str(int(regime))],key,p,gold,old,result)
                    assert out['delta_tp']==-out['delta_fp']
                continue
            for rule in ('fixed','fp','tp'):
                assert_equal(check[rule]['held'],key,yes,gold,old,predictions[key+'_'+rule])
                if rule!='fixed' and role!='official':
                    result = predictions[key+'_same_'+rule]
                    rec = check['matched'][rule]
                    assert_equal(rec['change'],key,yes,gold,old,result)
                    for name,mask in project_masks.items():
                        z = assert_equal(rec['projects'][name],key+'/'+name,mask,gold,old,result)
                        assert z['delta_'+rule]==0
    return dict(variants=len(info['options']),predictions_sha256=checksum(bits))


def main():
    release = json_at(ROOT/'round38/release_summary.json')
    archive = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    digest = checksum(archive)
    assert digest==release['archive_sha256']
    receipts = {}
    payload = {}
    for name in ('comparison','switch_comparison','swap'):
        here = HERE/name
        t = here/'selection.json'
        v = json_at(t)
        assert v['csv_sha256']==checksum(here/'comparison.csv')
        with (here/'comparison.csv').open() as file:
            table = list(csv.DictReader(file))
        assert v['tested']==len(table)
        assert any(q['eligible']=='True' for q in table)==bool(v['cheap'] or v['dense'])
        receipts[name] = dict(selection_sha256=checksum(t),rows=v['tested'],
                               selected_cheap=v['cheap'],selected_dense=v['dense'])
        payload[name] = {}
    for role in PARTS:
        frames,_,sig = load(role)
        _,r = parts(frames)
        first = source_receipt(FIT,role,sig)
        second = source_receipt(HERE/'switch',role,sig)
        for loc,details in payload.items():
            details[role] = verify(HERE/loc,role,r,first,second,digest)
    path = HERE/'micro.json'
    record = json_at(path)
    assert record['code_sha256']==checksum(HERE/'live45.py')
    assert record['roster_sha256']==checksum(HERE/'micro_rows.npz')
    assert {k:median(a) for k,a in record['trials'].items()}==record['medians']
    assert all(len(v)==6 for v in record['trials'].values())
    write_json(HERE/'audit.json',dict(tables=receipts,held=payload,micro_sha256=checksum(path),
                confirmed=release['score'],gap=release['target_gap'],code_sha256=checksum(Path(__file__))))
    print('verified',receipts,'cost',record['medians'],flush=True)


if __name__=='__main__':
    main()
