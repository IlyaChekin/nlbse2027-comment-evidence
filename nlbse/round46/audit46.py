"""Verify model assets, window alignment, held decisions and timing receipts."""
import csv
from pathlib import Path
from statistics import median
from types import SimpleNamespace
import sys

import numpy as np

from views46 import (HERE,ROOT,MODEL,UNIX,DEST as WINDOWS,PARTS,f,checksum,json_at,load,
                     parts,source,write_json)
from fit46 import DEST as FIT,BANK
from extract46 import operations
from compare46 import DEST
sys.path.insert(0,str(ROOT/'round42'))
from audit42 import assert_equal


def inputs():
    meta = json_at(WINDOWS/'receipt.json')
    assert meta['code_sha256']==checksum(HERE/'views46.py')
    assert meta['roster_sha256']==checksum(WINDOWS/'rows.npz')
    assert meta['upstream_sha256']==checksum(MODEL/'upstream.json')
    assert all(checksum(MODEL/k)==v for k,v in meta['assets'].items())
    assert all(checksum(UNIX/k)==v for k,v in meta['unix_tokenizer'].items())
    _,upstream = source()
    expected = {'qwen':upstream['model.safetensors']['lfs']['sha256'],'unix':checksum(UNIX/'pytorch_model.bin')}
    assert expected['qwen']==checksum(MODEL/'model.safetensors')
    receipts,storage = {},{}
    for k in meta['windows']:
        file = WINDOWS/(k+'.npz')
        assert meta['windows'][k]['sha256']==checksum(file)
        with np.load(file) as z:
            cap = int(k.split('_')[1])
            assert len(z['lengths'])==len(z['full'])==meta['rows']
            assert np.array_equal(np.diff(z['ends']),z['lengths'])
            assert max(z['lengths'])<=cap and z['ends'][-1]==len(z['tokens'])
            assert int(z['full'].sum())==meta['windows'][k]['full_source']
            other = storage.setdefault(cap,z['full'])
            assert np.array_equal(other,z['full'])
        path=HERE/'features'/k/'receipt.json'
        data = json_at(path)
        assert data['window_receipt_sha256']==checksum(WINDOWS/'receipt.json')
        assert data['window_sha256']==checksum(file) and data['rows']==meta['rows']
        assert data['code_sha256']==checksum(HERE/'extract46.py')
        assert data['model_sha256']==expected[k.split('_')[0]]
        for name,digest in data['features'].items():
            assert digest==checksum(path.with_name(name))
        receipts[k] = checksum(path)
    file = HERE/'micro/receipt.json'
    micro = json_at(file)
    assert micro['same_states']
    assert micro['intermediate_exact_rows']==128 and micro['model_sha256']==expected['qwen']
    assert micro['rows_sha256']==checksum(file.with_name('rows.npz'))
    assert micro['code_sha256']==checksum(HERE/'micro46.py') and micro['extractor_sha256']==checksum(HERE/'extract46.py')
    assert len(micro['tokenizer_trials'])==5 and micro['tokenizer_median']==median(micro['tokenizer_trials'])
    for key,z in micro['forward'].items():
        assert len(z)==5
        assert micro['median'][key]=={k:median(v[k] for v in z) for k in z[0]}
    return receipts,meta['matrix'],checksum(file)


def verify(role,certificates,expected,signature,train,comparison,code,selector):
    fit = train/role
    fit_note = json_at(fit/'receipt.json')
    assert fit_note['code_sha256']==checksum(HERE/code)
    assert fit_note['model_sha256']==checksum(fit/'weights.joblib')
    assert fit_note['predictions_sha256']==checksum(fit/'scores.npz')
    assert fit_note['matrix_sha256']==signature
    for key,(folder,_) in BANK.items():
        assert fit_note['source_features'][key]==certificates[folder]
    cohort,t,ticket = load(role)
    _,held = parts(cohort)
    assert signature==ticket
    stage = comparison/role
    path = stage/'results.json'
    assert expected==checksum(path)
    data = json_at(path)
    assert data['fit_sha256']==checksum(fit/'receipt.json')
    assert data['code_sha256']==checksum(HERE/selector)
    predicates = stage/'predictions.npz'
    assert data['predictions_sha256']==checksum(predicates)
    with np.load(predicates) as stored:
        y,base = held['y'],held['signed']>0.
        assert np.array_equal(stored['ids'],held['ids'])
        assert np.array_equal(stored['base'],base) and np.array_equal(stored['truth'],y)
        groups = {p:held['project']==p for p in np.unique(held['project'])}
        full = np.ones(len(y),bool)
        for key,opts in data['options'].items():
            for k in ('fixed','fp','tp'):
                assert_equal(opts[k]['held'],key+'/'+k,full,y,base,stored[key+'_'+k])
                if role!='official' and k!='fixed':
                    match = stored[key+'_same_'+k]
                    assert_equal(opts['matched'][k]['change'],key,full,y,base,match)
                    for group,m in groups.items():
                        res = assert_equal(opts['matched'][k]['projects'][group],key+'/'+group,m,y,base,match)
                        assert res['delta_'+k]==0
    return dict(options=len(data['options']),source_sha256=expected,predictions_sha256=checksum(predicates))


def focused():
    directory = HERE/'routed'
    card = directory/'receipt.json'
    r = json_at(card)
    assert r['code_sha256']==checksum(HERE/'routed46.py')
    select = DEST/'selection.json'
    assert r['selection_sha256']==checksum(select)
    choice = json_at(select)['models']['qwen384']['quality_only']
    assert r['selection']==choice
    source,_,_ = load('official')
    _,frame = parts(source)
    file = directory/'predictions.npz'
    assert r['predictions_sha256']==checksum(file)
    p = np.load(file)
    before = frame['signed']>0.
    candidate,mask = p['pred'],p['scope']
    assert np.array_equal(p['ids'],frame['ids']) and np.array_equal(p['base'],before)
    assert r['active']==int(mask.sum())==choice['review_rows']
    assert np.array_equal(candidate[mask],p['rank']>choice['cut'])
    assert np.array_equal(candidate[~mask],before[~mask])
    assert_equal(r['review'],choice['key'],np.ones(len(mask),bool),frame['y'],before,candidate)
    other = np.load(DEST/'official/predictions.npz')
    assert r['same_predictions']==np.array_equal(candidate,other[choice['key']+'_'+choice['rule']])
    z = r['medians']
    assert len(r['trials'])==7 and z=={k:median(v[k] for v in r['trials']) for k in z}
    charge = .2*(z['wall_seconds']+z['tokenization_seconds'])/1000.+.2*z['analytic_gflops']/3e6
    quality=.3*(r['review']['f1_after']-r['review']['f1_before'])
    assert r['score_delta_cost']==charge and r['score_delta_quality']==quality
    assert r['net_estimate']==quality-charge
    universe = np.load(WINDOWS/'rows.npz')['ids']
    positions = np.searchsorted(universe,frame['ids'][mask])
    assert np.array_equal(universe[positions],frame['ids'][mask])
    lengths = np.load(WINDOWS/'qwen_384.npz')['lengths'][positions]
    cfg = SimpleNamespace(**json_at(MODEL/'config.json'))
    forward = SimpleNamespace(config=cfg)
    unpadded = sum(operations(forward,1,int(length),True) for length in lengths)
    cached = other[choice['key']+'_'+choice['rule']]
    return dict(rows=r['active'],measured_piece=r['score_delta_cost'],quality_score=quality,
                net_estimate=quality-charge,decision_differences=int(np.count_nonzero(candidate!=cached)),
                unpadded_gflops=unpadded,unpadded_compute_score_cost=.2*unpadded/3e6,
                cached_quality_gain=.3*choice['review_f1'],sha256=checksum(card))


def main():
    sources,matrices,micro = inputs()
    archive = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    release = json_at(ROOT/'round38/release_summary.json')
    assert release['archive_sha256']==checksum(archive)
    coverage = HERE/'coverage.json'
    meter = json_at(coverage)
    assert meter['code_sha256']==checksum(HERE/'coverage46.py')
    for role,report in meter['rotations'].items():
        assert report['predictions_sha256']==checksum(FIT/role/'scores.npz')
        assert report['windows_sha256']==checksum(WINDOWS/'receipt.json')
    table = {}
    studies = ((FIT,DEST,'fit46.py','compare46.py'),
               (HERE/'paired',HERE/'paired_comparison','pairs46.py','compare46.py'),
               (FIT,HERE/'length','fit46.py','long46.py'))
    for train,results,code,selector in studies:
        plan = results/'selection.json'
        selected = json_at(plan)
        assert selected['code_sha256']==checksum(HERE/selector)
        assert selected['csv_sha256']==checksum(results/'comparison.csv')
        with (results/'comparison.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows)==selected['tested']
        assert any(r['eligible']=='True' for r in rows)==any(selected['selection'].values())
        assert release['archive_sha256']==selected['archive_sha256']
        checks = {role:verify(role,sources,selected['sources'][role],matrices[role],train,results,code,selector) for role in PARTS}
        table[results.name] = dict(selection=selected['selection'],tested=selected['tested'],checked=checks,
                                   selection_sha256=checksum(plan))
    data = dict(score=release['score'],gap=release['target_gap'],tables=table,
                coverage_sha256=checksum(coverage),routed=focused(),code_sha256=checksum(Path(__file__)),micro_sha256=micro,
                archive_sha256=release['archive_sha256'])
    write_json(HERE/'audit.json',data)
    print(data,flush=True)


if __name__=='__main__':
    main()
