"""Recompute the held-out confusion counts and verify the structural timing receipts."""
from pathlib import Path
import csv
import sys
from statistics import median

import numpy as np

from features43 import HERE, ROOT, OUTPUT, PROGRAM, checksum, f, json_at, write_json
from pipeline import PUBLIC
from eval35 import cut
sys.path[:0] = [str(ROOT/'round41'),str(ROOT/'round42')]
from source41 import load, parts, PARTS
from audit42 import assert_equal


def verify(folder,stage,ranks,role):
    bitfile = folder/'predictions.npz'
    assert stage['predictions_sha256']==checksum(bitfile)
    assert stage['weights_sha256']==checksum(folder/'weights.joblib')
    row, y, base = ranks['ids'],ranks['y'],ranks['signed']>0.
    data = np.load(bitfile)
    assert np.array_equal(data['ids'],row)
    assert np.array_equal(data['truth'],y)
    assert np.array_equal(data['base'],base)
    full = np.ones(len(y),bool)
    group = {p:ranks['project']==p for p in np.unique(ranks['project'])}
    for key,q in stage['options'].items():
        for rule in ('fixed','tp','fp'):
            assert_equal(q[rule]['held'],key+'/'+rule,full,y,base,data[key+'_'+rule])
        if role!='official':
            for budget in ('tp','fp'):
                pred = data[key+'_same_'+budget]
                cell = q['matched'][budget]
                assert_equal(cell['change'],key,full,y,base,pred)
                for name,mask in group.items():
                    stats = assert_equal(cell['projects'][name],key+'/'+name,mask,y,base,pred)
                    assert stats['delta_'+budget]==0
    return dict(rows=len(row),options=len(stage['options']),predictions_sha256=checksum(bitfile))


def checks(folder,plan,source,rows):
    assert plan['code_sha256']==checksum(source)
    assert plan['csv_sha256']==checksum(folder/'comparison.csv')
    result = {}
    with (folder/'comparison.csv').open() as stream:
        points = list(csv.DictReader(stream))
    assert len(points)==plan.get('points',plan.get('rows'))
    assert any(p['eligible']=='True' for p in points)==bool(plan['chosen'])
    for role in PARTS:
        dest = folder/role
        file = dest/'results.json'
        meta = json_at(file)
        assert checksum(file)==plan['sources'][role]
        expected = HERE/'study43.py' if folder.name=='comparison' else source
        assert meta['code_sha256']==checksum(expected)
        assert meta['training']['cache_sha256']==checksum(HERE/'features/receipt.json')
        result[role] = verify(dest,meta,rows[role],role)
    return result


def profile(release,archive,fast=False):
    folder = HERE/('public_conditional_fast' if fast else 'public_conditional')
    pairs = 6 if fast else 5
    if not (folder/'results.json').exists():
        return None
    path = folder/'results.json'
    record = json_at(path)
    receipt = record['receipt']
    previous = ROOT/'round38/public_primary/candidate.npz'
    files = {'selection':HERE/'conditional/selection.json','public':PUBLIC,'anchor':previous,
             'archive':archive,'scanner':HERE/'scan_numeric.mjs' if fast else PROGRAM,
             'first_hit':ROOT/'round32/first_hit_public.npz',
             'readout':HERE/'conditional/official/weights.joblib',
             'code':HERE/'live43.py' if fast else HERE/'source_snapshots/live43.py',
             'features':HERE/'features43.py'}
    for tag,file in files.items():
        assert receipt[tag+'_sha256']==checksum(file)
    assert receipt['scanner_comparisons']==7279
    if fast:
        assert receipt['accelerator_sha256']==checksum(HERE/'fast43.py')
        earlier = HERE/'public_conditional/candidate.npz'
        assert receipt['prior_candidate_sha256']==checksum(earlier)
        assert receipt['matched_prediction_rows']==30655
        assert receipt['geometry_comparisons']>0
    assert record['choice']==json_at(files['selection'])['chosen']
    base = np.load(previous)
    truth = base['truth']
    assert truth.shape==(30655,2)
    all_bits = base['pred']
    measured,forecasts = {},{}
    for name,run in record['scores'].items():
        sample = folder/(name+'.npz')
        assert run['predictions_sha256']==checksum(sample)
        saved = np.load(sample)
        assert np.array_equal(saved['truth'],truth)
        bitpair = saved['pred']
        forecasts[name] = bitpair,saved['margin'],saved['scope']
        q = f.metric(truth,bitpair)
        assert q==run['quality']
        costs = run['trials']
        assert len(costs)==pairs and all(row['scanner_calls']==7279 for row in costs)
        assert run['cost']=={k:median(row[k] for row in costs) for k in run['cost']}
        s = [f.competition_formula(q['mean']['f1'],row['forward_seconds'],row['analytic_gflops'])
             for row in costs]
        assert s==[row['score'] for row in costs]
        assert abs(median(s)-run['score'])<1e-14
        measured[name] = s
    assert np.array_equal(forecasts['reference'][0],all_bits)
    after,rank,selected = forecasts['candidate']
    assert np.array_equal(after[:,1],all_bits[:,1])
    if fast:
        assert np.array_equal(after,np.load(earlier)['pred'])
    assert np.array_equal(after[~selected],all_bits[~selected])
    y = truth[:,0].astype(bool)
    ref = all_bits[:,0]
    assert record['change']['frozen']==f.comparison(y,ref,after[:,0])
    if selected.any():
        for k in ('tp','fp'):
            local = ref.copy()
            local[selected] = rank[selected]>cut(y[selected],ref[selected],rank[selected],k)
            assert record['change'][k]==f.comparison(y,ref,local)
    control,candidate = (record['scores'][name] for name in ('reference','candidate'))
    paired = [b-a for a,b in zip(measured['reference'],measured['candidate'],strict=True)]
    assert record['paired']==paired
    wins = sum(o>0 for o in paired)
    eligible = (wins>=(5 if fast else 4) and candidate['score']>max(control['score'],release['score']) and
                candidate['quality']['mean']['f1']>control['quality']['mean']['f1'])
    assert eligible==record['qualified']
    return dict(candidate=candidate['score'],reference=control['score'],wins=wins,pairs=len(paired),
                rows=len(truth),baseline_decisions=all_bits.size,scope=int(selected.sum()),
                frozen=record['change']['frozen'],qualified=eligible,sha256=checksum(path))


def main():
    parser = HERE/'parser.sha256'
    for entry in parser.read_text().splitlines():
        expected,path = entry.split()
        assert checksum(HERE/'wheels'/Path(path).name)==expected
    feature = json_at(HERE/'features/receipt.json')
    assert feature['context_sha256']==checksum(HERE/'features43.py')
    assert feature['parser_wheels_manifest_sha256']==checksum(parser)
    assert feature['scanner_receipt_sha256']==checksum(OUTPUT/'receipt.json')
    assert feature['rows_sha256']==checksum(HERE/'features/rows.npz')
    for name,details in feature['features'].items():
        assert details['sha256']==checksum(HERE/'features'/(name+'.npz'))
    timing = json_at(HERE/'micro.json')
    assert timing['code_sha256']==checksum(HERE/'source_snapshots/live43.py')
    assert timing['geometry_sha256']==checksum(OUTPUT/'training.npz')
    assert timing['rows_sha256']==checksum(HERE/'micro_rows.npz')
    assert timing['model_sha256']==checksum(HERE/'comparison/official/weights.joblib')
    assert {k:median(v) for k,v in timing['times'].items()}==timing['medians']
    assert all(len(v)==5 for v in timing['times'].values())
    held = {r:parts(load(r)[0])[1] for r in PARTS}
    out = {}
    for group,code in (('comparison','select43.py'),('conditional','conditional43.py')):
        folder = HERE/group
        plan = folder/'selection.json'
        spec = json_at(plan)
        out[group] = dict(verified=checks(folder,spec,HERE/code,held),selected=spec['chosen'],
                          selection_sha256=checksum(plan))
    public = json_at(ROOT/'round38/release_summary.json')
    archive = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert public['archive_sha256']==checksum(archive)
    result = dict(report=out,benchmark_sha256=checksum(HERE/'micro.json'),
                  release=public['score'],gap=public['target_gap'],public=profile(public,archive),
                  optimized=profile(public,archive,fast=True),
                  code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
