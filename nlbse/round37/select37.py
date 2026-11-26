"""Freeze a candidate on development and held-out projects ahead of public measurement."""
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'round35'))
from select35 import PARTS, enumerate_options, read, sha, supported


def verify():
    sources = {}
    release = read(ROOT/'round34/release_summary.json')
    study = read(ROOT/'round35/broad/selection.json')
    for part in PARTS:
        folder = HERE/'comparison'/part
        table = folder/'results.json'
        record = read(table)
        assert record['rotation']==part
        assert sha(HERE/'eval37.py')==record['code_sha256']
        assert sha(folder/'readouts.joblib')==record['weights_sha256']
        assert sha(folder/'predictions.npz')==record['predictions_sha256']
        source = HERE/'views'/part
        receipt = read(source/'receipt.json')
        assert sha(source/'receipt.json')==record['section_receipt_sha256']
        assert sha(HERE/'section37.py')==receipt['code_sha256']
        assert sha(source/'scores.npz')==receipt['state_sha256']
        before = ROOT/'round35/broad'/part
        assert sha(before/'results.json')==study['result_sha256']['broad'][part]
        assert record['cohorts_sha256']==receipt['cohorts_sha256']==sha(before/'cohorts.npz')
        assert record['baseline_sha256']==receipt['baseline_sha256']==read(before/'results.json')['baseline_sha256']
        sources[part] = (record,sha(table))
    return sources,release


def key(row):
    return row['width'],row['source'],row['mode'],row['strength'],row['rule']


def main():
    proof,release = verify()
    options = enumerate_options('sections',proof)
    lookup = {key(row):row for row in options}
    for row in options:
        width,source,mode,mix,rule = key(row)
        for label,variant in (('scores','control'),('native','native')):
            tier = variant+'_trees' if mode.endswith('_trees') else variant
            other = lookup.get((width,source,tier,mix,rule))
            delta = other or row
            for field in ('a_tp','b_tp','a_fp','b_fp','cal_f1','review_f1'):
                row[f'{field}_over_{label}'] = row[field]-delta[field]
    approved = [row for row in options if supported(row)]
    established = [r for r in approved if r['mode'] in ('control','control_trees')]
    improvement = [r for r in approved if r['mode'] not in ('control','control_trees')]
    regions = [r for r in improvement if not r['mode'].startswith('native')]
    native = [r for r in improvement if r['mode'].startswith('native')]
    preference = lambda r:(min(r['cal_f1'],r['review_f1']),r['cal_f1'],r['review_f1'],-r['cal_rows'])
    supported_regions = [r for r in regions if
                         (min(r['a_tp_over_native'],r['b_tp_over_native'])>=0 and
                          r['a_tp_over_native']+r['b_tp_over_native']>0) or
                         (min(r['a_fp_over_native'],r['b_fp_over_native'])>=0 and
                          r['a_fp_over_native']+r['b_fp_over_native']>0)]
    best = {name:max(pool,key=preference,default=None) for name,pool in
            [('native_or_new',improvement),('old_scores',established),('regions',regions),
             ('regions_beyond_native',supported_regions),('native',native)]}
    semantic = [r for r in supported_regions if not r['mode'].startswith('half')]
    best['semantic'] = max(semantic,key=preference,default=None)
    picked = best['native_or_new']
    table = HERE/'comparison.csv'
    with table.open('w',newline='') as out:
        writer = csv.DictWriter(out,fieldnames=options[0].keys())
        writer.writeheader()
        writer.writerows(options)
    result = dict(selected=picked,best=best,tested=len(options),supported=len(approved),
                  fresh_supported=len(improvement),regional_supported=len(regions),
                  regional_beyond_native=len(supported_regions), source_table={p:r[1] for p,r in proof.items()},
                  csv_sha256=sha(table),code_sha256=sha(Path(__file__)),base_archive_sha256=release['archive_sha256'])
    (HERE/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
