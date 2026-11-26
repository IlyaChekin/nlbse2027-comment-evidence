"""Independently verify paired-supervision predictions, budgets and model receipts."""
from pathlib import Path
import csv

import numpy as np

from fit42 import HERE, ROOT, PARTS, checksum, f, json_at, load, parts, write_json


def assert_equal(item, name, mask, y, base, prediction):
    seen = f.comparison(y[mask],base[mask],prediction[mask])
    if 'recovered' in item:
        subset, prior, predicted = y[mask], base[mask], prediction[mask]
        seen.update(recovered=int((subset & ~prior & predicted).sum()),
                    lost=int((subset & prior & ~predicted).sum()))
    if seen!=item:
        raise ValueError(f'Predictions disagree with report: {name}')
    return seen


def check(rotation,wide,report_hash):
    stage = HERE/('wide' if wide else 'comparison')/rotation
    fitted = HERE/('wide' if wide else 'fitted')/rotation
    meta = stage/'results.json'
    res = json_at(meta)
    assert checksum(meta)==report_hash
    fit_receipt = fitted/('fit.json' if wide else 'receipt.json')
    receipt = json_at(fit_receipt)
    assert checksum(fit_receipt)==res['fit_sha256' if wide else 'fit_receipt_sha256']
    if wide:
        assert receipt['code_sha256']==checksum(HERE/'wide42.py')
        refs = receipt['hashes']
    else:
        assert receipt['code_sha256']==checksum(HERE/'fit42.py')
        refs = {'scores.npz':receipt['scores_sha256'],'pair_roster.npz':receipt['pair_roster_sha256'],
                'weights.joblib':receipt['weights_sha256']}
    for name,stamp in refs.items():
        assert checksum(fitted/name)==stamp
    matrix,_,stamp = load(rotation)
    assert receipt['matrix_sha256']==stamp
    _, frame = parts(matrix)
    table = stage/'predictions.npz'
    assert checksum(table)==res['predictions_sha256']
    bank = np.load(table)
    y,base = bank['y'],bank['base']
    assert np.array_equal(bank['ids'],frame['ids'])
    assert np.array_equal(y,frame['y']) and np.array_equal(base,frame['signed']>0.)
    groups = {p:frame['project']==p for p in np.unique(frame['project'])}
    all_rows = np.ones(len(y),bool)
    for k,v in res['variants'].items():
        for rule in ('fixed','fp','tp'):
            candidate = bank[k+'_'+rule]
            assert_equal(v[rule]['held'],k+'/'+rule,all_rows,y,base,candidate)
        if rotation!='official':
            bits = bank[k+'_matched']
            case = v['matched']['fp']
            assert_equal(case['change'],k,all_rows,y,base,bits)
            for prj,selected in groups.items():
                summary = assert_equal(case['projects'][prj],k+'/'+prj,selected,y,base,bits)
                assert summary['delta_fp']==0
    return dict(held_rows=len(y),options=len(res['variants']),results_sha256=checksum(meta),
                training_sha256=checksum(fit_receipt))


def main():
    panel = HERE/'comparison/selection.json'
    report = json_at(panel)
    assert report['code_sha256']==checksum(HERE/'evaluate42.py')
    assert report['csv_sha256']==checksum(panel.with_name('comparison.csv'))
    with panel.with_name('comparison.csv').open() as fh:
        lines = list(csv.DictReader(fh))
    assert len(lines)==report['points']
    assert (report['picked'] is not None)==any(x['eligible']=='True' for x in lines)
    wide = HERE/'wide/selection.json'
    extension = json_at(wide)
    assert extension['code_sha256']==checksum(HERE/'wide42.py')
    assert (extension['chosen'] is not None)==any(x['eligible'] for x in extension['rows'])
    fixed = {k:check(k,False,report['sources'][k]) for k in PARTS}
    extra = {k:check(k,True,extension['sources'][k]) for k in PARTS}
    release = json_at(ROOT/'round38/release_summary.json')
    archive = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert release['archive_sha256']==checksum(archive)
    results = dict(control=release['score'],gap=release['target_gap'],
                   paired=fixed,extension=extra,signed_comparison=[checksum(panel),checksum(wide)],
                   chosen_paired=report['picked'],chosen_wide=extension['chosen'],
                   code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',results)
    print(results,flush=True)


if __name__=='__main__':
    main()
