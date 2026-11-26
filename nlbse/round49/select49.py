"""Audit comparison artifacts and require held-out improvements for routing."""
import csv
from pathlib import Path

import numpy as np

from route49 import HERE, ROOT, DEST, LAYOUT, PARTS, f, checksum, json_at, load, write_json
from source41 import parts

OUT = HERE/'comparison'


def tally(y,z):
    return int((z&y).sum()),int((z&~y).sum())


def signed(raw):
    return raw['f1_after']-raw['f1_before']


def load_comparison(rotation):
    source,_,stamp=load(rotation)
    cal,held=parts(source)
    f0=DEST/rotation/'results.json'
    card=json_at(f0)
    assert card['code_sha256']==checksum(HERE/'route49.py')
    assert card['matrix_sha256']==stamp and card['layout_sha256']==checksum(LAYOUT/'receipt.json')
    assert card['predictions_sha256']==checksum(f0.with_name('predictions.npz'))
    assert card['weights_sha256']==checksum(f0.with_name('models.joblib'))
    y=held['y']
    base=held['signed']>0.
    with np.load(f0.with_name('predictions.npz')) as x:
        assert np.array_equal(x['ids'],held['ids'])
        assert np.array_equal(x['calibration_ids'],cal['ids']) and not np.intersect1d(x['ids'],x['calibration_ids']).size
        assert np.array_equal(x['truth'],y) and np.array_equal(x['base'],base)
        assert np.array_equal(x['old'],held['old']) and np.array_equal(x['size'],held['length'])
        for key,values in card['results'].items():
            q=x[key+'_mask']
            t=x[key+'_prediction']
            result=values['held']
            assert f.comparison(y,base,t)==result['frozen']
            assert int(q.sum())==int(held['old'].sum())==result['budget']['calls']
            assert int(held['length'][q].sum())==result['budget']['characters']<=result['budget']['old_characters']
            assert np.array_equal(t[q==held['old']],base[q==held['old']])
            for k,j in [('fp',1),('tp',0)]:
                if rotation=='official':
                    continue
                scored=x[key+'_at_'+k]
                assert f.comparison(y,base,scored)==result['matched'][k]['change']
                for group in np.unique(held['project']):
                    mask=held['project']==group
                    assert tally(y[mask],base[mask])[j] == tally(y[mask],scored[mask])[j]
    return card,checksum(f0)


def select():
    OUT.mkdir(exist_ok=True)
    receipts={}
    studies={}
    for name in PARTS:
        studies[name],receipts[name]=load_comparison(name)
    A,B,official=(studies[k]['results'] for k in PARTS)
    options=[]
    for k, row in official.items():
        arm,learner,policy,fraction=k.split('/')
        controls=[f'{root}/{learner}/{policy}/{fraction}' for root in ('numeric','portfolio') if root!=arm]
        # Give the numeric control its best learner under the same accounting.
        hard=controls + [f'{v}/{other}/{policy}/{fraction}' for v in ('numeric','portfolio')
                         for other in ('direct','factor') if other!=learner]
        qa,qb=A[k]['held'],B[k]['held']
        changes={}
        for limit,key,direction in [('fp','tp',1),('tp','fp',-1)]:
            arr=[direction*q['matched'][limit]['change']['delta_'+key] for q in (qa,qb)]
            stronger=[u-max([0]+[direction*side[n]['held']['matched'][limit]['change']['delta_'+key] for n in hard])
                      for u,side in zip(arr,(A,B),strict=True)]
            changes[limit]=(arr,stronger)
        cal=row['calibration']['frozen']
        rev=row['held']['frozen']
        beat=signed(cal)-max([0.]+[signed(official[base]['calibration']['frozen']) for base in hard])
        supports=any(min(pair)>=0 and sum(pair)>0 for _,pair in changes.values())
        guard=signed(cal)>0. and signed(rev)>0. and beat>0. and supports
        options.append(dict(arm=arm,learner=learner,policy=policy,fraction=float(fraction),key=k,
                            a_tp=changes['fp'][0][0],b_tp=changes['fp'][0][1],
                            a_extra_tp=changes['fp'][1][0],b_extra_tp=changes['fp'][1][1],
                            a_fewer_fp=changes['tp'][0][0],b_fewer_fp=changes['tp'][0][1],
                            a_extra_fewer_fp=changes['tp'][1][0],b_extra_fewer_fp=changes['tp'][1][1],
                            cal_f1=signed(cal),review_f1=signed(rev),cal_vs_prior=beat,
                            cal_tp=cal['delta_tp'],cal_fp=cal['delta_fp'],
                            review_tp=rev['delta_tp'],review_fp=rev['delta_fp'],
                            cal_swaps=row['calibration']['budget']['swap'],review_swaps=row['held']['budget']['swap'],
                            eligible=guard and arm in ('layout','portfolio_layout')))
    path=OUT/'comparison.csv'
    with path.open('w',newline='') as t:
        w=csv.DictWriter(t,fieldnames=list(options[0]))
        w.writeheader()
        w.writerows(options)
    order=sorted(options,key=lambda r:(r['eligible'],min(r['review_f1'],r['cal_f1']),r['cal_vs_prior']),reverse=True)
    full=HERE/'public_ceiling/results.json'
    proof=json_at(full)
    assert proof['predictions_sha256']==checksum(full.with_name('endpoints.npz'))
    assert proof['code_sha256']==checksum(HERE/'ceiling49.py')
    summary=json_at(ROOT/'round38/release_summary.json')
    assert proof['archive_sha256']==summary['archive_sha256']==checksum(ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip')
    output=dict(checked=len(options),selection=next((r for r in order if r['eligible']),None),
                top=order[:8],proof=receipts,ceiling_sha256=checksum(full),
                fit_cases={k:v['cases'] for k,v in studies.items()},table_sha256=checksum(path),
                code_sha256=checksum(Path(__file__)))
    write_json(OUT/'selection.json',output)
    print(output,flush=True)


if __name__=='__main__':
    select()
