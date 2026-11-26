"""Select archived-span policies at matched project error budgets."""
import csv
from pathlib import Path

import numpy as np

from eval55 import DEST as EVAL, ARMS, FIT, HERE, PARTS, ROOT, checksum, f, json_at, load, parts, write_json
from eval51 import geometry
from select51 import differences, f1

DEST=HERE/'comparison'


def inspect(rotation):
    path=EVAL/rotation/'results.json'
    r=json_at(path)
    matrix,_,stamp=load(rotation)
    cal,held=parts(matrix)
    assert stamp==r['matrix_sha256']
    assert r['code_sha256']==checksum(HERE/'eval55.py')
    assert r['fit_sha256']==checksum(FIT/rotation/'fit.json')
    assert r['scores_sha256']==checksum(path.with_name('scores.npz'))
    assert r['predictions_sha256']==checksum(path.with_name('predictions.npz'))
    base=held['signed']>0.
    y=held['y']
    owners={p:held['project']==p for p in np.unique(held['project'])}
    clean={}
    with np.load(path.with_name('predictions.npz')) as z:
        assert np.array_equal(z['ids'],held['ids']) and np.array_equal(z['cal_ids'],cal['ids'])
        assert np.array_equal(z['base'],base) and np.array_equal(z['truth'],y)
        out=z['clean_file']
        for name,v in r['compared'].items():
            scope=geometry(held,v['panel'],v['width'])
            for rule in ('fixed','tp','fp'):
                q=z[name+'_'+rule]
                assert np.array_equal(q[~scope],base[~scope])
                r1=f.comparison(y,base,q)
                assert all(v[rule]['held'][k]==a for k,a in r1.items())
                if rotation=='official':
                    clean[name+'/'+rule]=f.comparison(y[out],base[out],q[out])
                if rotation!='official' and rule!='fixed':
                    exact=z[name+'_same_'+rule]
                    assert np.array_equal(exact[~scope],base[~scope])
                    for p,m in owners.items():
                        receipt=f.comparison(y[m],base[m],exact[m])
                        assert receipt['delta_'+rule]==0 and receipt==v['matched'][rule]['projects'][p]
    return r,checksum(path),clean,int(out.sum())


def main():
    DEST.mkdir(exist_ok=True)
    packs={p:inspect(p) for p in PARTS}
    A,B,O=(packs[p][0]['compared'] for p in PARTS)
    clean=packs['official'][2]
    rows=[]
    for key,v in O.items():
        arm=v['arm']
        if arm=='reference' or key not in A or key not in B:
            continue
        panel,width,_,view,mix=key.split('/')
        root=f'{panel}/{width}'
        ctl=[root+'/reference/threshold/0']
        if arm!='pooled':
            ctl.append(f'{root}/pooled/{view}/{mix}')
        if arm=='span':
            ctl.extend((f'{root}/local/{view}/{mix}',f'{root}/sham/{view}/{mix}'))
        # Both project groups preserve their original error budget. Compare to
        # head controls and threshold movement at the same retained scope.
        a,b=[differences(r,key,ctl) for r in (A,B)]
        def improved(which):
            u,t=a[which][1],b[which][1]
            return min(u,t)>=0 and u+t>0
        domains=improved(0) or improved(1)
        for rule in ('fixed','tp','fp'):
            calibrate,review=v[rule]['cal'],v[rule]['held']
            ctrl=max([0.] + [f1(O[i][rule]['cal']) for i in ctl])
            evidence=clean[key+'/'+rule]
            rows.append(dict(key=key,arm=arm,view=view,panel=panel,width=v['width'],mix=v['mix'],rule=rule,
                             cut=v[rule]['cut'],a_tp=a[0][0],b_tp=b[0][0],a_over_tp=a[0][1],b_over_tp=b[0][1],
                             a_removed_fp=a[1][0],b_removed_fp=b[1][0],a_over_fp=a[1][1],b_over_fp=b[1][1],
                             cal_f1=f1(calibrate),cal_over=f1(calibrate)-ctrl,review_f1=f1(review),
                             review_tp=review['delta_tp'],review_fp=review['delta_fp'],
                             file_held_tp=evidence['delta_tp'],file_held_fp=evidence['delta_fp'],
                             qualified=bool(domains and f1(calibrate)>ctrl and f1(review)>0. and f1(evidence)>=0.)))
    rows.sort(key=lambda r:(r['qualified'],min(r['cal_f1'],r['review_f1']),r['cal_over']),reverse=True)
    best={kind:next((v for v in rows if v['arm']==kind and v['qualified']),None) for kind in ARMS}
    table=DEST/'comparison.csv'
    with table.open('w',newline='') as output:
        w=csv.DictWriter(output,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(DEST/'selection.json',dict(selected=best,tested=len(rows),qualified=sum(z['qualified'] for z in rows),
              file_held=packs['official'][3],sources={k:v[1] for k,v in packs.items()},
              examples=rows[:12],non_overlap='development and project file-held flags from training matrices',
              csv_sha256=checksum(table),code_sha256=checksum(Path(__file__))))
    print('tested',len(rows),'qualified',sum(z['qualified'] for z in rows),best,flush=True)


if __name__=='__main__':
    main()
