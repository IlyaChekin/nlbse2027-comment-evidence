"""Verify annotation policies and compare against identical-feature controls."""
import csv
from pathlib import Path

import numpy as np

from decompose49 import HERE, ROOT, DEST, ANNOTATIONS, PARTS, f, checksum, json_at, load, write_json
from source41 import parts
from select49 import tally

OUT=HERE/'component_comparison'


def study(rotation, label_path):
    source,_,stamp=load(rotation)
    cal,held=parts(source)
    receipt=DEST/rotation/'results.json'
    data=json_at(receipt)
    assert data['matrix_sha256']==stamp
    assert data['code_sha256']==checksum(HERE/'decompose49.py')
    assert data['source_sha256']==checksum(label_path.with_name('receipt.json'))
    assert data['weights_sha256']==checksum(receipt.with_name('models.joblib'))
    prediction=receipt.with_name('predictions.npz')
    assert data['predictions_sha256']==checksum(prediction)
    y=held['y']
    base=held['signed']>0.
    groups={k:held['project']==k for k in np.unique(held['project'])}
    with np.load(prediction) as x:
        assert np.array_equal(x['ids'],held['ids']) and np.array_equal(x['cal_ids'],cal['ids'])
        assert np.array_equal(x['base'],base) and np.array_equal(x['truth'],y)
        assert not np.intersect1d(x['ids'],x['cal_ids']).size
        for k,val in data['results'].items():
            for cut in ('fixed','fp','tp'):
                expected=val[cut]['held']
                candidate=x[k+'_'+cut]
                counts=f.comparison(y,base,candidate)
                counts.update(recovered=int((y&~base&candidate).sum()),lost=int((y&base&~candidate).sum()))
                assert counts==expected
                if rotation!='official' and cut!='fixed':
                    pred=x[k+'_at_'+cut]
                    checks=val['matched'][cut]
                    assert f.comparison(y,base,pred)==checks['change']
                    index={'fp':1,'tp':0}[cut]
                    for zone,m in groups.items():
                        assert tally(y[m],pred[m])[index]==tally(y[m],base[m])[index]
    with np.load(label_path) as annotation:
        details,eligible=annotation['component'][held['ids']],annotation['selected'][held['ids']]
        assert np.array_equal(annotation['truth'][held['ids']],y)
    labels={'rule':eligible&(details==1), 'rule_and_external':eligible&((details&1)>0)&((details&6)>0),
            'external_only':eligible&((details&1)==0)&((details&6)>0),'unmatched':~eligible&y}
    assert np.array_equal(np.logical_or.reduce(list(labels.values())),y)
    coverage={k:dict(rows=int(m.sum()),fn=int((m & ~base).sum()),
                     fn_unqueried=int((m & ~base & ~held['old']).sum())) for k,m in labels.items()}
    return data['results'],coverage,checksum(receipt)


def improve(result):
    return result['f1_after']-result['f1_before']


def row_info(q,e,a,b,c):
    panel,width,feat,method,mix=q.split('/')
    assert method=='conditional'
    controls=[f'{panel}/{width}/{feat}/direct_all/{mix}',f'{panel}/{width}/{feat}/direct_matched/{mix}',
              f'{panel}/{width}/numeric/threshold/0']
    if feat!='numeric':
        controls.extend(f'{panel}/{width}/numeric/{kind}/{mix}' for kind in ('direct_all','direct_matched'))
    compared=[]
    for split in (a,b):
        target=split[q]
        diff={}
        for mode,criterion,factor in [('fp','tp',1),('tp','fp',-1)]:
            to=factor*target['matched'][mode]['change']['delta_'+criterion]
            prior=max([0]+[factor*split[n]['matched'][mode]['change']['delta_'+criterion] for n in controls])
            diff[mode]=(to,to-prior)
        compared.append(diff)
    for proto in ('fixed','fp','tp'):
        cal=e[proto]['calibration']
        held=e[proto]['held']
        gain=improve(cal)
        effort=improve(held)
        excess=gain-max([0.]+[improve(c[k][proto]['calibration']) for k in controls])
        rel=[[r[m][1] for r in compared] for m in ('fp','tp')]
        project_support=any(min(q)>=0 and sum(q)>0 for q in rel)
        yield dict(key=q,arm=feat,panel=panel,width=e['width'],mix=e['mix'],rule=proto,cut=e[proto]['cut'],
                  cal_rows=e['cal_rows'],review_rows=e['held_rows'],
                  a_tp=compared[0]['fp'][0],b_tp=compared[1]['fp'][0],a_after_baseline=rel[0][0],b_after_baseline=rel[0][1],
                  a_fewer_fp=compared[0]['tp'][0],b_fewer_fp=compared[1]['tp'][0],a_extra_fewer_fp=rel[1][0],
                  b_extra_fewer_fp=rel[1][1],cal_f1=gain,review_f1=effort,cal_vs_prior=excess,
                  cal_tp=cal['delta_tp'],cal_fp=cal['delta_fp'],review_tp=held['delta_tp'],review_fp=held['delta_fp'],
                  eligible=project_support and gain>0. and effort>0. and excess>0.)


def main():
    OUT.mkdir(exist_ok=True)
    coverage,receipts,rows={}, {},{}
    source=ANNOTATIONS/'labels.npz'
    assert checksum(source)==json_at(source.with_name('receipt.json'))['predictions_sha256']
    for rotation in PARTS:
        rows[rotation],coverage[rotation],receipts[rotation]=study(rotation,source)
    a,b,c=(rows[k] for k in PARTS)
    detail=[row for q,e in c.items() if e['mode']=='conditional' and q in a and q in b for row in row_info(q,e,a,b,c)]
    p=OUT/'comparison.csv'
    with p.open('w',newline='') as handle:
        stream=csv.DictWriter(handle,fieldnames=list(detail[0]))
        stream.writeheader()
        stream.writerows(detail)
    ranks=sorted(detail,key=lambda r:(r['eligible'],min(r['cal_f1'],r['review_f1']),r['cal_vs_prior']),reverse=True)
    ref=json_at(ROOT/'round38/release_summary.json')
    write_json(OUT/'selection.json',dict(rows=len(detail),selection=next((e for e in ranks if e['eligible']),None),
               examples=ranks[:8],coverage=coverage,inputs=receipts,comparison_sha256=checksum(p),
               archive_sha256=ref['archive_sha256'],code_sha256=checksum(Path(__file__))))
    print('selected',next((e for e in ranks if e['eligible']),None),'comparison',len(ranks),flush=True)
    print('better_examples',ranks[:3],flush=True)
    print('evidence',coverage,flush=True)


if __name__=='__main__':
    main()
