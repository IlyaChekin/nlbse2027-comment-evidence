"""Audit measured-label probes and any vulnerability improvement they support."""
import csv
from pathlib import Path

import numpy as np

from probe51 import HERE, FIT, AVAIL, PARTS, DEST as PROBE, f, load, checksum, json_at, write_json
from eval51 import geometry
from select51 import differences, f1
from source41 import parts

DEST=HERE/'probe_comparison'


def read(rotation):
    matrices,_,stamp=load(rotation)
    cal,test=parts(matrices)
    filename=PROBE/rotation/'results.json'
    record=json_at(filename)
    assert record['matrix_sha256']==stamp
    assert record['training_sha256']==checksum(FIT/rotation/'fit.json')
    assert record['support_sha256']==checksum(AVAIL/'receipt.json')
    assert record['code_sha256']==checksum(HERE/'probe51.py')
    assert record['weights_sha256']==checksum(filename.with_name('weights.pt'))
    path=filename.with_name('predictions.npz')
    assert record['predictions_sha256']==checksum(path)
    y=test['y']
    base=test['signed']>0.
    scope={}
    with np.load(path) as z:
        assert np.array_equal(z['ids'],test['ids']) and np.array_equal(z['cal_ids'],cal['ids'])
        for key,row in record['results'].items():
            group=(row['panel'],row['width'])
            chosen=scope.setdefault(group,geometry(test,*group))
            for protocol in ('fixed','fp','tp'):
                pred=z[key+'_'+protocol]
                counts=f.comparison(y,base,pred)
                assert all(row[protocol]['held'][k]==v for k,v in counts.items())
                assert np.array_equal(pred[~chosen],base[~chosen])
                if rotation!='official' and protocol!='fixed':
                    pred=z[key+'_same_'+protocol]
                    assert f.comparison(y,base,pred)==row['matched'][protocol]['change']
                    for name,c in row['matched'][protocol]['projects'].items():
                        local=test['project']==name
                        assert f.comparison(y[local],base[local],pred[local])==c
                        assert c['delta_'+protocol]==0
    return record,checksum(filename)


def main():
    DEST.mkdir(exist_ok=True)
    entries,receipts={},{}
    for phase in PARTS:
        entries[phase],receipts[phase]=read(phase)
    a,b,c=(entries[p]['results'] for p in PARTS)
    csvrows=[]
    for path,v in c.items():
        if path not in a or path not in b or v['arm']=='reference':
            continue
        panel,width,backbone,supervision,mix=path.split('/')
        prefix=panel+'/'+width
        controls=[f'{prefix}/reference/threshold/0']
        if supervision=='measured':
            controls.append(f'{prefix}/{backbone}/all_matched/{mix}')
        d,e=[differences(rows,path,controls) for rows in (a,b)]
        same_fp=d[0][1]>=0 and e[0][1]>=0 and d[0][1]+e[0][1]>0
        same_tp=d[1][1]>=0 and e[1][1]>=0 and d[1][1]+e[1][1]>0
        for rule in ('fixed','fp','tp'):
            cal=v[rule]['cal']
            held=v[rule]['held']
            difference=f1(cal)-max([0.]+[f1(c[k][rule]['cal']) for k in controls])
            csvrows.append(dict(key=path,arm=backbone,view=supervision,panel=panel,width=v['width'],mix=v['mix'],
                                rule=rule,cut=v[rule]['cut'],a_tp=d[0][0],b_tp=e[0][0],a_over=d[0][1],b_over=e[0][1],
                                a_removed_fp=d[1][0],b_removed_fp=e[1][0],cal_over=difference,cal_f1=f1(cal),
                                review_f1=f1(held),cal_tp=cal['delta_tp'],cal_fp=cal['delta_fp'],
                                held_tp=held['delta_tp'],held_fp=held['delta_fp'],
                                selected=(same_fp or same_tp) and difference>0. and f1(held)>0.))
    table=DEST/'comparison.csv'
    with table.open('w',newline='') as q:
        w=csv.DictWriter(q,fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
    ranked=sorted(csvrows,key=lambda v:(v['selected'],min(v['cal_f1'],v['review_f1']),v['cal_over']),reverse=True)
    selected={k:next((v for v in ranked if v['selected'] and v['arm']==k),None)
              for k in ('original','components')}
    record=dict(options=len(csvrows),qualified=sum(v['selected'] for v in csvrows),selected=selected,
                examples=ranked[:8],scores={k:v['probes'] for k,v in entries.items()},
                proof=receipts,csv_sha256=checksum(table),code_sha256=checksum(Path(__file__)))
    write_json(DEST/'selection.json',record)
    print(len(csvrows),record['qualified'],selected,flush=True)
    print('examples',ranked[:4],flush=True)


if __name__=='__main__':
    main()
