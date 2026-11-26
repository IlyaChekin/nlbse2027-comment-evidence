"""Select a CodeT5+ contribution against calibrated equal-text controls."""
import csv
from itertools import product
from pathlib import Path

import numpy as np

from fit47 import HERE,ROOT,PARTS,DEST as FIT,f,checksum,json_at,load,parts,write_json
from compare46 import matched,f1,estimated_cost
from study44 import scope
from stack40 import diagnostic_order,cut,adjusted,per_project

DEST=HERE/'comparison'
CONTROL={
    'embed96':('numeric','unix96','qwen96','raw96'),
    'embed384':('numeric','unix384','qwen384','raw384'),
    'raw96':('numeric','unix96','qwen96'),
    'raw384':('numeric','unix384','qwen384'),
    'knn_embed96':('numeric','unix96','qwen96','embed96'),
    'knn_embed384':('numeric','unix384','qwen384','embed384','knn_raw384'),
    'knn_raw384':('numeric','unix384','qwen384','raw384'),
    'merge384':('numeric','unix384','qwen384','embed384'),
    'transfer_embed96':('compact','transfer_unix96'),
    'transfer_embed384':('compact','transfer_unix384'),
    'transfer_knn_embed384':('compact','transfer_unix384'),
}
ARMS=tuple(dict.fromkeys((*CONTROL,*(q for v in CONTROL.values() for q in v))))
VIEWS=('all','consensus','veto')
WIDTHS=(.8,2.5,5.,None)


def compare(rotation):
    source,_,ticket=load(rotation)
    frames=parts(source)
    fold=FIT/rotation
    receipt=json_at(fold/'receipt.json')
    path=fold/'scores.npz'
    assert receipt['matrix_sha256']==ticket and receipt['score_sha256']==checksum(path)
    assert receipt['code_sha256']==checksum(HERE/'fit47.py')
    assert receipt['weights_sha256']==checksum(fold/'weights.joblib')
    assert all(v==checksum(HERE/'embeddings'/k/'receipt.json') for k,v in receipt['embeddings'].items())
    bound=ROOT/'round46/length'/rotation/'results.json'
    old=json_at(ROOT/'round46/length/selection.json')
    assert old['sources'][rotation]==checksum(bound)
    long=json_at(bound)['rules']['0.75']['chars']
    raw=np.load(path)
    base=[z['signed']>0. for z in frames]
    y=[z['y'] for z in frames]
    estimates={k:[] for k in ARMS}
    for role,row in zip(('cal','held'),frames,strict=True):
        include=scope(row,'all',None)
        assert np.array_equal(raw[role+'_ids'],row['ids'][include])
        for k in estimates:
            q=row['signed'].copy()
            q[include]=raw[role+'_'+k]
            estimates[k].append(q)
    report={}
    save=dict(ids=frames[1]['ids'],base=base[1],truth=y[1])
    choose=[*product(('ordinary',),VIEWS,WIDTHS),*product(('long',),('all','veto'),(2.5,None))]
    for format,panel,width in choose:
        region=[scope(z,panel,width) for z in frames]
        if format=='long':
            for use,z in zip(region,frames,strict=True):
                use &= z['length']>=long
        if not all(z.any() for z in region):
            continue
        for kind,w in [('threshold',0.),*product(ARMS,(.35,.7))]:
            key=f'{format}/{panel}/{width}/{kind}/{w:g}'
            old=[z['signed'] for z in frames]
            ranks=[(1.-w)*s[m]+w*q[m] for s,q,m in zip(old,estimates.get(kind,old),region,strict=True)]
            table=dict(kind=kind,format=format,panel=panel,width=width,mix=w,review_rows=int(region[1].sum()),
                       cal_rows=int(region[0].sum()),min_chars=long if format=='long' else None)
            if rotation!='official':
                order,_=diagnostic_order(ranks[1],old[1][region[1]],frames[1]['ids'][region[1]])
                v,prediction=per_project(y[1],frames[1]['project'],base[1],region[1],order,('fp','tp'))
                table['matched']=v
                for k in ('fp','tp'):
                    save[key+'_same_'+k]=prediction[k]
            for k in ('fixed','fp','tp'):
                edge=cut(y[0][region[0]],base[0][region[0]],ranks[0],k)
                a,_=adjusted(y[0],base[0],region[0],ranks[0],edge)
                b,p=adjusted(y[1],base[1],region[1],ranks[1],edge)
                table[k]=dict(cut=edge,cal=a,held=b)
                save[key+'_'+k]=p
            report[key]=table
    dest=DEST/rotation
    dest.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(dest/'predictions.npz',**save)
    write_json(dest/'results.json',dict(options=report,code_sha256=checksum(Path(__file__)),
                fit_sha256=checksum(fold/'receipt.json'),technical_length_sha256=checksum(bound),
                predictions_sha256=checksum(dest/'predictions.npz')))
    print(rotation,len(report),flush=True)
    return report


def prices():
    data=estimated_cost()
    for n in (96,384):
        d=json_at(HERE/'embeddings'/str(n)/'receipt.json')
        price=.2*d['cost']['analytic_gflops']/d['rows']/3_000_000
        for k in ('embed','raw','knn_embed','knn_raw'):
            data[k+str(n)]=price
    data['merge384']=data['embed384']+data['unix384']
    return data


def select(a,b,c):
    price=prices()
    table=[]
    for key in sorted(set(a)&set(b)&set(c)):
        row=c[key]
        arm=row['kind']
        if arm not in CONTROL:
            continue
        region,_,w=key.rsplit('/',2)
        indices=[f'{region}/threshold/0',*[f'{region}/{name}/{w}' for name in CONTROL[arm]]]
        summary={}
        for mode in ('fp','tp'):
            plain=[matched(r[key],mode) for r in (a,b)]
            extra=[v-max(matched(r[j],mode) for j in indices) for v,r in zip(plain,(a,b),strict=True)]
            summary[mode]=(plain,extra)
        good=any(min(x)>=0 and sum(x)>0 for _,x in summary.values())
        fee=price.get(arm,0.)*row['review_rows']
        for mode in ('fixed','fp','tp'):
            a0,b0=row[mode]['cal'],row[mode]['held']
            quality=f1(row[mode],'cal')
            hold=f1(row[mode],'held')
            prior=max(0.,max(f1(c[k][mode],'cal') for k in indices))
            eligible=good and quality>prior and hold>=0.
            table.append(dict(key=key,arm=arm,kind='direct' if arm in price else 'small',rule=mode,
                   format=row['format'],panel=row['panel'],width=row['width'],min_chars=row['min_chars'],
                   mix=row['mix'],cut=row[mode]['cut'],cal_rows=row['cal_rows'],review_rows=row['review_rows'],
                   a_tp=summary['fp'][0][0],b_tp=summary['fp'][0][1],
                   a_extra=summary['fp'][1][0],b_extra=summary['fp'][1][1],
                   a_fp_extra=summary['tp'][1][0],b_fp_extra=summary['tp'][1][1],
                   cal_f1=quality,review_f1=hold,cal_over=quality-prior,review_tp=b0['delta_tp'],
                   review_fp=b0['delta_fp'],expected_compute_score_cost=fee,
                   quality=eligible,eligible=eligible and (arm not in price or .3*hold>fee)))
    return table


def main():
    table=select(*(compare(k) for k in PARTS))
    out=DEST/'comparison.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0])
        w.writeheader()
        w.writerows(table)
    merit=lambda r:(min(r['cal_f1'],r['review_f1']),r['cal_f1']+r['review_f1'],-r['review_rows'])
    scored={}
    for arm in CONTROL:
        here=[v for v in table if v['arm']==arm]
        scored[arm]=dict(quality=max((v for v in here if v['quality']),key=merit,default=None),
                         selected=max((v for v in here if v['eligible']),key=merit,default=None),
                         example=max(here,key=lambda q:(min(q['a_extra'],q['b_extra']),
                                                        q['a_extra']+q['b_extra'],merit(q))))
    selection={k:max((v for v in table if v['kind']==k and v['eligible']),key=merit,default=None)
               for k in ('small','direct')}
    base=json_at(ROOT/'round38/release_summary.json')
    write_json(DEST/'selection.json',dict(tested=len(table),selection=selection,by_arm=scored,
        archive_sha256=base['archive_sha256'],csv_sha256=checksum(out),code_sha256=checksum(Path(__file__)),
        sources={k:checksum(DEST/k/'results.json') for k in PARTS}))
    print(len(table),selection,flush=True)


if __name__=='__main__':
    main()
