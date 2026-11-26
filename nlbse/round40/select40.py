"""Select expert policies with signed held-out and internal comparisons."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=('test_a','test_b','official')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verified():
    results={}
    hashes={}
    proof=read(ROOT/'round38/selection.json')
    for role in PARTS:
        folder=HERE/'comparison'/role
        item=folder/'results.json'
        data=read(item)
        assert data['rotation']==role
        assert data['sources']['release_options_sha256']==proof['sources'][role]
        assert data['weights_sha256']==sha(folder/'models.joblib')
        assert data['predictions_sha256']==sha(folder/'predictions.npz')
        assert data['code_sha256']==sha(HERE/'stack40.py')
        assert data['matrix_sha256']==sha(HERE/'data40.py')
        bank=HERE/'sparse'/role
        assert data['sources']['new_scores_sha256']==sha(bank/'scores.npz')
        assert read(bank/'receipt.json')['weights_sha256']==sha(bank/'weights.joblib')
        results[role]=data['options']
        hashes[role]=sha(item)
    return results,hashes


def change(d):
    return d['f1_after']-d['f1_before']


def at(d,regime):
    x=d['matched'][regime]['change']
    return x['delta_tp'] if regime=='fp' else -x['delta_fp']


def select(inputs):
    a,b,d=(inputs[k] for k in PARTS)
    rows=[]
    for key,c in d.items():
        if not c['mix'] or key not in a or key not in b:
            continue
        control=key.rsplit('/',1)[0]+'/0'
        t=[at(k[key],'fp') for k in (a,b)]
        fp=[at(k[key],'tp') for k in (a,b)]
        dt=[at(k[control],'fp') for k in (a,b)]
        df=[at(k[control],'tp') for k in (a,b)]
        tt=[x-y for x,y in zip(t,dt,strict=True)]
        ff=[x-y for x,y in zip(fp,df,strict=True)]
        best=any(min(z)>=0 and sum(z)>0 for z in (tt,ff))
        for rule in ('fixed','fp','tp'):
            x=c[rule]
            qa,qb=x['cal'],x['held']
            cal,review=change(qa),change(qb)
            dc=cal-change(d[control][rule]['cal'])
            eligible=best and cal>0. and dc>0. and review>=0.
            rows.append(dict(key=key,experts=c['experts'],model=c['model'],panel=c['panel'],width=c['width'],
                             mix=c['mix'],rule=rule,cut=x['cut'],cal_rows=c['cal_rows'],held_rows=c['held_rows'],
                             a_tp=t[0],b_tp=t[1],a_fp=fp[0],b_fp=fp[1],
                             a_tp_over=tt[0],b_tp_over=tt[1],a_fp_over=ff[0],b_fp_over=ff[1],
                             cal_f1=cal,review_f1=review,cal_over=dc,cal_tp=qa['delta_tp'],cal_fp=qa['delta_fp'],
                             review_tp=qb['delta_tp'],review_fp=qb['delta_fp'],eligible=eligible))
    return rows


def main():
    data,proof=verified()
    rows=select(data)
    candidates=[x for x in rows if x['eligible']]
    prioritize=lambda x:(min(x['cal_f1'],x['review_f1']),x['cal_f1'],x['review_f1'],-x['cal_rows'])
    best=lambda x:max(x,key=prioritize,default=None)
    groups={name:best([x for x in candidates if x['experts']==name])
            for name in ('score','old','new','all')}
    picked=dict(chosen=best(candidates),experts=groups,
                switch=best([x for x in candidates if x['model'] in ('tree','switch')]),
                stack=best([x for x in candidates if x['model']=='stack']),
                new_signal=best([x for x in candidates if x['experts'] in ('new','all')]))
    path=HERE/'comparison.csv'
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report=dict(picked=picked,tested=len(rows),eligible=len(candidates),source=proof,
                source_sha256=sha(Path(__file__)),csv_sha256=sha(path),
                archive_sha256=read(ROOT/'round38/release_summary.json')['archive_sha256'])
    (HERE/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
