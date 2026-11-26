"""Check matched-budget negative prototype results against score-only and teacher controls."""
import csv
import json

from select39 import HERE, ROOT, PARTS, sha, read, f1, budget


def main():
    data={}
    stamp={}
    release=read(ROOT/'round38/selection.json')
    for role in PARTS:
        folder=HERE/'negative'/role
        path=folder/'results.json'
        row=read(path)
        assert row['rotation']==role
        assert row['code_sha256']==sha(HERE/'negative39.py')
        assert row['predictions_sha256']==sha(folder/'predictions.npz')
        assert row['weights_sha256']==sha(folder/'models.joblib')
        assert row['base_sha256']==release['sources'][role]
        data[role]=row
        stamp[role]=sha(path)
    a,b,c=(data[q]['options'] for q in PARTS)
    rows=[]
    for key,row in c.items():
        prefix,kind=key.rsplit('/',1)
        if kind not in ('prototype','both'):
            continue
        baseline=prefix+('/base' if kind=='prototype' else '/teacher')
        gains={r:[budget(d[key],r)-budget(d[baseline],r) for d in (a,b)] for r in ('fp','tp')}
        matched=any(min(t)>=0 and sum(t)>0 for t in gains.values())
        for mode in ('fixed','fp','tp'):
            info=row[mode]
            cal,re=map(f1,(info['cal'],info['held']))
            base=c[baseline][mode]
            gc,gr=cal-f1(base['cal']),re-f1(base['held'])
            eligible=matched and cal>0. and gc>0. and re>=0. and gr>=0.
            rows.append(dict(key=key,rule=mode,delta_cal=cal,delta_review=re,cal_over=gc,review_over=gr,
                             tp_a=gains['fp'][0],tp_b=gains['fp'][1],fp_a=gains['tp'][0],fp_b=gains['tp'][1],
                             eligible=eligible))
    file=HERE/'negative/comparison.csv'
    with file.open('w',newline='') as out:
        table=csv.DictWriter(out,fieldnames=rows[0].keys())
        table.writeheader()
        table.writerows(rows)
    records=sorted(rows,key=lambda x:(x['eligible'],min(x['cal_over'],x['review_over']),x['delta_cal']),reverse=True)
    result=dict(eligible=sum(q['eligible'] for q in rows),tested=len(rows),top=records[:5],
                diagnostics={p:r['diagnostics'] for p,r in data.items()},receipts=stamp,
                csv_sha256=sha(file),code_sha256=sha(HERE/'negative_summary39.py'))
    (HERE/'negative/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
