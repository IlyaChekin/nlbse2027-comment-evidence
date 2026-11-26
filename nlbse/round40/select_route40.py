"""Compare acquisitions with the same call and size constraints."""
import csv
import json
from pathlib import Path

from select40 import HERE, ROOT, PARTS, sha, read, at, change


def main():
    plans={}
    hashes={}
    for role in PARTS:
        d=HERE/'routing'/role
        p=d/'results.json'
        report=read(p)
        assert report['code_sha256']==sha(HERE/'route40.py')
        assert report['predictions_sha256']==sha(d/'predictions.npz')
        assert report['weights_sha256']==sha(d/'models.joblib')
        previous=read(HERE/'comparison'/role/'results.json')
        assert report['sources']==previous['sources']
        hashes[role]=sha(p)
        plans[role]=report['values']
        for entry in report['values'].values():
            for t in entry['contracts']:
                assert t['characters']<=t['prior_characters'] and t['changed']%2==0
    a,b,c=(plans[k] for k in PARTS)
    records=[]
    for key,report in c.items():
        expert,plan=key.split('/',1)
        if expert=='score':
            continue
        baseline='score/'+plan
        gain_t=[at(d[key],'fp')-at(d[baseline],'fp') for d in (a,b)]
        gain_f=[at(d[key],'tp')-at(d[baseline],'tp') for d in (a,b)]
        matched=any(min(vals)>=0 and sum(vals)>0 for vals in (gain_t,gain_f))
        for regime in ('fixed','fp','tp'):
            x=report[regime]
            prior=c[baseline][regime]
            cal,held=change(x['cal']),change(x['held'])
            amount=cal-change(prior['cal'])
            improvement=matched and cal>0. and amount>0. and held>=0.
            records.append(dict(key=key,method=regime,cal_f1=cal,review_f1=held,cal_over=amount,
                  a_tp=gain_t[0],b_tp=gain_t[1],a_fp=gain_f[0],b_fp=gain_f[1],eligible=improvement,
                  cal_tp=x['cal']['delta_tp'],cal_fp=x['cal']['delta_fp'],
                  held_tp=x['held']['delta_tp'],held_fp=x['held']['delta_fp']))
    file=HERE/'routing/comparison.csv'
    with file.open('w',newline='') as out:
        csv_=csv.DictWriter(out,fieldnames=records[0].keys())
        csv_.writeheader()
        csv_.writerows(records)
    ordered=sorted(records,key=lambda x:(x['eligible'],min(x['cal_f1'],x['review_f1']),x['cal_over']),reverse=True)
    note=dict(tested=len(records),eligible=sum(x['eligible'] for x in records),top=ordered[:5],receipts=hashes,
              compare_sha256=sha(file),code_sha256=sha(Path(__file__)))
    (HERE/'routing/selection.json').write_text(json.dumps(note,indent=2)+'\n')
    print(note,flush=True)


if __name__=='__main__':
    main()
