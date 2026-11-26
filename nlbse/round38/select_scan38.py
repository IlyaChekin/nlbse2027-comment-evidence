"""Verify the full-scanner pilot and compare it against matched first-hit controls."""
import csv
import json

from select38 import HERE, ROOT, PARTS, read, sha, delta


def main():
    stage = {}
    proofs = {}
    plan = read(HERE/'selection.json')
    for part in PARTS:
        folder = HERE/'fullscan'/part
        path = folder/'results.json'
        row = read(path)
        assert row['code_sha256']==sha(HERE/'scan38.py')
        assert row['predictions_sha256']==sha(folder/'predictions.npz')
        assert row['weights_sha256']==sha(folder/'models.joblib')
        first = HERE/'comparison'/part/'results.json'
        assert sha(first)==plan['sources'][part]
        assert row['baseline_sha256']==read(first)['base_sha256'] and row['sources']==read(first)['sources']
        stage[part] = row['options']
        proofs[part] = sha(path)
    a,b,official = (stage[k] for k in PARTS)
    rows = []
    for key,v in official.items():
        arm = v['arm']
        if '-' not in arm:
            continue
        control = key.rsplit('/',1)[0]+'/'+arm.split('-')[1]
        total = {'tp':[],'fp':[]}
        for table in (a,b):
            for metric in ('tp','fp'):
                budget = 'fp' if metric=='tp' else 'tp'
                x=table[key]['matched'][budget]['change']['delta_'+metric]
                y=table[control]['matched'][budget]['change']['delta_'+metric]
                total[metric].append(x-y if metric=='tp' else y-x)
        for rule in ('fixed','fp','tp'):
            baseline = official[control][rule]
            row=v[rule]
            cal,review = delta(row['cal']),delta(row['held'])
            gain,transfer = cal-delta(baseline['cal']),review-delta(baseline['held'])
            both=any(min(values)>=0 and sum(values)>0 for values in total.values())
            rows.append(dict(key=key,rule=rule,cal_f1=cal,review_f1=review,cal_over_first=gain,review_over_first=transfer,
                        A_tp=total['tp'][0],B_tp=total['tp'][1],A_fp=total['fp'][0],B_fp=total['fp'][1],
                        selected=both and cal>0 and review>=0 and gain>0. and transfer>0.))
    file=HERE/'fullscan/comparison.csv'
    with file.open('w',newline='') as out:
        writer = csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows,key=lambda r:(r['selected'],min(r['cal_over_first'],r['review_over_first']),r['cal_f1']),reverse=True)
    verdict=dict(tested=len(rows),eligible=sum(r['selected'] for r in rows),best=ranked[:5],
                 sources=proofs,comparison_sha256=sha(file),code_sha256=sha(HERE/'select_scan38.py'))
    (HERE/'fullscan/selection.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':
    main()
