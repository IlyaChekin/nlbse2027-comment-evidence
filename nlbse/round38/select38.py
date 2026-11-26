"""Freeze a two-branch comparison before inspecting the public predictions."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARTS = ('test_a','test_b','official')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def inputs():
    result,receipts = {}, {}
    previous = read(ROOT/'round37/selection.json')
    for role in PARTS:
        path = HERE/'comparison'/role
        report = path/'results.json'
        q = read(report)
        assert q['rotation']==role and q['base_sha256']==previous['source_table'][role]
        assert q['predictions_sha256']==sha(path/'predictions.npz')
        assert q['weights_sha256']==sha(path/'models.joblib')
        assert q['code_sha256']==sha(HERE/'rank38.py')
        for branch in ('negative','positive'):
            folder = HERE/f'{branch}_teacher'/role
            p = folder/'receipt.json'
            proof = read(p)
            assert q['sources'][branch]==sha(p)
            assert proof['scores_sha256']==sha(folder/'scores.npz')
        result[role] = q['options']
        receipts[role] = sha(report)
    return result,receipts


def delta(row):
    return row['f1_after']-row['f1_before']


def summary(result):
    a,b,c = (result[x] for x in PARTS)
    rows = []
    for key,item in c.items():
        if key not in a or key not in b or not item['mix']:
            continue
        control = key.rsplit('/',1)[0]+'/0'
        scope = item['arms']
        t = []
        fp = []
        tp_gain = []
        fp_gain = []
        for compared in (a,b):
            x,z = (compared[k]['matched'] for k in (key,control))
            at_fp = x['fp']['change']
            at_tp = x['tp']['change']
            tp_gain.append(at_fp['delta_tp']-z['fp']['change']['delta_tp'])
            fp_gain.append(z['tp']['change']['delta_fp']-at_tp['delta_fp'])
            t.append(at_fp['delta_tp'])
            fp.append(-at_tp['delta_fp'])
        kind = ('joint' if len([w for w in scope.values() if w!='off'])==2 else
                next(k for k,v in scope.items() if v!='off'))
        for rule in ('fixed','fp','tp'):
            selected = item[rule]
            cal,review = selected['cal'],selected['held']
            rows.append(dict(key=key,rule=rule,negative=scope['negative'],positive=scope['positive'],
                     width_negative=item['widths'][0],width_positive=item['widths'][1],kind=kind,mix=item['mix'],
                     cal_rows=item['cal_rows'],held_rows=item['held_rows'],cut=selected['cut'],
                     cal_f1=delta(cal),held_f1=delta(review),cal_tp=cal['delta_tp'],cal_fp=cal['delta_fp'],
                     held_tp=review['delta_tp'],held_fp=review['delta_fp'],
                     cal_over=delta(cal)-delta(c[control][rule]['cal']),
                     a_tp=t[0],b_tp=t[1],a_fp=fp[0],b_fp=fp[1],
                     a_tp_over=tp_gain[0],b_tp_over=tp_gain[1],a_fp_over=fp_gain[0],b_fp_over=fp_gain[1],
                     deployable=not any(w.startswith('teacher') for w in scope.values())))
    return rows


def eligible(x):
    fr = x['a_tp_over'],x['b_tp_over']
    pr = x['a_fp_over'],x['b_fp_over']
    agree = (min(fr)>=0 and sum(fr)>0) or (min(pr)>=0 and sum(pr)>0)
    return agree and x['cal_f1']>0. and x['cal_over']>0. and x['held_f1']>=0.


def main():
    collection,proof = inputs()
    rows = summary(collection)
    work = [a for a in rows if eligible(a)]
    prioritize = lambda r:(min(r['cal_f1'],r['held_f1']),r['cal_f1'],r['held_f1'],-r['cal_rows'])
    best = {}
    for kind in ('negative','positive','joint'):
        for family in ('cheap','teacher'):
            pool = [r for r in work if r['kind']==kind and r['deployable']==(family=='cheap')]
            best[kind+'_'+family] = max(pool,key=prioritize,default=None)
    winner = max((r for r in work if r['deployable']),key=prioritize,default=None)
    file = HERE/'comparison.csv'
    with file.open('w',newline='') as out:
        csv_ = csv.DictWriter(out,fieldnames=rows[0].keys())
        csv_.writeheader()
        csv_.writerows(rows)
    report = dict(selected=winner,arms=best,tested=len(rows),eligible=len(work),
                  releases=read(ROOT/'round37/release_summary.json')['archive_sha256'],
                  sources=proof,csv_sha256=sha(file),code_sha256=sha(Path(__file__)))
    (HERE/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
