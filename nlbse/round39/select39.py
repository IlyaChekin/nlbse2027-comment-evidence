"""Freeze a localized correction on calibration, review and held-out projects."""
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


def inputs():
    results={}
    receipts={}
    previous=read(ROOT/'round38/selection.json')
    scan=HERE/'locations/receipt.json'
    proof=read(scan)
    assert proof['extended_scanner_sha256']==sha(HERE/'scan_location.mjs')
    assert proof['original_scanner_sha256']==sha(ROOT/'round32/scan_first.mjs')
    assert proof['predictions_sha256']==sha(scan.with_name('training.npz'))
    assert proof['code_sha256']==sha(HERE/'location39.py')
    for row in PARTS:
        d=HERE/'comparison'/row
        p=d/'results.json'
        r=read(p)
        assert r['rotation']==row
        assert r['geometry_sha256']==sha(scan)
        assert r['base_sha256']==previous['sources'][row]
        assert r['model_sha256']==sha(d/'models.joblib')
        assert r['predictions_sha256']==sha(d/'predictions.npz')
        for key,n in [('code','study39.py'),('context','context39.py'),('data','data39.py')]:
            assert r[key+'_sha256']==sha(HERE/n)
        results[row]=r['options']
        receipts[row]=sha(p)
    return results,receipts


def f1(r):
    return r['f1_after']-r['f1_before']


def budget(p, rule):
    r=p['matched'][rule]['change']
    return r['delta_tp'] if rule=='fp' else -r['delta_fp']


def main():
    stages,proof=inputs()
    a,b,c=(stages[q] for q in PARTS)
    rows=[]
    for key,trial in c.items():
        mode=trial['mode']
        if not trial['mix'] or key not in a or key not in b:
            continue
        head=key.rsplit('/',2)[0]
        first='base-pair' if mode.endswith('-pair') else 'base'
        base=f"{head}/{first}/{trial['mix']:g}"
        threshold=head+'/threshold/0'
        control=base if mode not in ('base','base-pair') else threshold
        for regime in ('fixed','tp','fp'):
            cal,held=(trial[n+'_'+regime] for n in ('cal','held'))
            reference=c[control]
            cal_delta,review_delta=f1(cal),f1(held)
            improve_cal=cal_delta-f1(reference['cal_'+regime])
            improve_review=review_delta-f1(reference['held_'+regime])
            at_fp=[budget(x[key],'fp') for x in (a,b)]
            at_tp=[budget(x[key],'tp') for x in (a,b)]
            gain_t=[u-budget(x[control],'fp') for u,x in zip(at_fp,(a,b),strict=True)]
            gain_f=[u-budget(x[control],'tp') for u,x in zip(at_tp,(a,b),strict=True)]
            wins=any(min(v)>=0 and sum(v)>0 for v in (gain_t,gain_f))
            good=(wins and cal_delta>0. and review_delta>=0. and
                  (improve_cal>0. or (improve_cal>=0. and improve_review>0.)) and improve_review>=0.)
            rows.append(dict(key=key,mode=mode,rule=regime,panel=trial['panel'],width=trial['width'],
                             mix=trial['mix'],cut=trial['cut_'+regime],cal_rows=trial['cal_rows'],held_rows=trial['held_rows'],
                             cal_f1=cal_delta,review_f1=review_delta,cal_over=improve_cal,review_over=improve_review,
                             A_tp=at_fp[0],B_tp=at_fp[1],A_fp=at_tp[0],B_fp=at_tp[1],
                             A_tp_over=gain_t[0],B_tp_over=gain_t[1],A_fp_over=gain_f[0],B_fp_over=gain_f[1],
                             cal_tp=cal['delta_tp'],cal_fp=cal['delta_fp'],
                             review_tp=held['delta_tp'],review_fp=held['delta_fp'],eligible=good))
    file=HERE/'comparison.csv'
    with file.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    pick=lambda r:(min(r['cal_f1'],r['review_f1']),r['cal_over']+r['review_over'],-r['cal_rows'])
    found=[r for r in rows if r['eligible']]
    cheapest=[r for r in found if r['mode'].startswith(('coordinate','layout'))]
    top=dict(chosen=max([r for r in found if not r['mode'].startswith('base')],key=pick,default=None),
             best_numeric=max(cheapest,key=pick,default=None),
             best_local=max([r for r in found if r['mode'].startswith('local')],key=pick,default=None),
             best_center=max([r for r in found if r['mode'].startswith('center')],key=pick,default=None),
             best_control=max([r for r in found if r['mode'].startswith('base')],key=pick,default=None))
    records=dict(top,options=len(rows),eligible=len(found),receipts=proof,code_sha256=sha(Path(__file__)),
                 comparison_sha256=sha(file),archive_sha256=read(ROOT/'round38/release_summary.json')['archive_sha256'])
    (HERE/'selection.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
