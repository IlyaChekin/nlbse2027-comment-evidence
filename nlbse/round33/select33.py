"""Compare routers on both project splits and the independent internal files."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=('test_a','test_b','official')


def sha(p):
    with p.open('rb') as h:
        return hashlib.file_digest(h,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text())


def run(root=HERE/'comparison', out=HERE):
    table=[]
    proofs={}
    state={}
    for name in PARTS:
        dest=root/name
        path=dest/'results.json'
        rec=read(path)
        assert rec['rotation']==name
        assert sha(HERE/'experiment33.py')==rec['code_sha256']
        assert sha(HERE/'policy33.py')==rec['policy_sha256']
        assert sha(dest/'predictions.npz')==rec['predictions_sha256']
        assert sha(dest/'models.joblib')==rec['models_sha256']
        assert sha(HERE/'points'/name/'receipt.json')==rec['points_sha256']
        proofs[name]=sha(path)
        state[name]=rec
    opts=[state[name]['settings'] for name in PARTS]
    a,b,review=opts
    comparator='released_shared_0_0'
    winner=None
    permitted=[]
    for key,item in review.items():
        assert a.keys()==b.keys()==review.keys()
        mode=item['mode']
        aa,bb=a[key],b[key]
        for node in (aa,bb,item):
            for role in ('calibration','review'):
                usage=node[role]
                assert usage['calls']==usage['old_calls'] and usage['characters']<=usage['old_characters']
        diagnostics={}
        for subset,rec,base in [('a',aa,a[comparator]),('b',bb,b[comparator])]:
            for method in ('matched_fp','matched_tp'):
                sign='delta_tp' if method=='matched_fp' else 'delta_fp'
                opt=rec['options'][method]
                exact=opt['exact']
                value=opt['overall'][sign]
                control=base['options'][method]['overall'][sign]
                gain=(value-control)*(1 if method=='matched_fp' else -1)
                diagnostics.update({subset+'_'+method+'_shift':value,subset+'_'+method+'_gain':gain,
                                    subset+'_'+method+'_exact':exact})
        for criterion in ('fixed','tp','fp'):
            p=item['options'][criterion]
            cal=p['cal']
            held=p['frozen']
            value=cal['f1_after']-cal['f1_before']
            review_gain=held['f1_after']-held['f1_before']
            control=item['options'][criterion]['cal']['f1_before']
            baseline_recal=review[comparator]['options'][criterion]['cal']['f1_after']
            if abs(cal['f1_before']-control)>1e-12:
                raise ValueError('Inconsistent calibration baseline')
            row=dict(key=key,mode=mode,criterion=criterion,source=item['source'],strength=item['strength'],
                     fraction=item['fraction'],cal_gain=value,cal_over_recut=cal['f1_after']-baseline_recal,
                     review_gain=review_gain,cal_tp=cal['delta_tp'],cal_fp=cal['delta_fp'],
                     review_tp=held['delta_tp'],review_fp=held['delta_fp'],
                     a_count=aa['review']['calls'],b_count=bb['review']['calls'],cal_swaps=item['calibration']['swap'],
                     a_swaps=aa['review']['swap'],b_swaps=bb['review']['swap'],threshold=p['cut'],**diagnostics)
            table.append(row)
            if mode=='released' or value<=0 or row['cal_over_recut']<=0 or review_gain<0.:
                continue
            tp=(row['a_matched_fp_gain'],row['b_matched_fp_gain'])
            fp=(row['a_matched_tp_gain'],row['b_matched_tp_gain'])
            valid_tp=row['a_matched_fp_exact'] and row['b_matched_fp_exact'] and min(tp)>=0 and sum(tp)>0
            valid_fp=row['a_matched_tp_exact'] and row['b_matched_tp_exact'] and min(fp)>=0 and sum(fp)>0
            if not (valid_tp or valid_fp):
                continue
            permitted.append(row)
    if permitted:
        winner=max(permitted,key=lambda r:(r['cal_gain'],r['review_gain'],-r['cal_swaps'],r['criterion']=='fixed'))
    out.mkdir(parents=True,exist_ok=True)
    csv_file=out/'table.csv'
    with csv_file.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    result=dict(selected=winner,eligible=len(permitted),count=len(table),tables=proofs,table_sha256=sha(csv_file),
                oracle={name:state[name]['oracle'] for name in PARTS})
    (out/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(eligible=len(permitted),count=len(table),selected=winner,oracle={
          k:v['held']['same_budget'] for k,v in result['oracle'].items()}),indent=2))



if __name__=='__main__':
    run()
