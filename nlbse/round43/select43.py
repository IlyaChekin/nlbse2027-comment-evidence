"""Freeze structural policies only after matched project and development comparisons."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEST = HERE/'comparison'
PARTS = ('test_a','test_b','official')
CONTROLS = {'text':('geometry',),'masked':('geometry',),'center':('geometry',),
            'ast':('geometry','masked','text'),'ast_text':('geometry','text'),
            'ast_masked':('geometry','masked')}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def budget(row,rule):
    q = row['matched'][rule]['change']
    return q['delta_tp'] if rule=='fp' else -q['delta_fp']


def change(row,stage):
    q = row[stage]
    return q['f1_after']-q['f1_before']


def main():
    proofs = {}
    options = {}
    for name in PARTS:
        path = DEST/name/'results.json'
        proofs[name] = sha(path)
        obj = load(path)
        assert obj['code_sha256']==sha(HERE/'study43.py')
        assert obj['predictions_sha256']==sha(path.with_name('predictions.npz'))
        assert obj['weights_sha256']==sha(path.with_name('weights.joblib'))
        assert obj['training']['baseline_sha256']==load(ROOT/'round38/selection.json')['sources'][name]
        assert obj['training']['cache_sha256']==sha(HERE/'features/receipt.json')
        options[name] = obj['options']
    a,b,d = (options[part] for part in PARTS)
    rows = []
    for key in sorted(set(a)&set(b)&set(d)):
        v = d[key]
        if v['arm']=='threshold':
            continue
        head,mix = v['arm'],v['mix']
        kind,loss = head.rsplit('_',1)
        scope = key.rsplit('/',2)[0]
        threshold = scope+'/threshold/0'
        checks = [threshold,*[f'{scope}/{ref}_{loss}/{mix:g}' for ref in CONTROLS.get(kind,())]]
        perf = {}
        for rule in ('fp','tp'):
            numbers = [budget(x[key],rule) for x in (a,b)]
            # Keep the strongest matched control separately on each held-out project group.
            peaks = [max(budget(x[c],rule) for c in checks) for x in (a,b)]
            perf[rule] = numbers,[n-o for n,o in zip(numbers,peaks,strict=True)]
        holds = any(min(diff)>=0 and sum(diff)>0 for _,diff in perf.values())
        for rule in ('fixed','fp','tp'):
            c,h = (change(v[rule],k) for k in ('cal','held'))
            min_c = max(0.,*[change(d[x][rule],'cal') for x in checks])
            rows.append(dict(key=key,kind=kind,loss=loss,mode=head,panel=v['panel'],width=v['width'],
                       mix=mix,rule=rule,threshold=v[rule]['cut'],cal_rows=v['cal_rows'],review_rows=v['review_rows'],
                       a_fp_removed=perf['tp'][0][0],b_fp_removed=perf['tp'][0][1],
                       a_fp_over_controls=perf['tp'][1][0],b_fp_over_controls=perf['tp'][1][1],
                       a_tp=perf['fp'][0][0],b_tp=perf['fp'][0][1],
                       a_tp_over_controls=perf['fp'][1][0],b_tp_over_controls=perf['fp'][1][1],
                       cal_f1=c,review_f1=h,cal_over_controls=c-min_c,
                       review_fp=v[rule]['held']['delta_fp'],review_tp=v[rule]['held']['delta_tp'],
                       project_gate=holds,eligible=holds and c>min_c and h>=0.))
    table = DEST/'comparison.csv'
    with table.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    rank = lambda x:(min(x['cal_f1'],x['review_f1']),x['cal_f1']+x['review_f1'],-x['review_rows'])
    result = {}
    for arm in sorted(set(r['kind'] for r in rows)):
        cases = [r for r in rows if r['kind']==arm]
        chosen = max((r for r in cases if r['eligible']),key=rank,default=None)
        matched = max((r for r in cases if r['project_gate']),key=rank,default=None)
        best = max(cases,key=lambda r:(min(r['a_fp_over_controls'],r['b_fp_over_controls']),
                         r['a_fp_over_controls']+r['b_fp_over_controls'],rank(r)))
        result[arm] = dict(tested=len(cases),passed=sum(r['eligible'] for r in cases),chosen=chosen,
                           matched=matched,best_same_tp=best)
    eligible = [r for r in rows if r['eligible']]
    ast = [r for r in eligible if r['kind'].startswith('ast')]
    choose = lambda group:max(group,key=rank,default=None)
    release = load(ROOT/'round38/release_summary.json')
    out = dict(rows=len(rows),arms=result,chosen=choose(eligible),chosen_ast=choose(ast),sources=proofs,
               release_sha256=release['archive_sha256'],csv_sha256=sha(table),code_sha256=sha(Path(__file__)))
    (DEST/'selection.json').write_text(json.dumps(out,indent=2)+'\n')
    print('tested',out['rows'],'chosen',out['chosen'],'ast',out['chosen_ast'])


if __name__=='__main__':
    main()
