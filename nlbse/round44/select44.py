"""Select a frozen correction without access to public labels."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEST = HERE/'comparison'
PARTS = ('test_a','test_b','official')
CONTROL = {'lexical':('numeric',),'signature':('numeric','lexical'),'body':('numeric','lexical','middle'),
           'middle':('numeric','lexical'),'usage':('numeric','lexical'),'syntax':('numeric','lexical','middle'),
           'syntax_text':('numeric','lexical','middle')}


def checksum(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def json_at(path):
    return json.loads(path.read_text())


def fp(row,rule):
    e = row['matched'][rule]['change']
    return e['delta_tp'] if rule=='fp' else -e['delta_fp']


def delta(row,part):
    a = row[part]
    return a['f1_after']-a['f1_before']


def main():
    references,options = {},{}
    for part in PARTS:
        base = DEST/part
        p = base/'results.json'
        v = json_at(p)
        assert v['predictions_sha256']==checksum(base/'predictions.npz')
        assert v['weights_sha256']==checksum(base/'weights.joblib')
        assert v['code_sha256']==checksum(HERE/'study44.py')
        assert v['training']['features_sha256']==checksum(HERE/'features/receipt.json')
        references[part] = checksum(p)
        options[part] = v['options']
    a,b,c = (options[r] for r in PARTS)
    results = []
    for key in sorted(set(a)&set(b)&set(c)):
        q = c[key]
        if q['arm']=='threshold':
            continue
        label,weight = key.rsplit('/',2)[1:]
        region = key.rsplit('/',2)[0]
        view,kind = label.rsplit('_',1)
        controls = [f'{region}/threshold/0',*[f'{region}/{v}_{kind}/{weight}' for v in CONTROL.get(view,())]]
        prior = f'{region}/threshold/0'
        by_cut = {}
        for rule in ('fp','tp'):
            values = [fp(x[key],rule) for x in (a,b)]
            relative = [v-max(fp(x[k],rule) for k in controls) for v,x in zip(values,(a,b),strict=True)]
            control = [v-fp(x[prior],rule) for v,x in zip(values,(a,b),strict=True)]
            by_cut[rule] = values,relative,control
        same_fp = by_cut['fp']
        same_tp = by_cut['tp']
        stable = min(same_fp[1])>=0 and sum(same_fp[1])>0
        for rule in ('fixed','fp','tp'):
            cal = delta(q[rule],'cal')
            held = delta(q[rule],'held')
            over = cal-max(0.,max(delta(c[k][rule],'cal') for k in controls))
            results.append(dict(key=key,arm=label,model=kind,view=view,panel=q['panel'],width=q['width'],
                        mix=q['mix'],cut=q[rule]['cut'],rule=rule,cal_rows=q['cal_rows'],review_rows=q['review_rows'],
                        a_tp=same_fp[0][0],b_tp=same_fp[0][1],a_extra=same_fp[1][0],b_extra=same_fp[1][1],
                        a_extra_over_threshold=same_fp[2][0],b_extra_over_threshold=same_fp[2][1],
                        a_fp_removed=same_tp[0][0],b_fp_removed=same_tp[0][1],
                        cal_f1=cal,over_cal_f1=over,review_f1=held,
                        review_tp=q[rule]['held']['delta_tp'],review_fp=q[rule]['held']['delta_fp'],
                        cal_tp=q[rule]['cal']['delta_tp'],cal_fp=q[rule]['cal']['delta_fp'],
                        projects_pass=stable,eligible=stable and over>0. and held>=0.))
    table = DEST/'comparison.csv'
    with table.open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    scores = lambda r:(min(r['cal_f1'],r['review_f1']),r['cal_f1']+r['review_f1'],-r['review_rows'])
    acceptable = [r for r in results if r['eligible']]
    report = {}
    for view in sorted(set(r['view'] for r in results)):
        own = [r for r in results if r['view']==view]
        choose = max((r for r in own if r['eligible']),key=scores,default=None)
        diagnostic = max(own,key=lambda r:(min(r['a_extra'],r['b_extra']),r['a_extra']+r['b_extra'],scores(r)))
        report[view] = dict(tested=len(own),passing=sum(r['eligible'] for r in own),chosen=choose,best_matched=diagnostic)
    selected = max(acceptable,key=scores,default=None)
    out = dict(tested=len(results),eligible=len(acceptable),chosen=selected,arms=report,sources=references,
               control_sha256=json_at(ROOT/'round38/release_summary.json')['archive_sha256'],
               csv_sha256=checksum(table),code_sha256=checksum(Path(__file__)))
    (DEST/'selection.json').write_text(json.dumps(out,indent=2)+'\n')
    print('tested',len(results),'selected',selected,flush=True)


if __name__=='__main__':
    main()
