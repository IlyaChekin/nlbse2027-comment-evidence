"""Evaluate pair supervision under unchanged project false-positive budgets."""
import csv
from itertools import product
from pathlib import Path

import joblib
import numpy as np

from fit42 import HERE, ROOT, PARTS, DEST as FIT, ARMS, checksum, json_at, load, parts, supported, write_json
from evaluate41 import scope, f1, matched
from stack40 import assemble, diagnostic_order, cut, adjusted, per_project

DEST = HERE/'comparison'
PANELS = {'mixed':None,'negative':0,'unobserved':2}
WIDTHS = (.9,2.25,4.,None)
STRENGTHS = (.35,.65,1.)
CONTROLS = {'soft':'label','pair_gold':'label','pair_order':'pair_gold',
            'pair_joint':'soft','pair_residual':'soft','soft_mlp':'soft','pair_mlp':'soft_mlp'}
RULES = ('fixed','fp','tp')


def evaluation(rotation):
    matrices,threshold,ticket = load(rotation)
    frames = parts(matrices)
    fitted = FIT/rotation
    file = fitted/'scores.npz'
    receipt = json_at(fitted/'receipt.json')
    assert receipt['matrix_sha256']==ticket
    assert receipt['code_sha256']==checksum(HERE/'fit42.py')
    assert receipt['weights_sha256']==checksum(fitted/'weights.joblib')
    assert receipt['pair_roster_sha256']==checksum(fitted/'pair_roster.npz')
    assert receipt['scores_sha256']==checksum(file)
    old = ROOT/'round40/comparison'/rotation
    prior = json_at(old/'results.json')
    assert prior['weights_sha256']==checksum(old/'models.joblib')
    measured = joblib.load(old/'models.joblib')['calibration']
    experts = [assemble(z,threshold,'all',measured)[3] for z in frames]
    base = [z['signed']>0. for z in frames]
    truth = [z['y'] for z in frames]
    pred = dict(ids=frames[1]['ids'],y=truth[1],base=base[1])
    scores = np.load(file)
    model = {}
    for arm in ARMS:
        result = []
        for role,z in zip(('cal','held'),frames,strict=True):
            support = supported(z)
            assert np.array_equal(scores[role+'_ids'],z['ids'][support])
            full = z['signed'].copy()
            full[support] = scores[role+'_'+arm]
            result.append(full)
        model[arm] = result
    report = {}
    for kind,panel,width in product(('open','lexical'),PANELS,WIDTHS):
        include = [scope(z,e,kind,PANELS[panel],width) for z,e in zip(frames,experts,strict=True)]
        if not all(m.any() for m in include):
            continue
        for arm in ('threshold',*ARMS):
            for strength in ((0.,) if arm=='threshold' else STRENGTHS):
                key = f'{kind}/{panel}/{width}/{arm}/{strength:g}'
                outputs = []
                for j,(z,active) in enumerate(zip(frames,include,strict=True)):
                    p = z['signed'][active]
                    outputs.append(p if arm=='threshold' else (1.-strength)*p+strength*model[arm][j][active])
                detail = dict(kind=kind,panel=panel,width=width,arm=arm,strength=strength,
                              n_cal=int(include[0].sum()),n_held=int(include[1].sum()))
                if rotation!='official':
                    order,ties = diagnostic_order(outputs[1],frames[1]['signed'][include[1]],frames[1]['ids'][include[1]])
                    m,d = per_project(truth[1],frames[1]['project'],base[1],include[1],order,('fp','tp'))
                    detail.update(matched=m,ties=ties)
                    pred[key+'_matched'] = d['fp']
                for rule in RULES:
                    edge = cut(truth[0][include[0]],base[0][include[0]],outputs[0],rule)
                    c,_ = adjusted(truth[0],base[0],include[0],outputs[0],edge)
                    h,p = adjusted(truth[1],base[1],include[1],outputs[1],edge)
                    detail[rule] = dict(cut=edge,cal=c,held=h)
                    pred[key+'_'+rule] = p
                report[key] = detail
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',**pred)
    write_json(out/'results.json',dict(variants=report,predictions_sha256=checksum(out/'predictions.npz'),
              fit_receipt_sha256=checksum(fitted/'receipt.json'),round40_calibration_sha256=prior['weights_sha256'],
              code_sha256=checksum(Path(__file__))))
    print(rotation,len(report),flush=True)
    return report


def select(variants):
    a,b,d = [variants[k] for k in PARTS]
    shared = set(a)&set(b)&set(d)
    rows = []
    for key in sorted(shared):
        z = d[key]
        if z['arm']=='threshold':
            continue
        prefix,arm,strength = key.rsplit('/',2)
        threshold = prefix+'/threshold/0'
        ctl = prefix+'/'+CONTROLS[arm]+'/'+strength if arm in CONTROLS else threshold
        stats = {k:[matched(row[key],k) for row in (a,b)] for k in ('fp','tp')}
        gain = {k:[n-matched(row[ctl],k) for n,row in zip(v,(a,b),strict=True)] for k,v in stats.items()}
        over_threshold = {k:[n-matched(row[threshold],k) for n,row in zip(v,(a,b),strict=True)] for k,v in stats.items()}
        winner = any(min(v)>=0 and sum(v)>0 and min(over_threshold[k])>=0 and sum(over_threshold[k])>0
                     for k,v in gain.items())
        for rule in RULES:
            cal = f1(z[rule]['cal'])
            review = f1(z[rule]['held'])
            own_ref = max(0.,f1(d[ctl][rule]['cal']),f1(d[threshold][rule]['cal']))
            rows.append(dict(key=key,control=ctl,arm=arm,kind=z['kind'],panel=z['panel'],width=z['width'],
                             strength=z['strength'],rule=rule,cut=z[rule]['cut'],n_cal=z['n_cal'],n_review=z['n_held'],
                             a_tp=stats['fp'][0],b_tp=stats['fp'][1],a_fp_removed=stats['tp'][0],b_fp_removed=stats['tp'][1],
                             a_tp_over_control=gain['fp'][0],b_tp_over_control=gain['fp'][1],
                             a_fp_over_control=gain['tp'][0],b_fp_over_control=gain['tp'][1],
                             a_tp_over_threshold=over_threshold['fp'][0],b_tp_over_threshold=over_threshold['fp'][1],
                             a_fp_over_threshold=over_threshold['tp'][0],b_fp_over_threshold=over_threshold['tp'][1],
                             cal_f1=cal,review_f1=review,over_cal=cal-own_ref,
                             review_tp=z[rule]['held']['delta_tp'],review_fp=z[rule]['held']['delta_fp'],
                             matched_win=winner,eligible=winner and cal>own_ref and review>=0.))
    return rows


def main():
    variants = {k:evaluation(k) for k in PARTS}
    rows = select(variants)
    table = DEST/'comparison.csv'
    with table.open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    choose = lambda r:(min(r['cal_f1'],r['review_f1']),r['cal_f1'],r['review_f1'],-r['n_review'])
    counts = {}
    for arm in ARMS:
        local = [x for x in rows if x['arm']==arm]
        counts[arm] = dict(n=len(local),eligible=sum(x['eligible'] for x in local),
                           chosen=max((x for x in local if x['eligible']),key=choose,default=None),
                           best_matched=max((x for x in local if x['matched_win']),key=choose,default=None))
    picked = max((x for x in rows if x['eligible']),key=choose,default=None)
    write_json(DEST/'selection.json',dict(points=len(rows),picked=picked,arms=counts,
                          csv_sha256=checksum(table),code_sha256=checksum(Path(__file__)),
                          sources={k:checksum(DEST/k/'results.json') for k in PARTS}))
    print('tested',len(rows),'candidate',picked,flush=True)


if __name__=='__main__':
    main()
