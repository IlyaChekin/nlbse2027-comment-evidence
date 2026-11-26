"""Compare strong and distilled channels against matched threshold and head controls."""
from itertools import product
import csv
from pathlib import Path

import joblib
import numpy as np

from source41 import HERE, ROOT, PARTS, checksum, json_at, write_json, load, supported, parts
from heads41 import DEST as HEADS
from stack40 import assemble, diagnostic_order, cut, adjusted, per_project

DEST = HERE/'comparison'
ARMS = ('threshold','score','compact','distill_graph','distill_unix','distill_average','graph','unix','average')
DIRECT = ('graph','unix','average')
WIDTHS = (.9,2.25,None)
SCOPES = {'mixed':None,'negative':0,'unobserved':2}


def scope(frame,expert,kind,panel,width):
    mask = supported(frame)
    if panel is not None:
        mask &= frame['observed_regime']==panel
    if width is not None:
        mask &= abs(frame['signed'])<width
    if kind=='lexical':
        mask &= (expert[:,3]>0.)!=(expert[:,4]>0.)
    return mask


def run(rotation):
    matrices,threshold,ticket = load(rotation)
    frames = parts(matrices)
    folder = HEADS/rotation
    receipt = json_at(folder/'receipt.json')
    assert receipt['matrix_sha256']==ticket
    assert receipt['code_sha256']==checksum(HERE/'heads41.py')
    assert receipt['predictions_sha256']==checksum(folder/'scores.npz')
    assert receipt['model_sha256']==checksum(folder/'weights.joblib')
    old = ROOT/'round40/comparison'/rotation
    ref = json_at(old/'results.json')
    assert checksum(old/'models.joblib')==ref['weights_sha256']
    cal = joblib.load(old/'models.joblib')['calibration']
    experts = [assemble(x,threshold,'all',cal)[3] for x in frames]
    margins = [x['signed'] for x in frames]
    labels = [x['y'] for x in frames]
    bits = [x>0 for x in margins]
    pred = dict(ids=frames[1]['ids'],base=bits[1],y=labels[1])
    report = {}
    preds = dict(np.load(folder/'scores.npz'))
    support = [supported(frame) for frame in frames]
    roles = ('cal','held')
    for phase,row,allowed in zip(roles,frames,support,strict=True):
        assert np.array_equal(preds[phase+'_ids'],row['ids'][allowed])
    heads = {}
    for arm in ARMS[1:]:
        full = []
        for phase,rank,allowed in zip(roles,margins,support,strict=True):
            estimates = rank.copy()
            estimates[allowed] = preds[phase+'_'+arm]
            full.append(estimates)
        heads[arm] = full
    for mask,panel,width in product(('open','lexical'),SCOPES,WIDTHS):
        area = [scope(frame,q,mask,SCOPES[panel],width) for frame,q in zip(frames,experts,strict=True)]
        if not all(a.any() for a in area):
            continue
        for arm in ARMS:
            for strength in ((0.,) if arm=='threshold' else (.5,1.)):
                name = '/'.join((mask,panel,str(width),arm,f'{strength:g}'))
                p = [raw[ix] if arm=='threshold' else (1.-strength)*raw[ix]+strength*new[ix]
                     for raw,ix,new in zip(margins,area,heads.get(arm,margins),strict=True)]
                row = dict(cal_rows=int(area[0].sum()),held_rows=int(area[1].sum()),
                           panel=panel,mask=mask,width=width,strength=strength,arm=arm)
                if rotation!='official':
                    order,ties = diagnostic_order(p[1],margins[1][area[1]],frames[1]['ids'][area[1]])
                    row['matched'],fixed = per_project(labels[1],frames[1]['project'],bits[1],area[1],order,('fp','tp'))
                    row['ties'] = ties
                    pred[name+'_budget_fp'] = fixed['fp']
                for rule in ('fixed','fp','tp'):
                    edge = cut(labels[0][area[0]],bits[0][area[0]],p[0],rule)
                    c,_ = adjusted(labels[0],bits[0],area[0],p[0],edge)
                    h,z = adjusted(labels[1],bits[1],area[1],p[1],edge)
                    row[rule] = dict(cut=edge,cal=c,held=h)
                    pred[name+'_'+rule] = z
                report[name] = row
    out = DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',**pred)
    write_json(out/'results.json',dict(options=report,head_receipt_sha256=checksum(folder/'receipt.json'),
                     predictions_sha256=checksum(out/'predictions.npz'),data_ticket=ticket,
                     round40_calibration_sha256=ref['weights_sha256'],script_sha256=checksum(Path(__file__))))
    print(rotation,len(report),flush=True)
    return report


def f1(v):
    return v['f1_after']-v['f1_before']


def matched(row,budget):
    change = row['matched'][budget]['change']
    return change['delta_tp'] if budget=='fp' else -change['delta_fp']


def shortlist(reports):
    a,b,dev=(reports[key] for key in PARTS)
    common = set(a)&set(b)&set(dev)
    rows = []
    for key in sorted(common):
        result = dev[key]
        arm = result['arm']
        if arm=='threshold':
            continue
        prefix,_,s = key.rsplit('/',2)
        cutoff = prefix+'/threshold/0'
        control = prefix+('/compact/' if arm.startswith('distill_') else '/score/')+s
        if arm=='score':
            control = cutoff
        if not {cutoff,control}<=common:
            continue
        counts = {budget:[matched(r[key],budget) for r in (a,b)] for budget in ('fp','tp')}
        over = {budget:[v-matched(r[cutoff],budget) for v,r in zip(values,(a,b),strict=True)]
                for budget,values in counts.items()}
        gain = {budget:[v-matched(r[control],budget) for v,r in zip(values,(a,b),strict=True)]
                for budget,values in counts.items()}
        held_ok = any(min(over[k])>=0 and sum(over[k])>0 and min(gain[k])>=0 and sum(gain[k])>0
                      for k in ('fp','tp'))
        for rule in ('fixed','fp','tp'):
            cal,review = result[rule]['cal'],result[rule]['held']
            enough = held_ok and f1(cal)>max(0.,f1(dev[cutoff][rule]['cal']),f1(dev[control][rule]['cal']))
            rows.append(dict(arm=arm,key=key,rule=rule,control=control,cut=result[rule]['cut'],
                       calls_cal=result['cal_rows'],calls_review=result['held_rows'],
                       a_tp=counts['fp'][0],b_tp=counts['fp'][1],
                       a_fp=counts['tp'][0],b_fp=counts['tp'][1],
                       a_tp_over_control=gain['fp'][0],b_tp_over_control=gain['fp'][1],
                       a_fp_over_control=gain['tp'][0],b_fp_over_control=gain['tp'][1],
                       cal_change=f1(cal),review_change=f1(review),review_tp=review['delta_tp'],
                       review_fp=review['delta_fp'],held_ok=held_ok,eligible=enough and f1(review)>=0.))
    return rows


def main():
    reports = {name:run(name) for name in PARTS}
    rows = shortlist(reports)
    csvfile = DEST/'comparison.csv'
    with csvfile.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    key = lambda r:(min(r['cal_change'],r['review_change']),r['cal_change'],r['review_change'],-r['calls_review'])
    categories = {'direct':DIRECT,'transfer':tuple(k for k in ARMS if k.startswith('distill_')),
                  'controls':('score','compact')}
    result = {}
    for name,options in categories.items():
        here = [r for r in rows if r['arm'] in options]
        passed = [r for r in here if r['eligible']]
        result[name] = dict(points=len(here),passed=len(passed),
                            shortlisted=max(passed,key=key,default=None),
                            best_review_filtered=max((r for r in here if r['held_ok']),key=key,default=None))
    write_json(DEST/'selection.json',dict(groups=result,points=len(rows),csv_sha256=checksum(csvfile),
                       sources={r:checksum(DEST/r/'results.json') for r in PARTS},
                       code_sha256=checksum(Path(__file__))))
    print(result,flush=True)


if __name__=='__main__':
    main()
