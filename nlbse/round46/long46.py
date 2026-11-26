"""Test a cheap length router without consulting the downstream labels."""
import csv
from itertools import product
from pathlib import Path

import numpy as np

from fit46 import HERE,ROOT,PARTS,DEST as FIT,f,checksum,json_at,load,parts,write_json,negative
from views46 import DEST as WINDOWS
from compare46 import VIEWS,choose
from stack40 import diagnostic_order,cut,adjusted,per_project
from study44 import scope

DEST = HERE/'length'
GOALS = (.75,.9)


def length_threshold(row,roster,cut384):
    include = negative(row)
    index = np.searchsorted(roster,row['ids'][include])
    assert np.array_equal(roster[index],row['ids'][include])
    lengths = row['length'][include]
    observed = cut384[index]
    rules = {}
    for recall in GOALS:
        for threshold in range(300,6001,50):
            seen = lengths>=threshold
            if seen.sum()>=100 and observed[seen].mean()>=recall:
                rules[str(recall)] = dict(chars=threshold,estimated_precision=float(observed[seen].mean()),
                                          training_rows=int(seen.sum()))
                break
    return rules


def study(rotation):
    source,_,sig = load(rotation)
    frames = parts(source)
    windows = json_at(WINDOWS/'receipt.json')
    assert windows['roster_sha256']==checksum(WINDOWS/'rows.npz')
    index = np.load(WINDOWS/'rows.npz')['ids']
    p = WINDOWS/'qwen_384.npz'
    assert windows['windows']['qwen_384']['sha256']==checksum(p)
    removed = ~np.load(p)['full']
    rules = length_threshold(source['oof'],index,removed)
    fit = FIT/rotation
    path = fit/'scores.npz'
    proof = json_at(fit/'receipt.json')
    assert proof['predictions_sha256']==checksum(path) and proof['matrix_sha256']==sig
    archive = np.load(path)
    estimate = {arm:[] for arm in VIEWS}
    for role,r in zip(('cal','held'),frames,strict=True):
        active = negative(r)
        assert np.array_equal(archive[role+'_ids'],r['ids'][active])
        for arm in VIEWS:
            values = r['signed'].copy()
            values[active] = archive[role+'_'+arm]
            estimate[arm].append(values)
    margins = [z['signed'] for z in frames]
    base = [z>0. for z in margins]
    truth = [z['y'] for z in frames]
    saved = dict(ids=frames[1]['ids'],base=base[1],truth=truth[1])
    display = {}
    for goal,region,width in product(rules,('all','veto','consensus'),(2.5,None)):
        chars = rules[goal]['chars']
        selects = [scope(row,region,width)&(row['length']>=chars) for row in frames]
        if not all(v.any() for v in selects):
            continue
        for arm,mix in [('threshold',0.),*product(VIEWS,(.35,.7))]:
            key = f'{goal}/{region}/{width}/{arm}/{mix:g}'
            arrays = [(1.-mix)*old[m]+mix*new[m] for old,new,m in
                      zip(margins,estimate.get(arm,margins),selects,strict=True)]
            outcome = dict(arm=arm,panel=region,width=width,mix=mix,minimum_chars=chars,
                           cal_rows=int(selects[0].sum()),review_rows=int(selects[1].sum()))
            if rotation!='official':
                use=selects[1]
                decided,_=diagnostic_order(arrays[1],margins[1][use],frames[1]['ids'][use])
                outcome['matched'],items=per_project(truth[1],frames[1]['project'],base[1],use,decided,('fp','tp'))
                for k in ('fp','tp'):
                    saved[key+'_same_'+k] = items[k]
            for k in ('fixed','fp','tp'):
                use=selects[0]
                edge = cut(truth[0][use],base[0][use],arrays[0],k)
                c,_=adjusted(truth[0],base[0],use,arrays[0],edge)
                h,z=adjusted(truth[1],base[1],selects[1],arrays[1],edge)
                outcome[k]=dict(cut=edge,cal=c,held=h)
                saved[key+'_'+k]=z
            display[key]=outcome
    home=DEST/rotation
    home.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(home/'predictions.npz',**saved)
    write_json(home/'results.json',dict(options=display,rules=rules,fit_sha256=checksum(fit/'receipt.json'),
                       predictions_sha256=checksum(home/'predictions.npz'),code_sha256=checksum(Path(__file__))))
    print(rotation,rules,len(display),flush=True)
    return display


def main():
    tables={r:study(r) for r in PARTS}
    report=choose(tables)
    for row in report:
        row['minimum_chars']=tables['official'][row['key']]['minimum_chars']
    good=[x for x in report if x['eligible']]
    metric=lambda z:(min(z['cal_f1'],z['review_f1']),z['cal_f1']+z['review_f1'],-z['review_rows'])
    selection={k:max((z for z in good if z['kind']==k),key=metric,default=None) for k in ('direct','small')}
    diag=max(report,key=lambda q:(min(q['a_tp_over'],q['b_tp_over']),q['a_tp_over']+q['b_tp_over'],metric(q)))
    best_qwen=max((z for z in report if z['arm'].startswith(('qwen','transfer_qwen'))),
                   key=lambda q:(min(q['a_tp_over'],q['b_tp_over']),q['a_tp_over']+q['b_tp_over'],metric(q)))
    path=DEST/'comparison.csv'
    with path.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=report[0])
        w.writeheader()
        w.writerows(report)
    release = json_at(ROOT/'round38/release_summary.json')
    write_json(DEST/'selection.json',dict(selection=selection,best_matched=diag,best_qwen=best_qwen,tested=len(report),
               archive_sha256=release['archive_sha256'],eligible=len(good),csv_sha256=checksum(path),
               sources={r:checksum(DEST/r/'results.json') for r in PARTS},code_sha256=checksum(Path(__file__))))
    print(len(report),selection,best_qwen,flush=True)


if __name__=='__main__':
    main()
