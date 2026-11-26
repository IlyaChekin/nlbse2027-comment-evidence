"""Check new pretraining against equal-window and compact controls."""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from fit46 import HERE,ROOT,PARTS,f,checksum,json_at,load,parts,write_json,DEST as FIT,BANK
from stack40 import diagnostic_order,cut,adjusted,per_project
from study44 import scope

DEST = HERE/'comparison'
VIEWS = ('numeric','compact','unix96','unix384','qwen96','qwen384','qwen12','joint96','joint384',
         'transfer_unix96','transfer_unix384','transfer_qwen96','transfer_qwen384','transfer_qwen12')
PANELS = ('all','consensus','veto')
WIDTHS = (.8,2.5,5.,None)
MIX = (.35,.7)
CONTROLS = {
    'compact':('numeric',),'unix96':('numeric',),'unix384':('numeric',),
    'qwen96':('numeric','unix96'),'qwen384':('numeric','unix384'),'qwen12':('numeric','unix96'),
    'joint96':('numeric','unix96','qwen96'),'joint384':('numeric','unix384','qwen384'),
    'transfer_unix96':('compact',),'transfer_unix384':('compact',),
    'transfer_qwen96':('compact','transfer_unix96'),'transfer_qwen384':('compact','transfer_unix384'),
    'transfer_qwen12':('compact','transfer_unix96'),
}


def evaluation(rotation,root,out_dir,views,code):
    bank,_,stamp = load(rotation)
    frames = parts(bank)
    path = root/rotation
    report = json_at(path/'receipt.json')
    assert report['matrix_sha256']==stamp
    assert report['code_sha256']==checksum(HERE/code)
    assert report['model_sha256']==checksum(path/'weights.joblib')
    assert report['predictions_sha256']==checksum(path/'scores.npz')
    assert report['pair_sha256']==checksum(ROOT/'round37/matched'/f'{rotation}.npz')
    assert all(v==checksum(HERE/'features'/BANK[k][0]/'receipt.json')
               for k,v in report['source_features'].items())
    saved = np.load(path/'scores.npz')
    estimates = {k:[] for k in views}
    truth = [z['y'] for z in frames]
    margins = [z['signed'] for z in frames]
    base = [z>0. for z in margins]
    for role,frame in zip(('cal','held'),frames,strict=True):
        in_scope = scope(frame,'all',None)
        assert np.array_equal(saved[role+'_ids'],frame['ids'][in_scope])
        for arm in views:
            q = frame['signed'].copy()
            q[in_scope] = saved[role+'_'+arm]
            estimates[arm].append(q)
    data = {}
    pred = dict(ids=frames[1]['ids'],truth=truth[1],base=base[1])
    for panel,width in product(PANELS,WIDTHS):
        select = [scope(frame,panel,width) for frame in frames]
        if not all(u.any() for u in select):
            continue
        for arm,mix in [('threshold',0.),*product(views,MIX)]:
            key = f'{panel}/{width}/{arm}/{mix:g}'
            values = [(1.-mix)*r[s]+mix*p[s] for r,s,p in zip(margins,select,estimates.get(arm,margins),strict=True)]
            note = dict(arm=arm,panel=panel,width=width,mix=mix,
                        cal_rows=int(select[0].sum()),review_rows=int(select[1].sum()))
            if rotation!='official':
                rank,_ = diagnostic_order(values[1],margins[1][select[1]],frames[1]['ids'][select[1]])
                note['matched'],variants = per_project(truth[1],frames[1]['project'],base[1],select[1],rank,('fp','tp'))
                for style in ('fp','tp'):
                    pred[key+'_same_'+style] = variants[style]
            for rule in ('fixed','fp','tp'):
                edge = cut(truth[0][select[0]],base[0][select[0]],values[0],rule)
                c,_ = adjusted(truth[0],base[0],select[0],values[0],edge)
                q,z = adjusted(truth[1],base[1],select[1],values[1],edge)
                note[rule] = dict(cut=edge,cal=c,held=q)
                pred[key+'_'+rule] = z
            data[key] = note
    out = out_dir/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',**pred)
    write_json(out/'results.json',dict(options=data,fit_sha256=checksum(path/'receipt.json'),
                       predictions_sha256=checksum(out/'predictions.npz'),code_sha256=checksum(Path(__file__))))
    print(rotation,len(data),flush=True)
    return data


def matched(row,rule):
    z = row['matched'][rule]['change']
    return z['delta_tp'] if rule=='fp' else -z['delta_fp']


def f1(row,place):
    a = row[place]
    return a['f1_after']-a['f1_before']


def estimated_cost():
    price = {}
    for k,(home,_) in BANK.items():
        data = json_at(HERE/'features'/home/'receipt.json')
        compute = data['cost']['analytic_gflops']
        price[k] = .2*compute/data['rows']/3_000_000
        if k=='qwen12':
            price[k] *= .5
    for cap in (96,384):
        price['joint'+str(cap)] = price['qwen'+str(cap)]+price['unix'+str(cap)]
    return price


def choose(data):
    a,b,c = (data[k] for k in PARTS)
    price = estimated_cost()
    output = []
    for key in sorted(set(a)&set(b)&set(c)):
        row = c[key]
        arm = row['arm']
        if arm=='threshold':
            continue
        region,_,weight = key.rsplit('/',2)
        controls = [region+'/threshold/0',*(f'{region}/{name}/{weight}' for name in CONTROLS.get(arm,()))]
        amounts = {}
        for rule in ('fp','tp'):
            raw = [matched(z[key],rule) for z in (a,b)]
            extra = [r-max(matched(z[p],rule) for p in controls) for r,z in zip(raw,(a,b),strict=True)]
            amounts[rule] = raw,extra
        project = any(min(z)>=0 and sum(z)>0 for _,z in amounts.values())
        charge = price.get(arm,0.)*row['review_rows']
        for rule in ('fixed','fp','tp'):
            gain = f1(row[rule],'cal')
            review = f1(row[rule],'held')
            best = max(0.,max(f1(c[p][rule],'cal') for p in controls))
            pass_cost_proxy = .3*review>charge if arm in price else True
            r = dict(key=key,arm=arm,kind=('direct' if arm in price else 'small'),panel=row['panel'],width=row['width'],
                       mix=row['mix'],cut=row[rule]['cut'],rule=rule,cal_rows=row['cal_rows'],
                       review_rows=row['review_rows'],a_tp=amounts['fp'][0][0],b_tp=amounts['fp'][0][1],
                       a_tp_over=amounts['fp'][1][0],b_tp_over=amounts['fp'][1][1],
                       a_fp_over=amounts['tp'][1][0],b_fp_over=amounts['tp'][1][1],
                       cal_f1=gain,review_f1=review,over_cal=gain-best,
                       review_tp=row[rule]['held']['delta_tp'],review_fp=row[rule]['held']['delta_fp'],
                       approximate_compute_score_cost=charge,estimate_ignores_time=True,
                       project_gate=project,quality=project and gain>best and review>=0.,
                       eligible=project and gain>best and review>=0. and pass_cost_proxy)
            output.append(r)
    return output


def main(paired=False):
    src = HERE/'paired' if paired else FIT
    output = HERE/'paired_comparison' if paired else DEST
    code = 'pairs46.py' if paired else 'fit46.py'
    views = tuple(k for k in VIEWS if not k.startswith('transfer_')) if paired else VIEWS
    report = choose({k:evaluation(k,src,output,views,code) for k in PARTS})
    path = output/'comparison.csv'
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=report[0])
        w.writeheader()
        w.writerows(report)
    rank = lambda z:(min(z['cal_f1'],z['review_f1']),z['cal_f1']+z['review_f1'],-z['review_rows'])
    stages = {}
    for arm in views:
        own = [q for q in report if q['arm']==arm]
        stages[arm] = dict(eligible=sum(q['eligible'] for q in own),quality=sum(q['quality'] for q in own),
                     best=max((q for q in own if q['eligible']),key=rank,default=None),
                     quality_only=max((q for q in own if q['quality']),key=rank,default=None),
                     project=max((q for q in own if q['project_gate']),key=rank,default=None),
                     best_matched=max(own,key=lambda z:(min(z['a_tp_over'],z['b_tp_over']),
                                                       z['a_tp_over']+z['b_tp_over'],rank(z))))
    selection = {arm:max((q for q in report if q['kind']==arm and q['eligible']),key=rank,default=None)
                 for arm in ('small','direct')}
    release = json_at(ROOT/'round38/release_summary.json')
    out = dict(selection=selection,models=stages,tested=len(report),eligible=sum(q['eligible'] for q in report),
               archive_sha256=release['archive_sha256'],sources={k:checksum(output/k/'results.json') for k in PARTS},
               csv_sha256=checksum(path),code_sha256=checksum(Path(__file__)))
    write_json(output/'selection.json',out)
    print('tested',len(report),'selection',selection,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired',action='store_true')
    main(p.parse_args().paired)
