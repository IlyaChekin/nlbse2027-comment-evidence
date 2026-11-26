"""Learn expected changes of the final decision on file-disjoint supervision."""
import argparse
from itertools import product
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from points33 import HERE, PARTS, OUT, f, checksum, json_at, write_json, strict_cut
from policy33 import matrix, reassign, prediction

FRACTIONS=(.1,.25,.5,1.)
STRENGTHS=(.35,.65,1.)
MODES=('conservative','balanced','recall','classification','separate')
DEST=HERE/'comparison'


def read_phase(name, root, receipt):
    path = root/f'{name}.npz'
    if checksum(path)!=receipt['phases'][name]:
        raise ValueError('The cached counterfactual scores changed')
    with np.load(path) as z:
        return dict(z)


def crop(item, mask):
    return {k:v[mask] for k,v in item.items()}


def train(data, threshold):
    x = matrix(data['pre'],data['length'],threshold)
    truth = data['y']
    flip = ((data['queried']>0.).astype(np.int8)-(data['unqueried']>0.).astype(np.int8))
    gain_tp = flip * truth
    gain_fp = flip * ~truth
    params = dict(max_iter=115,max_leaf_nodes=23,min_samples_leaf=max(35,len(truth)//1900),
                  learning_rate=.065, l2_regularization=4.,early_stopping=False,random_state=42)
    bank = {}
    for name,cost in [('conservative',1.30),('balanced',.90),('recall',.56)]:
        bank[name] = HistGradientBoostingRegressor(**params).fit(x,gain_tp-cost*gain_fp)
    # Four possible changes, plus no change. Retain separate estimates of gains and harms.
    label = np.select((gain_tp==1,gain_tp==-1,gain_fp==1,gain_fp==-1),(1,2,3,4),default=0)
    cls = HistGradientBoostingClassifier(**params).fit(x,label)
    bank['classifier'] = cls
    bank['class_utility'] = np.asarray([0.,1.,-1.,-.90,.90])[cls.classes_]
    negative = x[:,7]<=0.
    for key,group in [('negative',negative),('positive',~negative)]:
        bank[key] = HistGradientBoostingRegressor(**params).fit(x[group],gain_tp[group]-.9*gain_fp[group])
    return bank,dict(fit_rows=len(x),flip_count=int((flip!=0).sum()),beneficial=int(((flip!=0)&((data['queried']>0.)==truth)).sum()))


def cut(y,base,score,mode):
    return strict_cut(y,base,score,mode)


def in_scope(data, mask, threshold, mode):
    score = np.where(mask,data['queried'],data['unqueried'])
    if mode=='fixed':
        threshold = 0.
    bits = score>threshold
    return bits,score


def matched(data, score, baseline, mode):
    rows = []
    predict = np.zeros(len(score),bool)
    for p in np.unique(data['project']):
        use = data['project']==p
        y, b, q = data['y'][use],baseline[use],score[use]
        bound = cut(y,b,q,mode)
        predict[use] = q>bound
        change = f.comparison(y,b,predict[use])
        rows.append(dict(project=str(p),cut=bound,**change))
    same = all(r['delta_'+mode]==0 for r in rows)
    return dict(overall=f.comparison(data['y'],baseline,predict),exact=same,projects=rows),predict


def budget(data, selected):
    old = data['old']
    size = data['length']
    removed = old & ~selected
    added = selected & ~old
    return dict(calls=int(selected.sum()),characters=int(size[selected].sum()),swap=int(added.sum()),
                old_calls=int(old.sum()),old_characters=int(size[old].sum()),removed_large=int((removed & (size>=2048)).sum()),
                added_large=int((added & (size>=2048)).sum()))


def oracle(data):
    y = data['y']
    a,b = data['unqueried']>0.,data['queried']>0.
    helper = b==y
    ideal = np.where(helper,b,a)
    base = data['replay']>0.
    raw = np.select((helper & (a!=b),(~helper)&(a!=b)),(1.,-1.),default=0.)
    selected = reassign(data['old'],data['length'],raw,data['pre'][:,8],1.,1.)
    held = np.where(selected,b,a)
    return dict(any_budget=f.comparison(y,base,ideal),same_budget=f.comparison(y,base,held),
                routing=budget(data,selected))


def evaluate(rotation):
    root = OUT/rotation
    receipt = json_at(root/'receipt.json')
    if receipt['code_sha256']!=checksum(HERE/'points33.py'):
        raise ValueError('Counterfactual producer changed')
    start = perf_counter()
    d = read_phase('dev',root,receipt)
    part = d if rotation=='official' else read_phase('test',root,receipt)
    cal,held = crop(d,d['cal']),crop(part,part['held'])
    if np.intersect1d(cal['ids'],held['ids']).size:
        raise ValueError('Threshold selection and review share rows')
    t = receipt['cutoff']
    available={'fit':crop(d,d['fit'])}
    if 'oof' in receipt['phases']:
        available['oof']=read_phase('oof',root,receipt)
    x = [matrix(z['pre'],z['length'],t) for z in (cal,held)]
    banks={}
    notes={}
    for key,source in available.items():
        banks[key],notes[key] = train(source,t)
        print(rotation,key,notes[key],round(perf_counter()-start,2),flush=True)
    skills = [('greedy', 'shared')] + list(product(MODES,banks))
    schedules=[('released','shared',0.,0.)]+[(mode,kind,strength,fraction) for mode,kind in skills
                                             for strength,fraction in product(STRENGTHS,FRACTIONS)]
    stats={}
    snapshot=dict(ids=held['ids'],y=held['y'],base=held['replay']>0.,old=held['old'],
                  unqueried=held['unqueried'],queried=held['queried'],length=held['length'])
    dc,dt=cal['replay']>0.,held['replay']>0.
    preferences={(mode,kind):[prediction(banks.get(kind),mode,z) for z in x]
                 for mode,kind in skills}
    for mode,kind,strength,fraction in schedules:
        key=f'{mode}_{kind}_{strength:g}_{fraction:g}'
        if mode=='released':
            qa,qb=cal['old'],held['old']
        else:
            qa,qb=[reassign(data['old'],data['length'],gain,data['pre'][:,8],strength,fraction)
                   for data,gain in zip((cal,held),preferences[mode,kind],strict=True)]
        _,p = in_scope(cal,qa,0.,'fixed')
        _,q = in_scope(held,qb,0.,'fixed')
        options={}
        if rotation!='official':
            for scheme in ('tp','fp'):
                result,bits=matched(held,q,dt,scheme)
                options['matched_'+scheme]=result
                snapshot[f'{key}_matched_{scheme}']=bits
        for scheme in ('tp','fp','fixed'):
            c=cut(cal['y'],dc,p,scheme)
            new0,new1=p>c,q>c
            options[scheme]=dict(cut=c,cal=f.comparison(cal['y'],dc,new0),
                                 frozen=f.comparison(held['y'],dt,new1))
            snapshot[f'{key}_{scheme}']=new1
        snapshot[f'{key}_requests']=qb
        snapshot[f'{key}_rank']=q.astype(np.float32)
        stats[key]=dict(mode=mode,source=kind,strength=strength,fraction=fraction,options=options,
                        calibration=budget(cal,qa),review=budget(held,qb))
    folder=DEST/rotation
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/'predictions.npz'
    np.savez_compressed(file,**snapshot)
    ck=folder/'models.joblib'
    joblib.dump(banks,ck,compress=3)
    proof=dict(oracle={k:oracle(v) for k,v in (('cal',cal),('held',held))},training=notes,settings=stats,rotation=rotation,
               points_sha256=checksum(root/'receipt.json'),models_sha256=checksum(ck),predictions_sha256=checksum(file),
               policy_sha256=checksum(HERE/'policy33.py'),code_sha256=checksum(HERE/'experiment33.py'),cutoff=t)
    write_json(folder/'results.json',proof)
    print(rotation,len(stats),'cal',proof['oracle']['cal'],'held',proof['oracle']['held'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    evaluate(parser.parse_args().rotation)
