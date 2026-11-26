"""Compare directional and budgeted expert policies with fixed project FP counts."""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from policy45 import HERE,ROOT,PARTS,DEST as FIT,f,checksum,json_at,load,parts,write_json
from stack40 import diagnostic_order,cut,adjusted,per_project

DEST = HERE/'comparison'
WIDTHS = (.8,2.,None)
PANELS = {'all':(0,1,2),'known':(0,1),'negative':(0,),'positive':(1,),'unseen':(2,)}
MIXES = (.4,.75)
RATES = (.08,.25)
SCREENS = ('margin','utility','priced','margin_matched')
MODES = ('lex','dense')


def scope(frame,panel,width):
    if 'observed_regime' in frame:
        group = frame['observed_regime']
    else:
        known = frame['old']&(frame['flag']>=0)
        group = np.where(known,np.where(frame['flag']==0,0,1),2)
    eligible = np.isin(group,PANELS[panel])
    if width is not None:
        eligible &= abs(frame['signed'])<width
    return eligible


def route_query(frame,mask,score,rate,net=None):
    # Match the number of queries within each baseline decision and scanner branch.
    if 'observed_regime' in frame:
        branch = frame['observed_regime']
    else:
        branch = np.where(frame['old']&(frame['flag']>=0),np.where(frame['flag']==0,0,1),2)
    sign = frame['signed']>0.
    selected = np.zeros(len(mask),bool)
    for regime,direction in product(np.unique(branch[mask]),(False,True)):
        idx = np.flatnonzero(mask&(branch==regime)&(sign==direction))
        count = int(round(len(idx)*rate))
        if net is not None:
            count = min(count,int((net[idx]>0.).sum()))
        if count:
            # Stable input order gives the same result regardless of stored parquet IDs.
            order = np.argsort(-score[idx],kind='stable')
            selected[idx[order[:count]]] = True
    return selected


def chosen_ranks(frame,mask,values,kind,screen,rate,mix):
    original = frame['signed']
    if kind in ('threshold','point','numeric'):
        q = values[kind] if kind!='threshold' else original
        ask = np.zeros(len(original),bool)
    else:
        q = values['numeric'].copy()
        utility = values['gate_'+kind]
        value = -abs(original) if screen in ('margin','margin_matched') else utility
        net = utility if screen in ('priced','margin_matched') else None
        ask = route_query(frame,mask,value,rate,net)
        q[ask] = values[kind][ask]
    return (1.-mix)*original[mask]+mix*q[mask],ask


def compare(rotation,output,switched):
    bank,t,stamp = load(rotation)
    frames = parts(bank)
    parent = FIT/rotation
    receipt = json_at(parent/'receipt.json')
    assert receipt['code_sha256']==checksum(HERE/'policy45.py')
    assert receipt['matrix_sha256']==stamp
    for name,sha in receipt['files'].items():
        assert sha==checksum(parent/name)
    cache = np.load(parent/'scores.npz')
    alternate = None
    switch_proof = None
    if switched:
        optional = HERE/'switch'/rotation
        switch_proof = checksum(optional/'receipt.json')
        switch = json_at(optional/'receipt.json')
        assert switch['fit_sha256']==checksum(parent/'receipt.json')
        assert switch['matrix_sha256']==stamp
        assert switch['scores_sha256']==checksum(optional/'scores.npz')
        assert switch['weights_sha256']==checksum(optional/'weights.joblib')
        assert switch['code_sha256']==checksum(HERE/'switch45.py')
        alternate = np.load(optional/'scores.npz')
    values = []
    for role,frame in zip(('cal','held'),frames,strict=True):
        assert np.array_equal(cache[role+'_ids'],frame['ids'])
        scores = {k:cache[role+'_'+k] for k in ('point','numeric','lex','dense','gate_lex','gate_dense')}
        if alternate is not None:
            assert np.array_equal(alternate[role+'_ids'],frame['ids'])
            scores.update({key:alternate[role+'_'+key] for key in ('gate_lex','gate_dense')})
        values.append(scores)
    margins = [x['signed'] for x in frames]
    baseline = [x>0. for x in margins]
    truth = [x['y'] for x in frames]
    save = dict(ids=frames[1]['ids'],y=truth[1],base=baseline[1])
    table = {}
    for panel,width in product(PANELS,WIDTHS):
        mask = [scope(r,panel,width) for r in frames]
        if not all(x.any() for x in mask):
            continue
        cases = [('threshold','none',0.,0.),
                 *((name,'none',0.,strength) for name,strength in product(('point','numeric'),MIXES)),
                 *product(MODES,SCREENS,RATES,MIXES)]
        canonical_counts = {}
        for kind,screen,rate,mix in cases:
            key = f'{panel}/{width}/{kind}/{screen}/{rate:g}/{mix:g}'
            pairs = [chosen_ranks(frame,active,data,kind,screen,rate,mix)
                     for frame,active,data in zip(frames,mask,values,strict=True)]
            rank,ask = zip(*pairs,strict=True)
            query_counts = tuple(int(s.sum()) for s in ask)
            if kind in MODES:
                ident = (kind if screen in ('priced','margin_matched') else 'cap',rate)
                assert canonical_counts.setdefault(ident,query_counts)==query_counts
            option = dict(panel=panel,width=width,kind=kind,screen=screen,rate=rate,mix=mix,
                          cal_rows=int(mask[0].sum()),review_rows=int(mask[1].sum()),
                          cal_queries=query_counts[0],review_queries=query_counts[1])
            if rotation!='official':
                ordered,_=diagnostic_order(rank[1],margins[1][mask[1]],frames[1]['ids'][mask[1]])
                option['matched'],bits=per_project(truth[1],frames[1]['project'],baseline[1],mask[1],ordered,('fp','tp'))
                for axis in ('tp','fp'):
                    save[key+'_same_'+axis] = bits[axis]
            for method in ('fixed','fp','tp'):
                edge=cut(truth[0][mask[0]],baseline[0][mask[0]],rank[0],method)
                dev,_=adjusted(truth[0],baseline[0],mask[0],rank[0],edge)
                q,bits=adjusted(truth[1],baseline[1],mask[1],rank[1],edge)
                option[method] = dict(cut=edge,cal=dev,held=q)
                save[key+'_'+method] = bits
            table[key] = option
    out = output/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',**save)
    write_json(out/'results.json',dict(options=table,fit_sha256=checksum(parent/'receipt.json'),
                     switch_sha256=switch_proof,
                     predictions_sha256=checksum(out/'predictions.npz'),code_sha256=checksum(Path(__file__))))
    print(rotation,len(table),flush=True)
    return table


def account(row,rule):
    z = row['matched'][rule]['change']
    return z['delta_tp'] if rule=='fp' else -z['delta_fp']


def f1(row,part):
    x = row[part]
    return x['f1_after']-x['f1_before']


def compare_across(roles):
    a,b,d = (roles[k] for k in PARTS)
    results = []
    for key in sorted(set(a)&set(b)&set(d)):
        x = d[key]
        if not x['mix']:
            continue
        panel,width,kind,screen,rate,weight = key.split('/')
        stem = f'{panel}/{width}'
        threshold = stem+'/threshold/none/0/0'
        plain = stem+f'/point/none/0/{weight}'
        numeric = stem+f'/numeric/none/0/{weight}'
        comparators = [threshold]
        if kind!='point':
            comparators.append(plain)
        if kind in MODES:
            comparators.append(numeric)
            if screen in ('utility','priced'):
                paired = 'margin' if screen=='utility' else 'margin_matched'
                comparators.append(stem+f'/{kind}/{paired}/{rate}/{weight}')
        metrics = {}
        for metric in ('tp','fp'):
            absolute = [account(default[key],metric) for default in (a,b)]
            above = [score-max(account(default[v],metric) for v in comparators)
                     for score,default in zip(absolute,(a,b),strict=True)]
            metrics[metric] = absolute,above
        stable = any(min(z)>=0 and sum(z)>0 for _,z in metrics.values())
        for rule in ('fixed','fp','tp'):
            cal,review = [f1(x[rule],k) for k in ('cal','held')]
            stronger = max(0.,*[f1(d[z][rule],'cal') for z in comparators])
            res = dict(key=key,kind=kind,screen=screen,rate=float(rate),mix=x['mix'],width=x['width'],
                       panel=panel,rule=rule,cut=x[rule]['cut'],cal_rows=x['cal_rows'],review_rows=x['review_rows'],
                       cal_queries=x['cal_queries'],review_queries=x['review_queries'],
                       a_tp=metrics['fp'][0][0],b_tp=metrics['fp'][0][1],
                       a_tp_over=metrics['fp'][1][0],b_tp_over=metrics['fp'][1][1],
                       a_fp_removed=metrics['tp'][0][0],b_fp_removed=metrics['tp'][0][1],
                       a_fp_over=metrics['tp'][1][0],b_fp_over=metrics['tp'][1][1],
                       cal_change=cal,review_change=review,over_cal=cal-stronger,
                       review_tp=x[rule]['held']['delta_tp'],review_fp=x[rule]['held']['delta_fp'],
                       held_pass=stable,eligible=stable and cal>stronger and review>=0.)
            results.append(res)
    return results


def main(switched=False):
    output = HERE/'switch_comparison' if switched else DEST
    studies = {k:compare(k,output,switched) for k in PARTS}
    plans = compare_across(studies)
    path = output/'comparison.csv'
    with path.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=plans[0])
        w.writeheader()
        w.writerows(plans)
    selection = {}
    def merit(a):
        return min(a['cal_change'],a['review_change']),a['cal_change']+a['review_change'],-a['review_queries']
    for kind,screen in [('point','none'),('numeric','none'),*product(MODES,SCREENS)]:
        candidates = [v for v in plans if v['kind']==kind and v['screen']==screen]
        key = kind+'/'+screen
        selection[key] = dict(tested=len(candidates),eligible=sum(v['eligible'] for v in candidates),
                selected=max((r for r in candidates if r['eligible']),key=merit,default=None),
                matched=max((r for r in candidates if r['held_pass']),key=merit,default=None))
    permitted = [p for p in plans if p['eligible']]
    selected = max((r for r in permitted if r['kind']!='dense'),key=merit,default=None)
    dense = max((r for r in permitted if r['kind']=='dense'),key=merit,default=None)
    release = json_at(ROOT/'round38/release_summary.json')
    proof = dict(candidates=selection,tested=len(plans),cheap=selected,dense=dense,
                 sources={phase:checksum(output/phase/'results.json') for phase in PARTS},
                 csv_sha256=checksum(path),code_sha256=checksum(Path(__file__)),
                 archive_sha256=release['archive_sha256'])
    write_json(output/'selection.json',proof)
    print('tested',len(plans),'cheap',selected,'dense',dense,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--switch-gate',action='store_true')
    main(parser.parse_args().switch_gate)
