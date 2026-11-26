"""Exchange positive and negative decisions under a label-free count constraint."""
import csv
from itertools import product
from pathlib import Path

import numpy as np

from policy45 import HERE,ROOT,PARTS,DEST as FIT,f,checksum,json_at,load,parts,write_json
from compare45 import chosen_ranks,scope

DEST = HERE/'swap'
PANELS = ('all','known','positive','negative','unseen')
WIDTHS = (1.6,3.8,None)
EXCHANGES = (.0025,.01,.04)
MARGINS = (0.,.35,1.)
MODES = (('point','none'),('numeric','none'),('lex','margin'),('lex','utility'),
         ('dense','margin'),('dense','utility'))
RATE = .08
BLEND = .75
BRANCHES = (0,1,2)


def swap(frame,scope,rank,rate,edge):
    signed = frame['signed']
    prediction = signed>0.
    before = prediction.copy()
    groups = frame['observed_regime']
    estimate = signed.copy()
    estimate[scope] = rank
    pairs = []
    for branch in BRANCHES:
        index = np.flatnonzero(scope&(groups==branch))
        negative = index[~before[index]]
        positive = index[before[index]]
        maximum = min(len(negative),len(positive),int(round(len(index)*rate)))
        if maximum==0:
            continue
        up = negative[np.argsort(-estimate[negative],kind='stable')[:maximum]]
        down = positive[np.argsort(estimate[positive],kind='stable')[:maximum]]
        used = estimate[up]-estimate[down]>edge
        prediction[up[used]] = True
        prediction[down[used]] = False
        pairs.append((up[used],down[used]))
        assert prediction[index].sum()==before[index].sum()
    return prediction,int(sum(len(a) for a,b in pairs))


def compare(rotation):
    matrices,t,stamp = load(rotation)
    frames = parts(matrices)
    fit = FIT/rotation
    proof = fit/'receipt.json'
    ref = json_at(proof)
    switch = HERE/'switch'/rotation
    stamp2 = json_at(switch/'receipt.json')
    assert stamp2['fit_sha256']==checksum(proof)
    assert ref['matrix_sha256']==stamp==stamp2['matrix_sha256']
    assert ref['files']['scores.npz']==checksum(fit/'scores.npz')
    assert stamp2['scores_sha256']==checksum(switch/'scores.npz')
    z=np.load(fit/'scores.npz')
    extra=np.load(switch/'scores.npz')
    all_scores = []
    for role,frame in zip(('cal','held'),frames,strict=True):
        assert np.array_equal(frame['ids'],z[role+'_ids'])
        assert np.array_equal(frame['ids'],extra[role+'_ids'])
        u = {k:z[role+'_'+k] for k in ('point','numeric','lex','dense')}
        u.update({k:extra[role+'_'+k] for k in ('gate_lex','gate_dense')})
        all_scores.append(u)
    reference = [frame['signed']>0. for frame in frames]
    saved = dict(ids=frames[1]['ids'],base=reference[1],truth=frames[1]['y'])
    options = {}
    for band,width in product(PANELS,WIDTHS):
        masks = [scope(q,band,width) for q in frames]
        if not all(m.any() for m in masks):
            continue
        for kind,gate in MODES:
            cache = [chosen_ranks(q,m,data,kind,gate,RATE,BLEND)
                     for q,m,data in zip(frames,masks,all_scores,strict=True)]
            for cap,gap in product(EXCHANGES,MARGINS):
                name = f'{band}/{width}/{kind}/{gate}/{cap:g}/{gap:g}'
                results,changes,bits = [],[],[]
                checks = {}
                for q,m,(rank,ask),old in zip(frames,masks,cache,reference,strict=True):
                    z,n = swap(q,m,rank,cap,gap)
                    card = f.comparison(q['y'],old,z)
                    assert card['delta_tp']==-card['delta_fp']
                    results.append(card)
                    changes.append(n)
                    bits.append(z)
                for code in np.unique(frames[1]['observed_regime']):
                    where = frames[1]['observed_regime']==code
                    checks[str(int(code))] = f.comparison(frames[1]['y'][where],reference[1][where],bits[1][where])
                options[name] = dict(kind=kind,gate=gate,panel=band,width=width,cap=cap,gap=gap,
                        cal=results[0],review=results[1],cal_exchanges=changes[0],review_exchanges=changes[1],
                        cal_queries=int(cache[0][1].sum()),review_queries=int(cache[1][1].sum()),
                        review_scanners=checks)
                saved[name] = bits[1]
    target = DEST/rotation
    target.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(target/'predictions.npz',**saved)
    write_json(target/'results.json',dict(options=options,fit_sha256=checksum(proof),
                   switch_sha256=checksum(switch/'receipt.json'),matrix_sha256=stamp,
                   predictions_sha256=checksum(target/'predictions.npz'),code_sha256=checksum(Path(__file__))))
    print(rotation,len(options),flush=True)
    return options


def measure(results):
    a,b,d = (results[k] for k in PARTS)
    table = []
    for key in sorted(set(a)&set(b)&set(d)):
        estimate = d[key]
        panel,width,kind,gate,n,gap = key.split('/')
        stem=f'{panel}/{width}'
        other = []
        if kind!='point':
            other.append(f'{stem}/point/none/{n}/{gap}')
        if kind in ('lex','dense'):
            other.append(f'{stem}/numeric/none/{n}/{gap}')
            if gate=='utility':
                other.append(f'{stem}/{kind}/margin/{n}/{gap}')
        gain = [x[key]['review']['delta_tp'] for x in (a,b)]
        extra = [g-max([0]+[x[q]['review']['delta_tp'] for q in other]) for g,x in zip(gain,(a,b),strict=True)]
        cal = estimate['cal']['delta_tp']
        review = estimate['review']['delta_tp']
        strong = max([0]+[d[q]['cal']['delta_tp'] for q in other])
        enough = min(extra)>=0 and sum(extra)>0 and cal>strong and review>=0
        table.append(dict(key=key,view=kind,gate=gate,panel=panel,width=estimate['width'],
                     cap=estimate['cap'],gap=estimate['gap'],cal_tp=cal,review_tp=review,
                     a_tp=gain[0],b_tp=gain[1],a_extra=extra[0],b_extra=extra[1],
                     cal_over=cal-strong,cal_exchanges=estimate['cal_exchanges'],
                     review_exchanges=estimate['review_exchanges'],
                     cal_queries=estimate['cal_queries'],review_queries=estimate['review_queries'],
                     eligible=enough))
    return table


def main():
    report = measure({k:compare(k) for k in PARTS})
    pick = lambda r:(min(r['cal_tp'],r['review_tp']),r['cal_tp']+r['review_tp'],-r['review_queries'])
    winners = [p for p in report if p['eligible']]
    policy = max((r for r in winners if r['view']!='dense'),key=pick,default=None)
    dense = max((r for r in winners if r['view']=='dense'),key=pick,default=None)
    ceiling = {}
    for kind,gate in MODES:
        eligible = [r for r in report if r['view']==kind and r['gate']==gate]
        ceiling[kind+'/'+gate] = max(eligible,key=lambda q:(min(q['a_extra'],q['b_extra']),
                           q['a_extra']+q['b_extra'],pick(q)))
    table = DEST/'comparison.csv'
    with table.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=report[0])
        w.writeheader()
        w.writerows(report)
    write_json(DEST/'selection.json',dict(tested=len(report),eligible=len(winners),cheap=policy,dense=dense,
               diagnostics=ceiling,sources={k:checksum(DEST/k/'results.json') for k in PARTS},
               csv_sha256=checksum(table),code_sha256=checksum(Path(__file__))))
    print('exchanges',len(report),'cheap',policy,'dense',dense,flush=True)


if __name__=='__main__':
    main()
