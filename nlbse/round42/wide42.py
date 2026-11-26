"""Fit a small additive rank correction with broader, score-matched within-file pairs."""
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch

from fit42 import (HERE,ROOT,PARTS,R41,checksum,crop,f,json_at,load,parts,supported,
                  write_json,fingerprint,positions,matched,design)
from evaluate41 import scope, f1, matched as budget_gain
from stack40 import assemble, diagnostic_order, cut, adjusted, per_project

DEST = HERE/'wide'
ARMS = ('strict','broad','broad_gold','broad_contrast')
WIDTHS = (1.,2.5,5.)
MIXES = (.25,.65,1.)
PANELS = {'mixed':None,'negative':0,'unobserved':2}


def neighbors(frame,eligible,corpus):
    groups = defaultdict(lambda:([],[]))
    rank = frame['signed']
    size = np.log1p(frame['length'])
    use = eligible & (abs(rank)<6.)
    for i in np.flatnonzero(use):
        row = frame['ids'][i]
        key = (corpus[2][row],corpus[3][row],int(frame['observed_regime'][i]))
        groups[key][int(frame['y'][i])].append(i)
    pos,neg = [],[]
    for negatives,positives in groups.values():
        if not negatives or not positives:
            continue
        x = np.array(negatives)
        n = np.argsort(rank[x],kind='stable')
        x = x[n]
        lookup = rank[x]
        for a in positives:
            center = np.searchsorted(lookup,rank[a])
            near = x[max(0,center-7):center+7]
            close = near[abs(rank[near]-rank[a])<2.5]
            if not len(close):
                continue
            distance = abs(rank[close]-rank[a])+.6*abs(size[close]-size[a])
            keep = close[np.lexsort((frame['ids'][close],distance))[:4]]
            pos.extend([a]*len(keep))
            neg.extend(keep)
    pairs = np.column_stack((pos,neg)).astype(np.int64)
    assert (frame['fold'][pairs[:,0]]==frame['fold'][pairs[:,1]]).all()
    return pairs


def regression(features,base,graph,unix,pairs,kind,broad):
    data = torch.as_tensor(features,device='cuda',dtype=torch.float32)
    center = data.mean(0)
    spread = data.std(0).clamp_min(.045)
    data = ((data-center)/spread).clamp(-8.,8.)
    rank = base[pairs[:,0]]-base[pairs[:,1]]
    g = graph[pairs[:,0]]-graph[pairs[:,1]]
    u = unix[pairs[:,0]]-unix[pairs[:,1]]
    if kind=='broad_contrast':
        confident = (g*u>0.) & ((g+u)*rank<=0.)
    else:
        confident = (g>0.)&(u>0.)
    if broad:
        pairs,rank,g,u = (m[confident] for m in (pairs,rank,g,u))
    a,b = pairs.T
    diff = data[a]-data[b]
    t = 1.2 if kind=='broad_gold' else .7*(g+u)*.5+.3*np.sign(g)*np.minimum(abs(g),abs(u))
    value = np.clip(t-rank,-4.,4.)
    if kind=='broad_contrast':
        # Teacher disagreement with the order can also identify stale supervision.
        value = np.where(g<0.,np.maximum(-1.25,value),value)
    _,inv,count = np.unique(a,return_inverse=True,return_counts=True)
    w = (1./count[inv])*(.55+.45*np.exp(-abs((base[a]+base[b])*.5)))
    w /= w.sum()
    weights = torch.as_tensor(w,device='cuda',dtype=torch.float32)
    labels = torch.as_tensor(value,device='cuda',dtype=torch.float32)
    # Penalize changes on all first-pass points and shrink the feature weights.
    gram = diff.T@(diff*weights[:,None])+.33*(data.T@data/len(data))
    beta = torch.linalg.solve(gram+.12*torch.eye(data.shape[1],device='cuda'),diff.T@(weights*labels))
    return dict(center=center.cpu().numpy(),scale=spread.cpu().numpy(),coef=beta.cpu().numpy(),
                rows=len(pairs),anchors=len(count),negative_teacher_order=int((g<0.).sum()),
                mean_teacher_margin=float((.5*(g+u)).mean()))


def main(rotation):
    start = perf_counter()
    matrices,t,ticket = load(rotation)
    raw = (matrices['oof'],*parts(matrices))
    rows = [crop(v,supported(v)) for v in raw]
    o = rows[0]
    corpus = f.inputs()
    teachers,states = (R41/k/rotation for k in ('heads','compact'))
    a,b = (json_at(p/'receipt.json') for p in (teachers,states))
    assert a['matrix_sha256']==b['matrix_sha256']==ticket
    assert checksum(teachers/'scores.npz')==a['predictions_sha256']
    assert checksum(states/'states.npz')==b['state_sha256']==a['compact_sha256']
    zz = np.load(teachers/'scores.npz')
    cc = np.load(states/'states.npz')
    assert np.array_equal(zz['oof_ids'],o['ids'])
    excluded = np.unique(np.r_[matrices['dev']['ids'],matrices.get('test',matrices['dev'])['ids']])
    copies = {fingerprint(corpus[0][i]) for i in excluded}
    permitted = np.fromiter((fingerprint(corpus[0][i]) not in copies for i in o['ids']),bool)
    assert int((~permitted).sum())==a['removed_validation_copies']
    teach = {k:zz['oof_'+k] for k in ('graph','unix')}
    strict,_,proof = matched(rotation,o,permitted,teach,zz['oof_ids'],corpus)
    broad = neighbors(o,permitted,corpus)
    matrix = []
    for row,role in zip(rows,('oof','dev','test' if 'test' in matrices else 'dev'),strict=True):
        order = positions(cc[role+'_ids'],row['ids'])
        assert np.array_equal(cc[role+'_ids'][order],row['ids'])
        matrix.append(design(row,t,cc[role][order]))
    print(rotation,'strict',len(strict),'broad',len(broad),flush=True)
    model = {}
    scores = {k:[np.full(len(v['ids']),np.nan,np.float32) for v in rows[1:]] for k in ARMS}
    ownership = o['observed_regime']
    for side in (0,2):
        use = (ownership==side)&permitted
        for kind in ARMS:
            source = strict if kind=='strict' else broad
            source = source[use[source].all(1)]
            local = np.searchsorted(np.flatnonzero(use),source)
            head = regression(matrix[0][use],o['signed'][use],teach['graph'][use],teach['unix'][use],
                              local,kind,kind!='strict')
            model[kind,side] = head
            for row,x,q in zip(rows[1:],matrix[1:],scores[kind],strict=True):
                mask = row['observed_regime']==side
                q[mask] = ((x[mask]-head['center'])/head['scale']).clip(-8.,8.)@head['coef']
            print(rotation,side,kind,{k:v for k,v in head.items() if k not in ('center','scale','coef')},flush=True)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    joblib.dump(model,out/'weights.joblib',compress=3)
    np.savez_compressed(out/'roster.npz',strict=o['ids'][strict],broad=o['ids'][broad],
                        strict_fold=o['fold'][strict[:,0]],broad_fold=o['fold'][broad[:,0]])
    bank = {}
    for side,row in zip(('cal','held'),rows[1:],strict=True):
        bank[side+'_ids'] = row['ids']
    for k,(q,p) in scores.items():
        for side,delta in zip(('cal','held'),(q,p),strict=True):
            assert np.isfinite(delta).all()
            bank[side+'_'+k] = delta
    np.savez_compressed(out/'scores.npz',**bank)
    meta = {f'{k}/{side}':{name:v for name,v in result.items() if name not in ('center','scale','coef')}
            for (k,side),result in model.items()}
    write_json(out/'fit.json',dict(matrix_sha256=ticket,fit_sources=[checksum(p/'receipt.json') for p in (teachers,states)],
               pair_source_sha256=proof,hashes={k:checksum(out/k) for k in ('roster.npz','weights.joblib','scores.npz')},
               notes=meta,code_sha256=checksum(Path(__file__)),seconds=perf_counter()-start))


def evaluate(rotation):
    matrices,t,ticket = load(rotation)
    frames = parts(matrices)
    folder = DEST/rotation
    receipt = json_at(folder/'fit.json')
    assert receipt['matrix_sha256']==ticket
    assert receipt['code_sha256']==checksum(Path(__file__))
    assert all(checksum(folder/k)==v for k,v in receipt['hashes'].items())
    z = np.load(folder/'scores.npz')
    old = ROOT/'round40/comparison'/rotation
    assert checksum(old/'models.joblib')==json_at(old/'results.json')['weights_sha256']
    coeff = joblib.load(old/'models.joblib')['calibration']
    sources = [assemble(a,t,'all',coeff)[3] for a in frames]
    ranks = [a['signed'] for a in frames]
    truth = [a['y'] for a in frames]
    decisions = [a>0. for a in ranks]
    preds = dict(ids=frames[1]['ids'],base=decisions[1],y=truth[1])
    predictions = {}
    for arm in ARMS:
        values = []
        for role,row in zip(('cal','held'),frames,strict=True):
            ok = supported(row)
            assert np.array_equal(z[role+'_ids'],row['ids'][ok])
            price = np.zeros(len(ok),np.float32)
            price[ok] = z[role+'_'+arm]
            values.append(price)
        predictions[arm] = values
    results = {}
    for kind,panel,width in product(('open','lexical'),PANELS,WIDTHS):
        scopes = [scope(a,e,kind,PANELS[panel],width) for a,e in zip(frames,sources,strict=True)]
        if not all(q.any() for q in scopes):
            continue
        for arm in ('threshold',*ARMS):
            for w in ((0.,) if arm=='threshold' else MIXES):
                part = [r[s]+w*d[s] for r,s,d in zip(ranks,scopes,predictions.get(arm,ranks),strict=True)]
                key = f'{kind}/{panel}/{width}/{arm}/{w:g}'
                entry = dict(kind=kind,panel=panel,width=width,arm=arm,mix=w,
                             cal_rows=int(scopes[0].sum()),review_rows=int(scopes[1].sum()))
                if rotation!='official':
                    q,_ = diagnostic_order(part[1],ranks[1][scopes[1]],frames[1]['ids'][scopes[1]])
                    entry['matched'],dd = per_project(truth[1],frames[1]['project'],decisions[1],scopes[1],q,('fp','tp'))
                    preds[key+'_matched'] = dd['fp']
                for rule in ('fixed','fp','tp'):
                    edge = cut(truth[0][scopes[0]],decisions[0][scopes[0]],part[0],rule)
                    c,_ = adjusted(truth[0],decisions[0],scopes[0],part[0],edge)
                    h,q = adjusted(truth[1],decisions[1],scopes[1],part[1],edge)
                    entry[rule] = dict(cut=edge,cal=c,held=h)
                    preds[key+'_'+rule] = q
                results[key] = entry
    np.savez_compressed(folder/'predictions.npz',**preds)
    write_json(folder/'results.json',dict(variants=results,fit_sha256=checksum(folder/'fit.json'),
                     predictions_sha256=checksum(folder/'predictions.npz'),code_sha256=checksum(Path(__file__))))
    return results


def compare():
    a,b,d = (evaluate(k) for k in PARTS)
    rows = []
    for k in sorted(set(a)&set(b)&set(d)):
        q = d[k]
        if not q['mix']:
            continue
        prefix,arm,strength = k.rsplit('/',2)
        ctl = prefix+'/threshold/0'
        narrow = prefix+'/strict/'+strength
        data = {}
        for rule in ('fp','tp'):
            raw = [budget_gain(t[k],rule) for t in (a,b)]
            data[rule] = (raw,[x-budget_gain(t[ctl],rule) for x,t in zip(raw,(a,b),strict=True)],
                        [x-budget_gain(t[narrow],rule) for x,t in zip(raw,(a,b),strict=True)])
        for rule in ('fixed','fp','tp'):
            c,h = q[rule]['cal'],q[rule]['held']
            cal,review = f1(c),f1(h)
            floor = max(0.,f1(d[ctl][rule]['cal']))
            if arm!='strict':
                floor = max(floor,f1(d[narrow][rule]['cal']))
            matched = any(min(x)>=0 and sum(x)>0 and (arm=='strict' or (min(y)>=0 and sum(y)>0))
                          for _,x,y in data.values())
            rows.append(dict(key=k,rule=rule,setting={j:v for j,v in q.items() if j not in ('matched','fixed','fp','tp')},
                   threshold=q[rule]['cut'],budget=data,cal_f1=cal,review_f1=review,
                   cal_tp=c['delta_tp'],cal_fp=c['delta_fp'],review_tp=h['delta_tp'],review_fp=h['delta_fp'],
                   matched=matched,eligible=matched and cal>floor and review>=0.))
    key = lambda q:(min(q['cal_f1'],q['review_f1']),q['cal_f1']+q['review_f1'],-q['setting']['review_rows'])
    chosen = max((q for q in rows if q['eligible']),key=key,default=None)
    notable = max((q for q in rows if q['matched']),key=key,default=None)
    write_json(DEST/'selection.json',dict(chosen=chosen,notable=notable,rows=rows,
              sources={k:checksum(DEST/k/'results.json') for k in PARTS},code_sha256=checksum(Path(__file__))))
    print('wide',len(rows),'chosen',chosen,'notable',notable,flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='compare':
        compare()
    else:
        main(sys.argv[1])
