"""Learn corrections for disagreements between original and formatted W annotations."""
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'round49'),str(ROOT/'round39')]
from route49 import PARTS, LAYOUT, f, checksum, json_at, load, physical, write_json
from source41 import parts, crop
from components49 import DEST as LABELS
from context39 import profile
from layout49 import layout
from policy33 import logit, matrix
from project_split import fingerprint
from rank38 import fit as fit_linear, predict as linear
from stack40 import diagnostic_order, per_project
from eval35 import adjusted, cut

DEST=HERE/'models'
LOC=ROOT/'round39/locations'
VIEWS=('numeric','location','structure')
HEADS=('direct','conditional','linear_conditional','paired')
PANELS=('positive','negative','observed')
WIDTHS=(1.25,3.8,None)
BLENDS=(.35,.70,1.)
PARAM=dict(max_iter=120,max_leaf_nodes=19,min_samples_leaf=55,learning_rate=.054,
           l2_regularization=10.,early_stopping=False,random_state=150)


def visible(frame):
    return frame['old'] & (frame['flag']>=0)


def aligned(ids, location_ids, location):
    take=np.searchsorted(location_ids,ids)
    assert np.array_equal(location_ids[take],ids)
    return location[take]


def descriptors(case, threshold, geometry, views, texts=None, anatomy=None):
    first=geometry[:,1]
    score=case['signed']
    lead=matrix(case['pre'],case['length'],threshold)
    common=np.column_stack((score.clip(-15.,15.),lead[:,:15],
                            first[:,None]==np.arange(1,21),np.log1p(geometry[:,2])))
    estimates={}
    if 'numeric' in views:
        estimates['numeric']=common
    if any(v!='numeric' for v in views):
        whole=np.log1p(geometry[:,8])
        place=np.column_stack((geometry[:,3:6]/np.maximum(geometry[:,8,None],1),
                               np.log1p(geometry[:,6:8]),whole))
        context=np.stack([profile(t,row)[0] for t,row in zip(texts,geometry,strict=True)])
        design=np.column_stack((common,place,context))
        if 'location' in views:
            estimates['location']=design
        if 'structure' in views:
            if anatomy is None:
                anatomy=np.stack([layout(t) for t in texts])
            estimates['structure']=np.column_stack((design,anatomy))
    return {k:v.astype(np.float32) for k,v in estimates.items()}


def pairs(case, files, rule, omit, fold):
    # The two rows share the source file, scanner outcome and first rule.
    # Keep both the length and baseline margin close to limit confounding.
    size=np.log1p(case['length'])
    margin=case['signed']
    groups=defaultdict(list)
    for i,k in enumerate(zip(files,rule,fold,strict=True)):
        groups[k].append(i)
    result=[]
    for rows in groups.values():
        idx=np.asarray(rows)
        a,b=idx[omit[idx]],idx[~omit[idx]]
        if not len(a) or not len(b):
            continue
        b=b[np.argsort(margin[b],kind='stable')]
        for i in a:
            middle=np.searchsorted(margin[b],margin[i])
            q=b[max(0,middle-24):middle+24]
            gap,delta=abs(size[q]-size[i]),abs(margin[q]-margin[i])
            feasible=(gap<=.70)&(delta<=1.)
            if feasible.any():
                candidates=q[feasible]
                choice=np.argmin(gap[feasible]+.75*delta[feasible])
                result.append((i,int(candidates[choice])))
    return np.asarray(result,dtype=np.int64).reshape(-1,2).T


def fit_heads(arr,case,w,e,groups,rule,origins):
    y=case['y']
    flag=case['flag']==1
    models,pair_bank={},{}
    diagnostics={}
    for name,x in arr.items():
        models[name]={}
        for kind in (False,True):
            allowed=flag==kind
            selected=np.flatnonzero(allowed)
            rank=case['signed'][allowed]
            omitted=w[allowed]!=kind
            target=y[allowed]
            assert np.array_equal(target,w[allowed]|e[allowed])
            weight=.5+1.7*np.exp(-.34*abs(rank))
            field=x[allowed]

            def tree(values,focus=None):
                k=np.ones(len(values),bool) if focus is None else focus
                return HistGradientBoostingClassifier(**PARAM).fit(field[k],values[k],sample_weight=weight[k])

            if kind not in pair_bank:
                keys=groups[selected]
                a,b=pairs(crop(case,allowed),keys,rule[allowed],omitted,origins[allowed])
                pair_bank[kind]=(a,b)
                diagnostics[str(int(kind))]=dict(rows=len(allowed.nonzero()[0]),disagreement=int(omitted.sum()),
                        historical_only=int((w[allowed]&~e[allowed]).sum()),pairs=len(a),
                        unique_concordant=int(len(np.unique(b))),external_without_W=int(((~w[allowed])&e[allowed]).sum()))
            a,b=pair_bank[kind]
            assert len(a)>20
            # Use oriented margins as a weak prior for the direction of correction.
            oriented=(1. if not kind else -1.)*rank
            direct=tree(target)
            typed=tree(omitted)
            other=tree(e[allowed],~w[allowed])
            plain=fit_linear(field,oriented,omitted,a[:0],b[:0])
            matched=fit_linear(field,oriented,omitted,a,b)
            models[name][kind]=dict(direct=direct,conditional=typed,linear_conditional=plain,paired=matched,external=other)
    return models,diagnostics


def forecast(model, matrices, case):
    weak,rank={},{}
    flag=case['flag']==1
    for mode,x in matrices.items():
        for modelname in HEADS:
            rank[mode,modelname]=np.empty(len(x))
            weak[mode,modelname]=np.full(len(x),np.nan)
        for state in (False,True):
            m=flag==state
            r=case['signed'][m]
            design=x[m]
            main=model[mode][state]
            prob=main['external'].predict_proba(design)[:,1]
            rank[mode,'direct'][m]=main['direct'].decision_function(design)
            for h in ('conditional','linear_conditional','paired'):
                head=main[h]
                if h=='conditional':
                    d=head.predict_proba(design)[:,1]
                else:
                    d=expit(linear(head,design,(-1. if state else 1.)*r))
                w=1.-d if state else d
                rank[mode,h][m]=logit(w+(1.-w)*prob)
                weak[mode,h][m]=w
    return rank,weak


def scope(case, panel, width):
    chosen=visible(case)
    if panel!='observed':
        chosen &= case['flag']==(1 if panel=='positive' else 0)
    if width is not None:
        chosen &= abs(case['signed'])<width
    return chosen


def comparisons(rotation,parts_full,estimates):
    a,b=parts_full
    gold=[r['y'] for r in parts_full]
    before=[r['signed']>0. for r in parts_full]
    snapshot=dict(ids=b['ids'],truth=gold[1],base=before[1],cal_ids=a['ids'])
    result={}
    for panel,width in product(PANELS,WIDTHS):
        masks=[scope(r,panel,width) for r in parts_full]
        if not all(m.sum()>10 for m in masks):
            continue
        for view,head,blend in [*product(VIEWS,HEADS,BLENDS),('numeric','threshold',0.)]:
            tag=f'{panel}/{width}/{view}/{head}/{blend:g}'
            source=estimates.get((view,head),[r['signed'] for r in parts_full])
            q=[(1.-blend)*r['signed'][m]+blend*o[m]
               for r,m,o in zip(parts_full,masks,source,strict=True)]
            entry=dict(view=view,head=head,blend=blend,panel=panel,width=width,
                       cal_rows=int(masks[0].sum()),held_rows=int(masks[1].sum()))
            if rotation!='official':
                order,_=diagnostic_order(q[1],b['signed'][masks[1]],b['ids'][masks[1]])
                entry['matched'],p=per_project(gold[1],b['project'],before[1],masks[1],order,('fp','tp'))
                for mode,bits in p.items():
                    snapshot[tag+'_same_'+mode]=bits
            for mode in ('fixed','fp','tp'):
                crossing=cut(gold[0][masks[0]],before[0][masks[0]],q[0],mode)
                base,_=adjusted(gold[0],before[0],masks[0],q[0],crossing)
                eval_,saved=adjusted(gold[1],before[1],masks[1],q[1],crossing)
                entry[mode]=dict(cut=crossing,cal=base,held=eval_)
                snapshot[tag+'_'+mode]=saved
            result[tag]=entry
    return result,snapshot


def inputs(rotation):
    sources,t,stamp=load(rotation)
    train=sources['oof']
    frames=parts(sources)
    d,h=sources['dev'],sources.get('test',sources['dev'])
    all_held=np.unique(np.r_[d['ids'],h['ids']])
    ids=train['ids']
    assert not np.intersect1d(all_held,ids).size
    texts,truth,project,file,_=f.inputs()
    if rotation!='official':
        assert not set(project[ids])&set(project[h['ids']])
    no_use={fingerprint(texts[i]) for i in all_held}
    keep=np.fromiter((fingerprint(texts[i]) not in no_use for i in ids),bool)
    choice=LABELS/'labels.npz'
    label_meta=json_at(choice.with_name('receipt.json'))
    assert label_meta['predictions_sha256']==checksum(choice)
    with np.load(choice) as z:
        assert np.array_equal(z['truth'][ids],train['y'])
        annotations,eligible=z['component'],z['selected']
    roster=LOC/'training.npz'
    link=json_at(LOC/'receipt.json')
    assert link['predictions_sha256']==checksum(roster)
    assert link['points'][rotation]==checksum(ROOT/'round33/points'/rotation/'receipt.json')
    assert link['extended_scanner_sha256']==checksum(ROOT/'round39/scan_location.mjs')
    with np.load(roster) as z:
        loc,positions=z['ids'],z['values']
    layout_source=LAYOUT/'features.npz'
    layout_meta=json_at(LAYOUT/'receipt.json')
    assert layout_meta['matrix_sources'][rotation]==stamp and layout_meta['predictions_sha256']==checksum(layout_source)
    with np.load(layout_source) as z:
        ix,shape=z['ids'],z['layout']

    included=(visible(train)&keep&eligible[ids],*(visible(z) for z in frames))
    reduced=[crop(z,ok) for z,ok in zip((train,*frames),included,strict=True)]
    locations=[aligned(z['ids'],loc,positions) for z in reduced]
    for frame,observed in zip(reduced,locations,strict=True):
        assert np.array_equal(observed[:,0],frame['flag'])
    metrics=[]
    for frame,observed in zip(reduced,locations,strict=True):
        q=frame['ids']
        anatomy=physical(frame,ix,shape)
        metrics.append(descriptors(frame,t,observed,VIEWS,[texts[i] for i in q],anatomy))
    proof=dict(matrix_sha256=stamp,location_sha256=checksum(LOC/'receipt.json'),
               layout_sha256=checksum(LAYOUT/'receipt.json'),components_sha256=checksum(choice.with_name('receipt.json')),
               exclusions=int((~keep).sum()))
    q=reduced[0]['ids']
    fit_names=np.asarray([a+'\0'+b for a,b in zip(project[q],(file[i] for i in q),strict=True)])
    return reduced,frames,included,metrics,locations,annotations,eligible,fit_names,proof


def main(rotation):
    initial=perf_counter()
    selected,frames,masks,features,locations,annotations,eligible,groups,proof=inputs(rotation)
    study=selected[0]
    comp=annotations[study['ids']]
    model,counts=fit_heads(features[0],study,(comp&1)>0,(comp&6)>0,groups,locations[0][:,1],study['fold'])
    results={}
    component_report={}
    truth={}
    for name,z,feat,active in zip(('cal','held'),selected[1:],features[1:],masks[1:],strict=True):
        ranks,w=forecast(model,feat,z)
        for k,measurement in ranks.items():
            full=np.zeros(len(active))
            full[active]=measurement
            results.setdefault(k,[]).append(full)
        known=eligible[z['ids']]
        W=(annotations[z['ids']]&1)>0
        reference=z['flag']==1
        y=z['y']
        component_report[name]=dict(rows=len(y),aligned=int(known.sum()),base_error=int((known&(W!=reference)).sum()),
                base_positive=int((known&W).sum()),heads={})
        truth[name+'_ids']=z['ids']
        truth[name+'_w']=W
        truth[name+'_known']=known
        truth[name+'_scanner']=reference
        for (view,arm),p in w.items():
            if arm=='direct':
                continue
            key=view+'/'+arm
            truth[name+'_'+key]=p
            component_report[name]['heads'][key]=f.comparison(W[known],reference[known],p[known]>.5)
    comparison,predictions=comparisons(rotation,frames,results)
    home=DEST/rotation
    home.mkdir(exist_ok=True,parents=True)
    saved=home/'predictions.npz'
    np.savez_compressed(saved,**predictions)
    targets=home/'annotations.npz'
    np.savez_compressed(targets,**truth)
    received=home/'weights.joblib'
    joblib.dump(model,received,compress=3)
    print(rotation,counts,len(comparison),round(perf_counter()-initial,2),flush=True)
    write_json(home/'results.json',dict(rotation=rotation,fit=counts,report=component_report,results=comparison,proof=proof,
               code_sha256=checksum(Path(__file__)),predictions_sha256=checksum(saved),annotation_sha256=checksum(targets),
               weights_sha256=checksum(received),seconds=perf_counter()-initial))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
