"""Freeze the complementary additions and veto recovery on internal rows."""
import json

import numpy as np

from select33 import HERE, ROOT, PARTS, read, sha

ADD=HERE/'separate'
VETO=HERE/'veto'


def gain(row):
    return row['f1_after']-row['f1_before']


def proof(base, code, weight_field):
    result=[]
    for p in PARTS:
        path=base/p
        log=read(path/'results.json')
        assert log['code_sha256']==sha(HERE/code)
        assert log['predictions_sha256']==sha(path/'predictions.npz')
        assert log[weight_field]==sha(path/'models.joblib')
        assert log['points_sha256']==sha(HERE/'points'/p/'receipt.json')
        if base==ADD:
            view=HERE/'views'/p
            view_data=read(view/'receipt.json')
            assert log['view_sha256']==view_data['predictions_sha256']==sha(view/'scores.npz')
            assert view_data['code_sha256']==sha(HERE/'views33.py')
        result.append(log)
    return result


def added(records,scanner_seconds):
    aa,bb,oo=[t['settings'] for t in records]
    compare='released_fit_0_0.9_0'
    possible=[]
    for key,chosen in oo.items():
        if chosen['mode']!='add':
            continue
        # Rank-only policy comparison always uses the published threshold.
        effects=chosen['options']['fixed']
        cost=[]
        for name in ('calibration','review'):
            q=chosen[name]
            ratio=max(q['calls']/q['old_calls'],q['characters']/q['old_characters'])-1.
            cost.append(.0002*ratio*scanner_seconds)
        margin_cal=.3*gain(effects['cal'])-cost[0]
        margin_held=.3*gain(effects['frozen'])-cost[1]
        boost=[]
        for group in (aa,bb):
            old=group[compare]['options']['matched_fp']
            matched=group[key]['options']['matched_fp']
            if not (old['exact'] and matched['exact']):
                boost.append(-1)
            else:
                boost.append(matched['overall']['delta_tp']-old['overall']['delta_tp'])
        if min(boost)>0 and margin_cal>0 and margin_held>=0:
            possible.append(dict(key=key,criterion='fixed',a_tp=boost[0],b_tp=boost[1],source=chosen['source'],
                                 extra=chosen['extra'],weight=chosen['weight'],fraction=chosen['frac'],
                                 project_cal_gain=margin_cal,project_review_gain=margin_held,
                                 cal=effects['cal'],review=effects['frozen'],cal_budget=chosen['calibration'],
                                 held_budget=chosen['review']))
    return max(possible,key=lambda r:(r['project_cal_gain'],r['project_review_gain'],not r['extra']),default=None)


def veto(records):
    a,b,c=[t['options'] for t in records]
    options=[]
    for k,row in c.items():
        if row['source']=='shared' or k not in a or k not in b:
            continue
        w='all' if row['width'] is None else f"{row['width']:g}"
        reference=f"{row['route']}_{w}_control_shared_0"
        e=[]
        for held in (a,b):
            sign=held[k]['options']['matched']['fp']['change']
            comparator=held[reference]['options']['matched']['fp']['change']
            e.append(sign['delta_tp']-comparator['delta_tp'])
        p=row['options']['fp']
        if min(e)>0 and gain(p['cal'])>0. and p['cal']['delta_fp']==0 and gain(p['frozen'])>=0.:
            options.append(dict(key=k,criterion='fp',a_tp=e[0],b_tp=e[1],source=row['source'],cal=p['cal'],
                                review=p['frozen'],threshold=p['cut'],route=row['route'],width=row['width'],
                                strength=row['strength'],style=row['style'],cal_rows=row['cal_rows'],
                                review_rows=row['review_rows']))
    return max(options,key=lambda r:(gain(r['cal']),gain(r['review']),r['a_tp']+r['b_tp'],-r['review_rows']),default=None)


def count(y,b):
    tp,fp,fn=int((b&y).sum()),int((b&~y).sum()),int((~b&y).sum())
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn))


def verify_disjoint(adding,guarding):
    output={}
    for p in PARTS:
        with np.load(ADD/p/'predictions.npz') as x,np.load(VETO/p/'predictions.npz') as z:
            assert np.array_equal(x['ids'],z['ids']) and np.array_equal(x['y'],z['truth'])
            before=x['base']
            assert np.array_equal(before,z['old'])
            q=x[adding['key']+'_'+adding['criterion']]
            w=z[guarding['key']+'_'+guarding['criterion']]
            assert np.array_equal(q[x['old']],before[x['old']])
            assert np.array_equal(w[~x['old']],before[~x['old']])
            new=np.where(x['old'],w,q)
            output[p]=dict(released=count(x['y'],before),addition=count(x['y'],q),
                           veto=count(x['y'],w),joint=count(x['y'],new))
    return output


def main():
    test=proof(ADD,'acquire33.py','model_sha256')
    ranks=proof(VETO,'rescue33.py','models_sha256')
    benchmark=read(ROOT/'round32/public/results.json')['results']['candidate']['cost']
    add=added(test,benchmark['scan_seconds'])
    proof_veto=veto(ranks)
    matched=verify_disjoint(add,proof_veto) if add is not None and proof_veto is not None else {}
    candidate=dict(addition=add,veto=proof_veto,held_frozen_combination=matched,
                   inputs={str(x.name):{p:sha(x/p/'results.json') for p in PARTS} for x in (ADD,VETO)})
    (HERE/'joint_selection.json').write_text(json.dumps(candidate,indent=2)+'\n')
    print(json.dumps(candidate,indent=2))


if __name__=='__main__':
    main()
