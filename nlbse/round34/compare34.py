"""Compare no-extra-encoder readouts with the same FP budget and calibration files."""
import argparse
from itertools import product

import joblib
import numpy as np

from sections34 import HERE, PARTS, f, checksum, json_at, write_json
from eval34 import loaded, crop, descriptors, adjusted, cut, point_classifier, per_project
from distill34 import DEST as DISTILLED
from compact34 import ARMS
from attention34 import DEST as ATTENTION, TRAINING as ATTENTION_ARMS
from approx34 import DEST as APPROX, MODES as APPROX_ARMS

DEST=HERE/'compact_comparison'
SCOPE=(.4,.75,1.,1.8,None)
STRENGTH=(.25,.5,.75,1.)


def design(data,t,logits=None):
    reference=descriptors(data,t)
    if logits is None:
        return reference
    current=logits[:,0].clip(-8.,8.)
    joint=np.column_stack((reference,current,current-data['queried']))
    if logits.shape[1]>1:
        parts=logits[:,1:].clip(-8.,8.)
        joint=np.column_stack((joint,parts,parts.max(1)-current))
    return joint


def stage(rotation, attention=False, approximate=False):
    phases,ref,teacher=loaded(rotation)
    root=(APPROX if approximate else (ATTENTION if attention else DISTILLED))/rotation
    receipt=json_at(root/'receipt.json')
    scores=root/'scores.npz'
    assert checksum(scores)==receipt['predictions_sha256']
    if approximate:
        assert checksum(HERE/'approx34.py')==receipt['code_sha256']
        assert receipt['teacher_sha256']==teacher['predictions_sha256']
        arms={k:2 for k in APPROX_ARMS}
    elif attention:
        assert checksum(HERE/'attention34.py')==receipt['code_sha256']
        assert receipt['teacher_sha256']==teacher['predictions_sha256']
        arms={k:2 for k in ATTENTION_ARMS}
    else:
        assert checksum(HERE/'distill34.py')==receipt['code_sha256']
        assert checksum(HERE/'compact34.py')==receipt['runtime_sha256']
        assert receipt['regional_teacher_sha256']==teacher['predictions_sha256']
        arms={k:v[1] for k,v in ARMS.items()}
    with np.load(scores) as z:
        for role,data in phases.items():
            mask=data['support']
            if not np.array_equal(z[role+'_ids'],data['ids'][mask]):
                raise ValueError('Compact head rows are reordered')
            for mode,dimension in arms.items():
                path=root/f'{mode}.pt'
                assert checksum(path)==receipt['heads'][mode]['sha256']
                x=np.zeros((len(mask),dimension),np.float32)
                x[mask]=z[role+'_'+mode]
                data[mode]=x
    d=phases['dev']
    t=phases.get('test',d)
    fit,cal,held=crop(d,d['fit']),crop(d,d['cal']),crop(t,t['held'])
    y,z=cal['y'],held['y']
    old,base=cal['replay']>0.,held['replay']>0.
    models={}
    rank={}
    threshold=ref['cutoff']
    for mode in ('compact',*arms):
        for fit_src,case in (('fit',fit),('oof',phases['oof'])):
            key=mode,fit_src
            x0=design(case,threshold,case.get(mode))
            head=point_classifier(x0,case['y'],case['queried'],case['support'])
            a,b=[head.decision_function(design(q,threshold,q.get(mode))) for q in (cal,held)]
            rank[key]=(a,b)
            models[key]=head
    masks={}
    out=dict(ids=held['ids'],base=base,eligible=held['support'],truth=z)
    for width in SCOPE:
        active,picked=cal['support'].copy(),held['support'].copy()
        if width is not None:
            active &= abs(cal['queried'])<width
            picked &= abs(held['queried'])<width
        tag='all' if width is None else f'{width:g}'
        keys=[('threshold','shared',0.)]+[(k,j,w) for (k,j),w in product(rank,STRENGTH)]
        for mode,provenance,weight in keys:
            key=f'{tag}_{mode}_{provenance}_{weight:g}'
            p,q=cal['queried'][active],held['queried'][picked]
            if weight:
                x,t=rank[mode,provenance]
                p=(1.-weight)*p+weight*x[active]
                q=(1.-weight)*q+weight*t[picked]
            table=dict(mode=mode,source=provenance,width=width,strength=weight,cal_rows=int(active.sum()),
                       held_rows=int(picked.sum()))
            if rotation!='official':
                table['matched'],pred=per_project(z,held['project'],base,picked,q,('tp','fp'))
                for protocol,value in pred.items():
                    out[key+'_matched_'+protocol]=value
            for scheme in ('fixed','fp','tp'):
                c=cut(y[active],old[active],p,scheme)
                counts,_=adjusted(y,old,active,p,c)
                quality,pred=adjusted(z,base,picked,q,c)
                table[scheme]=dict(cut=c,cal=counts,held=quality)
                out[key+'_'+scheme]=pred
            masks[key]=table
    comparison='approx_comparison' if approximate else ('attention_comparison' if attention else 'compact_comparison')
    dest=HERE/comparison/rotation
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'predictions.npz'
    np.savez_compressed(path,**out)
    ck=dest/'calibrators.joblib'
    joblib.dump(models,ck,compress=3)
    write_json(dest/'results.json',dict(rotation=rotation,options=masks,heads_sha256=checksum(root/'receipt.json'),
              predictions_sha256=checksum(path),weights_sha256=checksum(ck),
              code_sha256=checksum(HERE/'compare34.py'),base_cutoff=threshold))
    print(rotation,'compact options',len(masks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    p.add_argument('--attention',action='store_true')
    p.add_argument('--approximation',action='store_true')
    args=p.parse_args()
    stage(args.rotation,args.attention,args.approximation)
