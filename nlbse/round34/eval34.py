"""Compare regional teacher scores and matched ranking at a shared FP budget."""
import argparse
from itertools import product

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sections34 import HERE, R33, PARTS, POINTS, DEST as VIEWS, NAMES, support, checksum, json_at, write_json, f
from experiment33 import read_phase, crop, cut
from rescue33 import descriptors
from eval32 import per_project

DEST=HERE/'comparison'
WIDTHS=(.5,1.,1.8,None)
MIX=(.35,.7)
STYLES=('compact','head','regions','pair_compact','pair_regions')
FAMILY=('fit','oof')


def loaded(rotation):
    base=POINTS/rotation
    receipt=json_at(base/'receipt.json')
    folder=VIEWS/rotation
    proof=json_at(folder/'receipt.json')
    if proof['code_sha256']!=checksum(HERE/'sections34.py') or proof['points_sha256']!=checksum(base/'receipt.json'):
        raise ValueError('The per-window reference changed')
    archive=folder/'scores.npz'
    assert checksum(archive)==proof['predictions_sha256']
    data={}
    with np.load(archive) as scores:
        for part in receipt['phases']:
            full=read_phase(part,base,receipt)
            good=support(full)
            ids=full['ids'][good]
            assert np.array_equal(ids,scores[part+'_ids'])
            full['support']=good
            teacher=np.zeros((len(good),len(NAMES)))
            teacher[good]=scores[part]
            sizes=np.zeros(len(good),np.int32)
            sizes[good]=scores[part+'_lengths']
            full.update(teacher=teacher,tokens=sizes)
            data[part]=full
    return data,receipt,proof


def measures(data,threshold):
    low=descriptors(data,threshold)
    window=data['teacher']
    a,b,c,d=window.T
    # Keep the signed whole-function logit as a feature when testing additional windows.
    high=np.column_stack((low,a,b-a,c-a,d-a,np.max(window,1)-a,np.ptp(window,axis=1),
                          (data['tokens']>380),np.log1p(data['tokens'])))
    return low,np.column_stack((low,a)),high


def signatures(rotation,data):
    folder=R33/'matched_veto'
    source=folder/f'{rotation}.npz'
    result=json_at(folder/'results.json')['analysis'][rotation]
    if checksum(source)!=result['predictions_sha256']:
        raise ValueError('The matched examples changed')
    ids=data['ids']
    index={e:i for i,e in enumerate(ids)}
    with np.load(source) as pairs:
        pos=np.asarray([index[k] for k in pairs['positive']])
        neg=np.asarray([index[k] for k in pairs['negative']])
        assert np.array_equal(data['fold'][pos],pairs['fold'])
    assert data['support'][pos].all() and data['support'][neg].all()
    assert data['y'][pos].all() and not data['y'][neg].any()
    return pos,neg


def paired_classifier(matrix,a,b):
    delta=matrix[a]-matrix[b]
    _,inverse,total=np.unique(a,return_inverse=True,return_counts=True)
    weights=np.tile(1./total[inverse],2)
    x=np.concatenate((delta,-delta))
    y=np.arange(len(x))<len(delta)
    model=make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(C=.35,solver='liblinear',fit_intercept=False,random_state=42))
    return model.fit(x,y,logisticregression__sample_weight=weights)


def point_classifier(x,y,q,support):
    model=make_pipeline(StandardScaler(),
                        LogisticRegression(C=.45,solver='liblinear',random_state=42,max_iter=200))
    weight=.3+2.*np.exp(-.8*abs(q[support]))
    return model.fit(x[support],y[support],logisticregression__sample_weight=weight)


def average_win(x,a,b):
    _,inverse=np.unique(a,return_inverse=True)
    agreeing=(x[a]>x[b]).astype(float)+.5*(x[a]==x[b])
    return float((np.bincount(inverse,weights=agreeing)/np.bincount(inverse)).mean())


def rank_diagnostics(oof,a,b,bands,first):
    heads=oof['teacher']
    ar={}
    for k,h in [('original',oof['queried']),*zip(NAMES,heads.T,strict=True),('peak',heads.max(1)),
                ('mean',heads.mean(1))]:
        ar[k]=average_win(h,a,b)
    use=first>380
    if use.any():
        ar['long_same_pairs']={k:average_win(h,a[use],b[use]) for k,h in
                               [('original',oof['queried']),*zip(NAMES,heads.T,strict=True),('peak',heads.max(1))]}
        ar['long_pairs']=int(use.sum())
    forecasts={k:np.empty(len(a)) for k in ('pair_compact','pair_regions')}
    old={k:np.empty(len(a)) for k in forecasts}
    for k in np.unique(oof['fold'][a]):
        held=oof['fold'][a]==k
        keep=(oof['duplicates'][a] & (1<<int(k))==0)&(oof['duplicates'][b] & (1<<int(k))==0)
        assert not (held&keep).any()
        for name,source in [('pair_compact',bands[0]),('pair_regions',bands[2])]:
            m=paired_classifier(source,a[keep],b[keep])
            forecasts[name][held]=m.decision_function(source[a[held]])
            old[name][held]=m.decision_function(source[b[held]])
    _,inverse=np.unique(a,return_inverse=True)
    for k,v in forecasts.items():
        bits=(v>old[k])+.5*(v==old[k])
        ar[k]=float((np.bincount(inverse,weights=bits)/np.bincount(inverse)).mean())
    return ar


def adjusted(y,old,scope,rank,threshold):
    pos=old.copy()
    pos[scope]=rank>threshold
    before=y&old
    res=f.comparison(y,old,pos)
    res.update(recovered=int((y&~old&pos).sum()),lost=int((before&~pos).sum()))
    return res,pos


def evaluate(rotation):
    phases,ref,views=loaded(rotation)
    d=phases['dev']
    t=phases.get('test',d)
    oof=phases['oof']
    left,right,fit=crop(d,d['cal']),crop(t,t['held']),crop(d,d['fit'])
    threshold=ref['cutoff']
    a,b=signatures(rotation,oof)
    features_oof=measures(oof,threshold)
    pair=rank_diagnostics(oof,a,b,features_oof,oof['tokens'][a])
    cal,test,inner=[measures(data,threshold) for data in (left,right,fit)]
    truth,gold=left['y'],right['y']
    baseline0,baseline1=left['replay']>0.,right['replay']>0.
    bank={}
    for source,data,z in [('fit',fit,inner),('oof',oof,features_oof)]:
        bank[source]={mode:point_classifier(x,data['y'],data['queried'],data['support'])
                      for mode,x in zip(STYLES[:3],z,strict=True)}
    for name,feat in [('pair_compact',features_oof[0]),('pair_regions',features_oof[2])]:
        body=paired_classifier(feat,a,b)
        cal_shape=inner[0] if name=='pair_compact' else inner[2]
        calib=body.decision_function(cal_shape)[:,None]
        map_fit=fit['support']
        fit_y=fit['y']
        fuse=point_classifier(calib,fit_y,fit['queried'],map_fit)
        bank['oof'][name]=body,fuse
    orders={}
    for source,r in bank.items():
        for m,net in r.items():
            if m.startswith('pair_'):
                ix=0 if m=='pair_compact' else 2
                a1,b1=[net[1].decision_function(net[0].decision_function(vec[ix])[:,None]) for vec in (cal,test)]
            else:
                ix=STYLES.index(m)
                a1,b1=[net.decision_function(vec[ix]) for vec in (cal,test)]
            orders[source,m]=(a1,b1)
    result={}
    save=dict(ids=right['ids'],truth=gold,baseline=baseline1,available=right['support'],
              orig_query=right['queried'],length=right['tokens'])
    for width in WIDTHS:
        ua,ub=left['support'].copy(),right['support'].copy()
        if width is not None:
            ua &= abs(left['queried'])<width
            ub &= abs(right['queried'])<width
        name='all' if width is None else f'{width:g}'
        keys=[('control','threshold',0.)]+[(kind,mode,p) for kind,mode in orders for p in MIX]
        for kind,mode,amount in keys:
            p,q=left['queried'][ua],right['queried'][ub]
            if amount:
                oa,ob=orders[kind,mode]
                p=(1.-amount)*p+amount*oa[ua]
                q=(1.-amount)*q+amount*ob[ub]
            option=dict(width=width,model=mode,source=kind,strength=amount,calls=int(ub.sum()),cal_calls=int(ua.sum()))
            key=f'{name}_{kind}_{mode}_{amount:g}'
            if rotation!='official':
                option['matched'],bits=per_project(gold,right['project'],baseline1,ub,q,('tp','fp'))
                for k,v in bits.items():
                    save[f'{key}_matched_{k}']=v
            for rule in ('fixed','tp','fp'):
                boundary=cut(truth[ua],baseline0[ua],p,rule)
                c,_=adjusted(truth,baseline0,ua,p,boundary)
                re,val=adjusted(gold,baseline1,ub,q,boundary)
                option[rule]=dict(cut=boundary,cal=c,held=re)
                save[f'{key}_{rule}']=val
            result[key]=option
    folder=DEST/rotation
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictions.npz'
    np.savez_compressed(path,**save)
    weights=folder/'heads.joblib'
    joblib.dump(bank,weights,compress=3)
    write_json(folder/'results.json',dict(rotation=rotation,point_receipt=ref['phases'],views_sha256=views['predictions_sha256'],
               predictions_sha256=checksum(path),weights_sha256=checksum(weights),
               code_sha256=checksum(HERE/'eval34.py'),pairs=pair,options=result))
    print(rotation,'pairs',pair,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    evaluate(parser.parse_args().rotation)
