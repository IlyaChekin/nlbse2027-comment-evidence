"""Fit global stacks and disagreement-only selectors on file-disjoint experts."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data40 import HERE, ROOT, PARTS, checksum, json_at, write_json, attach, features as basic
from data38 import crop
from eval35 import cut, per_project, adjusted

DEST=HERE/'comparison'
PANELS=('all','negative','positive','unobserved')
WIDTHS=(.65,1.6,3.8,None)
MIX=(.35,.7,1.)
SETS={'score':slice(0,3),'old':slice(0,4),'new':np.asarray([0,1,2,4]),'all':slice(None)}


def model(x,y,weights,C=.20):
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='liblinear',
                     fit_intercept=True,max_iter=220,random_state=40))
    return clf.fit(x,y,logisticregression__sample_weight=weights)


def univariate(x,group):
    return np.column_stack((x,group[:,None]==np.arange(3),x[:,None]*(group[:,None]==np.arange(3))))


def calibrate(x,groups,y,old):
    hist=[]
    for j in range(x.shape[1]):
        source=univariate(x[:,j],groups)
        readout=model(source,y,np.ones(len(y)),C=.25)
        z=readout.decision_function(source)
        threshold=cut(y,old,z,'fp')
        hist.append((readout,threshold))
    return hist


def bank(source,groups,calibration,which):
    columns=range(source.shape[1])[SETS[which]] if isinstance(SETS[which],slice) else SETS[which]
    return np.column_stack([calibration[p][0].decision_function(univariate(source[:,p],groups))
                             -calibration[p][1] for p in columns])


def assemble(rows,t,which,calibration):
    own,sparse,new,source,groups=basic(rows,t)
    expert=bank(source,groups,calibration,which)
    baseline=rows['signed']
    delta=expert-baseline[:,None]
    crossed=(expert>0.)!=(baseline[:,None]>0.)
    target=np.column_stack((own,expert,delta,crossed,crossed.sum(1),
                           expert.max(1),expert.min(1),expert.mean(1)))
    if which in ('old','all'):
        target=np.column_stack((target,sparse))
    if which in ('new','all'):
        target=np.column_stack((target,new))
    return target,crossed.any(1),groups,expert


def establish(data,limit):
    fitting=data['oof']
    _,_,_,raw,groups=basic(fitting,limit)
    coefficients=calibrate(raw,groups,fitting['y'],fitting['signed']>0.)
    d=data['dev']
    t=data.get('test',d)
    roles=(crop(d,d['cal']),crop(t,t['held']))
    estimates={}
    models={}
    receipts={}
    shift=fitting['signed']
    scale=.4+1.6*np.exp(-.44*abs(shift))
    for which in SETS:
        design,change,_,local=assemble(fitting,limit,which,coefficients)
        ready=[assemble(rows,limit,which,coefficients) for rows in roles]
        for mode in ('stack','switch','limited',*(('tree',) if which=='all' else ())):
            fit=change if mode in ('switch','tree') else np.ones(len(shift),bool)
            if mode=='limited':
                method=models[which,'stack']
            elif mode=='tree':
                method=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=75,
                            learning_rate=.07,l2_regularization=6.,early_stopping=False,random_state=40)
                method.fit(design[fit],fitting['y'][fit],sample_weight=scale[fit])
            else:
                method=model(design[fit],fitting['y'][fit],scale[fit])
            models[which,mode]=method
            results=[]
            for vec,may_edit,group,alternatives in ready:
                available=np.ones(len(vec),bool) if mode=='stack' else may_edit
                results.append((method.decision_function(vec),available,group))
            estimates[which,mode]=results
        receipts[which]=dict(rows=len(change),disagreed=int(change.sum()),
                  fixable_false_positives=int((change & (shift>0.) & ~fitting['y']).sum()),
                  fixable_false_negatives=int((change & (shift<=0.) & fitting['y']).sum()))
    return coefficients,models,estimates,roles,receipts


def aperture(signed,eligible,group,panel,width):
    mask=eligible.copy()
    if panel!='all':
        mask &= group==PANELS.index(panel)-1
    if width is not None:
        mask &= abs(signed)<width
    return mask


def diagnostic_order(score, baseline, ids):
    # This order is used only to compare matched budgets in the labelled
    # reports. Deployable thresholds continue to use the original scores.
    order=np.lexsort((ids,baseline,score))
    rank=np.empty(len(order),np.float64)
    rank[order]=np.arange(len(order))
    return rank,int(np.sum(score[order][1:]==score[order][:-1]))


def compare(rotation, frame, scores):
    cal,held=frame
    values=[x['signed'] for x in frame]
    truth=[x['y'] for x in frame]
    base=[z>0. for z in values]
    forecast=dict(ids=held['ids'],truth=truth[1],base=base[1],rank_before=values[1])
    table={}
    for (which,head), arm in scores.items():
        for panel,width in product(PANELS,WIDTHS):
            area=[aperture(ref,visible,route,panel,width)
                  for ref,(_,visible,route) in zip(values,arm,strict=True)]
            if not all(a.any() for a in area):
                continue
            for strength in (0.,*MIX):
                key='/'.join((which,head,panel,str(width),f'{strength:g}'))
                rank=[(1.-strength)*old[scope]+strength*new[scope]
                      for old,(new,_,_),scope in zip(values,arm,area,strict=True)]
                result=dict(experts=which,model=head,panel=panel,width=width,mix=strength,
                            cal_rows=int(area[0].sum()),held_rows=int(area[1].sum()))
                if rotation!='official':
                    order,duplicates=diagnostic_order(rank[1],values[1][area[1]],held['ids'][area[1]])
                    result['rank_ties']=duplicates
                    result['matched'],bits=per_project(truth[1],held['project'],base[1],area[1],order,('fp','tp'))
                    forecast[key+'_budget_fp']=bits['fp']
                for style in ('fixed','fp','tp'):
                    edge=cut(truth[0][area[0]],base[0][area[0]],rank[0],style)
                    d,_=adjusted(truth[0],base[0],area[0],rank[0],edge)
                    p,z=adjusted(truth[1],base[1],area[1],rank[1],edge)
                    result[style]=dict(cut=edge,cal=d,held=p)
                    forecast[key+'_'+style]=z
                table[key]=result
    return table,forecast


def main(rotation):
    start=perf_counter()
    data,ref,signed=attach(rotation)
    a,b,c,d,proof=establish(data,ref['cutoff'])
    print(rotation,'fit',proof,flush=True)
    report,pred=compare(rotation,d,c)
    where=DEST/rotation
    where.mkdir(exist_ok=True,parents=True)
    path=where/'predictions.npz'
    weights=where/'models.joblib'
    np.savez_compressed(path,**pred)
    joblib.dump(dict(calibration=a,readout=b),weights,compress=3)
    write_json(where/'results.json',dict(rotation=rotation,options=report,sources=signed,fit=proof,
                    weights_sha256=checksum(weights),predictions_sha256=checksum(path),
                    matrix_sha256=checksum(HERE/'data40.py'),code_sha256=checksum(Path(__file__))))
    print(rotation,len(report),'elapsed',perf_counter()-start,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
