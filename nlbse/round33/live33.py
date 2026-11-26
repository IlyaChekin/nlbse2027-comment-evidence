"""Measure one frozen replacement router against the full published pipeline."""
import time
from statistics import median

import joblib
import numpy as np

from points33 import HERE, R32, f, checksum, json_at, write_json, strict_cut
from audit29 import canonical_mat, packages
from disagreement import clip_logit
from first_hit32 import scan
from length_eval import recipe
from live32 import load_choice
from pipeline import PUBLIC, load
from policy33 import matrix, prediction, reassign
from positive_runtime import apply as correct_positive, scope
from rank30 import rerank
from subgroups32 import features as group_features
from acquire33 import bounded
from rescue33 import descriptors as veto_features, territory as veto_scope

DEST = HERE/'public'


def load_router():
    source=HERE/'selection.json'
    policy=json_at(source)
    selected=policy['selected']
    if selected is None:
        raise ValueError('The held-out comparison found no supported router')
    evaluations=HERE/'comparison/official'
    signed=evaluations/'results.json'
    if checksum(signed)!=policy['tables']['official']:
        raise ValueError('Router evaluation changed')
    proof=json_at(signed)
    models=evaluations/'models.joblib'
    if checksum(models)!=proof['models_sha256'] or checksum(HERE/'policy33.py')!=proof['policy_sha256']:
        raise ValueError('Frozen router implementation changed')
    bank=joblib.load(models).get(selected['source'])
    return selected,bank,dict(selection_sha256=checksum(source),models_sha256=proof['models_sha256'])


def load_joint():
    path=HERE/'joint_selection.json'
    info=json_at(path)
    add,veto=info['addition'],info['veto']
    if add is None or veto is None or add['extra']:
        raise ValueError('No supported combination using the recorded encoder scores')
    bank={}
    receipt={'selection_sha256':checksum(path)}
    for key,folder,weight in [('addition','separate','model_sha256'),('veto','veto','models_sha256')]:
        dest=HERE/folder/'official'
        report=dest/'results.json'
        config=json_at(report)
        assert checksum(report)==info['inputs'][folder]['official']
        model=dest/'models.joblib'
        assert checksum(model)==config[weight]
        receipt[key+'_sha256']=checksum(model)
        source=add['source'] if key=='addition' else veto['source']
        trained=joblib.load(model)
        if key=='addition':
            bank[key]=trained[source,False]['new']
        else:
            profile='all_positive_prior' if veto['route']=='disagreement' else veto['route']
            bank[key]=trained[source][profile,veto['style']]
    return dict(addition=add,veto=veto),bank,receipt


def added_requests(bank, pre, size, old, threshold, cfg):
    useful=~old
    gain=np.full(len(old),-1.,dtype=np.float64)
    x=matrix(pre[useful],size[useful],threshold)
    if len(x):
        flip=bank['flip'].predict_proba(x)[:,1]
        y=bank['correct'].predict_proba(x)[:,1]
        g=y-cfg['weight']*(1.-y)
        gain[useful]=flip*np.where(pre[useful,7]>threshold,-g,g)
    calls=int(old.sum()*cfg['fraction'])
    chars=int(size[old].sum()*cfg['fraction'])
    ask=old | bounded(gain,useful,size,calls,chars)
    return ask,8*x.size+len(x)*sum(int(t.nodes['depth'].max()) for m in bank.values()
                                  for row in m._predictors for t in row)


def tree_budget(bank, mode, size):
    if bank is None:
        return 0
    if mode=='separate':
        model = (bank['negative'],bank['positive'])
    else:
        model = (bank['classifier' if mode=='classification' else mode],)
    costs = [sum(int(t.nodes['depth'].max()) for turn in m._predictors for t in turn) for m in model]
    return int(size*max(costs))


def forward(text, model, recipe_values, cfg, calibration, coef, candidate=None, bank=None):
    core,gate,regression=recipe_values
    start=time.perf_counter()
    raw,aux,cost=f.infer_student(model,text,f.pinned()[0],columns=22,return_all=True)
    stage=time.perf_counter()
    values=np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
    pred=f.estimate(regression,values)
    before,no,yes,asked=f.choose(values,pred,gate,core['gate'])
    selected=asked
    char=np.fromiter(map(len,text),dtype=np.int32)
    extra=0
    if candidate is not None:
        u=f.query_value(gate,before,no,yes,pred['w'],core['gate']['base'])
        pre=np.column_stack((values,pred['w'],no,yes,before,u))
        if 'addition' in candidate:
            selected,extra=added_requests(bank['addition'],pre,char,asked,core['threshold'],candidate['addition'])
            cutoff=clip_logit(np.asarray(np.float32(core['threshold']),dtype=np.float64))
            original=clip_logit(before.astype(np.float32).astype(np.float64))-cutoff
        else:
            x=matrix(pre,char,core['threshold'])
            raw_utility=prediction(bank,candidate['mode'],x)
            selected=reassign(asked,char,raw_utility,u,candidate['strength'],candidate['fraction'])
            extra=tree_budget(bank,candidate['mode'],len(text))+8*x.size
    known=np.zeros(len(text),bool)
    flag=np.zeros_like(known)
    first=np.zeros((len(text),2),dtype=np.int32)
    if selected.any():
        response,seconds=scan([s for s,on in zip(text,selected,strict=True) if on],True)
        known[selected]=response[:,0]>=0
        flag[selected]=response[:,0]==1
        first[selected]=response[:,1:]
    else:
        seconds=0.
    before[known]=np.where(flag[known],yes[known],no[known])
    state=dict(p=before.astype(np.float32),w=pred['w'],no=no,yes=yes,known=known,flag=flag)
    margin=np.full(len(text),np.nan)
    args=(text,values,state,core['threshold'],cfg['variant'],cfg['weights'],cfg['section'],cfg['section_head'])
    old,_,_,ops=rerank(*args,cfg['negative'],cfg['recall'],margin)
    b,adjusted,q,payment=correct_positive(text,values,state,core['threshold'],old,cfg['correction'],cfg['positive'])
    full=clip_logit(state['p'].astype(np.float64))-clip_logit(np.asarray(np.float32(core['threshold']),dtype=np.float64))
    negative=known & ~flag
    full[negative]=margin[negative]
    full[adjusted]=q-cfg['correction']['cut']
    assert np.array_equal(full>0.,b)
    positive=flag & known & (abs(full)<calibration['width'])
    if positive.any():
        x=group_features(full[positive],first[positive],calibration['style'])
        full[positive]+=calibration['strength']*(x@coef)-calibration['cut']
        extra+=2*x.size
    cut=0.
    if candidate is not None and 'veto' in candidate:
        option=candidate['veto']
        param=dict(flag=np.where(known,flag.astype(np.int8),-1),old=asked,unqueried=original,queried=full,
                   pre=pre,length=char)
        choose=veto_scope(param,option['route'])
        if option['width'] is not None:
            choose &= abs(full)<option['width']
        if choose.any():
            local={k:v[choose] for k,v in param.items()}
            z=veto_features(local,core['threshold'])
            strength=option['strength']
            adjusted=bank['veto'].decision_function(z)
            full[choose]=(1.-strength)*full[choose]+strength*adjusted-option['threshold']
            extra+=12*z.size
    elif candidate is not None:
        cut=candidate['threshold']
    tags=canonical_mat(text)
    cost['forward_seconds']+=time.perf_counter()-stage
    cost['wall_seconds']=time.perf_counter()-start
    cost['analytic_gflops']+=(ops['base_ops']+ops['extra_ops']+payment['nnz']+extra)/1e9
    cost.update(scanner_seconds=seconds,scanner_rows=int(selected.sum()),scanner_chars=int(char[selected].sum()),
                swaps=int((selected & ~asked).sum()))
    return np.column_stack((full>cut,tags)),full,selected,cost


def benchmark():
    use_joint=(HERE/'joint_selection.json').exists()
    policy,bank,sign=load_joint() if use_joint else load_router()
    calibration,coef,_=load_choice()
    published=R32/'public'
    summary=json_at(published/'results.json')
    path=published/'candidate.npz'
    assert checksum(path)==summary['results']['candidate']['predictions_sha256']
    ptext,y,_=load(PUBLIC)
    assert checksum(PUBLIC)==summary['public_sha256']
    with np.load(path) as archive:
        reference=archive['pred']
        assert np.array_equal(y,archive['truth'])
    model=f.pipe('official',f.small.tokenizer())
    recipes=recipe()
    cfg=packages()
    for active in (False,True):
        forward(ptext[:4096],model,recipes,cfg,calibration,coef,policy if active else None,bank)
    trials={False:[],True:[]}
    artifacts={}
    for k in range(5):
        for active in ((False,True) if k%2==0 else (True,False)):
            b,r,q,payment=forward(ptext,model,recipes,cfg,calibration,coef,policy if active else None,bank)
            if active in artifacts and (not np.array_equal(b,artifacts[active][0]) or not np.array_equal(q,artifacts[active][2])):
                raise ValueError('Live repeat changed predictions or requests')
            if not active:
                assert np.array_equal(reference,b)
            else:
                assert np.array_equal(reference[:,1],b[:,1])
            artifacts[active]=(b,r,q)
            quality=f.metric(y,b)
            score=f.competition_formula(quality['mean']['f1'],payment['forward_seconds'],payment['analytic_gflops'])
            trials[active].append(payment|{'score':score})
            print(k,'candidate' if active else 'reference',score,payment,flush=True)
    DEST.mkdir(exist_ok=True)
    report={}
    for active,name in ((False,'reference'),(True,'candidate')):
        b,r,q=artifacts[active]
        quality=f.metric(y,b)
        data=trials[active]
        cost={key:median(row[key] for row in data) for key in data[0] if key!='score'}
        score=f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
        file=DEST/f'{name}.npz'
        np.savez_compressed(file,pred=b,rank=r,asked=q,truth=y)
        report[name]=dict(score=score,quality=quality,cost=cost,trials=data,predictions_sha256=checksum(file))
    if use_joint:
        frac=policy['addition']['fraction']
        for key in ('scanner_rows','scanner_chars'):
            old=report['reference']['cost'][key]
            assert old<=report['candidate']['cost'][key]<=old*(1.+frac)
    else:
        assert report['reference']['cost']['scanner_rows']==report['candidate']['cost']['scanner_rows']
        assert report['reference']['cost']['scanner_chars']>=report['candidate']['cost']['scanner_chars']
    v=reference[:,0]
    truth=y[:,0].astype(bool)
    b,r,q=artifacts[True]
    metrics={k:f.comparison(truth,v,r>strict_cut(truth,v,r,k)) for k in ('fp','tp')}
    metrics['frozen']=f.comparison(truth,v,b[:,0])
    deltas=[n['score']-a['score'] for a,n in zip(trials[False],trials[True],strict=True)]
    write_json(DEST/'results.json',dict(comparison=report,quality=metrics,paired=deltas,
               source=sign,policy=policy,reference_sha256=checksum(path),public_sha256=checksum(PUBLIC),
               code_sha256=checksum(HERE/'live33.py')))
    print(metrics,deltas,flush=True)


if __name__=='__main__':
    benchmark()
