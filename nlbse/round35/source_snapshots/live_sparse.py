"""Profile the selected cheap correction against the frozen Round 34 release."""
import argparse
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from teacher35 import HERE, ROOT, f, checksum, json_at, write_json
from eval35 import design, cut
from sections34 import support
from live34 import heads, packages, canonical_mat, recipe, load_choice, scan, scope, correct_positive, rerank
from live34 import canonicalize, clip_logit, infer, full_context, lexical_features, ordered_batches, batch_plan, batch, small, pooling
from subgroups32 import features as first_rule_features
from sparse35 import vectorizer, view
from pipeline import PUBLIC, load

R34=ROOT/'round34'
OUT=HERE/'public'


def weights(selected):
    folder=HERE/selected['stage']/'official'
    p=folder/'results.json'
    entry=json_at(HERE/'selection.json')
    assert checksum(p)==entry['result_sha256'][selected['stage']]['official']
    proof=json_at(p)
    file=folder/'readouts.joblib'
    assert proof['weights_sha256']==checksum(file)
    trained=joblib.load(file)[selected['source'],selected['mode']]
    if selected['stage']=='sparse':
        assert isinstance(trained,dict)
        return trained,None,dict(readout_sha256=checksum(file))
    stored=folder/f"{selected['mode']}.pt"
    assert proof['heads'][selected['mode']]['sha256']==checksum(stored)
    state=torch.load(stored,map_location='cpu',weights_only=True)
    layer=torch.nn.Linear(state['weight'].shape[1],2).cuda().eval()
    layer.load_state_dict(state)
    return trained,layer,dict(readout_sha256=checksum(file),projection_sha256=checksum(stored))


@torch.inference_mode()
def encode(model,tok,texts,first,second):
    start=perf_counter()
    q=np.empty((len(texts),2),np.float32)
    a=np.empty((len(texts),22),np.float32)
    feature=np.empty((len(texts),2 if second is None else 4),np.float32)
    gflops,paid=0.,0.
    more=0.
    def prepare(i):
        code=texts[i:i+2048]
        return full_context(code,tok),lexical_features(code)
    with torch.autocast('cuda',dtype=torch.float16):
        for i,(seq,side) in ordered_batches(range(0,len(texts),2048),prepare,True):
            q[i:i+len(seq),1]=side[:,:4].any(axis=1)
            value=torch.as_tensor(side,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                ids,mask=batch(group,seq,tok.pad_token_id)
                t=perf_counter()
                hidden=model.encoder.token_states(ids,mask)
                base=model.encoder.readout(hidden,mask,value[group])
                pool=model.rule_head[0](pooling(hidden,mask))
                aux=model.rule_head[1](pool)
                with torch.autocast('cuda',enabled=False):
                    state=pool.half().float()
                    one=first(state)
                    learned=one if second is None else torch.cat((one,second(state)),1)
                out=torch.cat((torch.cat((base,aux),1).float().sigmoid(),learned),1).cpu().numpy()
                select=i+np.asarray(group)
                q[select,0],a[select],feature[select]=out[:,0],out[:,2:24],out[:,24:]
                paid+=perf_counter()-t
                n=len(group)
                width=model.width
                gflops+=small.operations(n,ids.shape[1],width)+n*(4*width*22+6*width)/1e9
                gflops+=n*width*8/1e9
                if second is not None:
                    more+=n*width*8/1e9
    return q,a,feature,dict(forward_seconds=paid,wall_seconds=perf_counter()-start,analytic_gflops=gflops+more,
                             projection_gflops=more)


def forward(text,model,core_values,config,rule_cfg,rule_coef,old_cfg,original_head,original_cal,choice=None,extra_head=None,
            calibrated=None):
    t0=perf_counter()
    core,gate,regression=core_values
    raw,aux,context,cost=encode(model.encoder,model.tok,text,original_head,extra_head)
    route=f.pinned()[0]
    second=(raw[:,0]>route['lower']) & (raw[:,0]<route['upper'])
    if second.any():
        t=perf_counter()
        masked=canonicalize([s for s,k in zip(text,second,strict=True) if k])
        pipe=SimpleNamespace(encoder=model.encoder.encoder,tok=model.tok,mode='control')
        additional,payment=infer(pipe,masked,encoding=f.tokenize)
        raw[second,0]=.5*(raw[second,0]+additional[:,0])
        cost['forward_seconds']+=payment['forward_seconds']+perf_counter()-t-payment['wall_seconds']
        cost['analytic_gflops']+=payment['analytic_gflops']
    start=perf_counter()
    values=np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
    est=f.estimate(regression,values)
    before,no,yes,asked=f.choose(values,est,gate,core['gate'])
    known=np.zeros(len(text),bool)
    flag=np.zeros_like(known)
    first=np.zeros((len(text),2),np.int32)
    if asked.any():
        found,_=scan([s for s,k in zip(text,asked,strict=True) if k],True)
        known[asked]=found[:,0]>=0
        flag[asked]=found[:,0]==1
        first[asked]=found[:,1:]
    p=np.where(known,np.where(flag,yes,no),before).astype(np.float32)
    state=dict(p=p,w=est['w'],no=no,yes=yes,known=known,flag=flag)
    margin=np.full(len(text),np.nan)
    bits,_,_,ops=rerank(text,values,state,core['threshold'],config['variant'],config['weights'],
                        config['section'],config['section_head'],config['negative'],config['recall'],margin)
    reference,corrected,rank,value=correct_positive(text,values,state,core['threshold'],bits,
                             config['correction'],config['positive'])
    current=scope(text,state,core['threshold'],None)[1]
    current[corrected]=rank-config['correction']['cut']
    negative=known & ~flag
    current[negative]=margin[negative]
    assert np.array_equal(current>0.,reference)
    good=known & flag & (abs(current)<rule_cfg['width'])
    x=first_rule_features(current[good],first[good],rule_cfg['style'])
    current[good]+=rule_cfg['strength']*(x@rule_coef)-rule_cfg['cut']
    base=clip_logit(before.astype(np.float32).astype(np.float64))-clip_logit(np.asarray(np.float32(core['threshold']),dtype=np.float64))
    live=dict(queried=current.copy(),unqueried=base,flag=np.where(known,flag.astype(np.int8),-1),old=asked,
              pre=np.column_stack((values,est['w'],no,yes,before,f.query_value(gate,before,no,yes,est['w'],core['gate']['base']))),
              length=np.fromiter(map(len,text),np.int32))
    chosen=support(live)
    guard=chosen & (abs(current)<old_cfg['width'])
    if guard.any():
        head=design({k:z[guard] for k,z in live.items()},core['threshold'],context[guard,:2])
        w=old_cfg['strength']
        current[guard]=(1.-w)*current[guard]+w*original_cal.decision_function(head)-old_cfg['cut']
    count=0
    extra_ops=0
    mask=np.zeros(len(text),bool)
    if choice is not None:
        mask=chosen.copy()
        if choice['width'] is not None:
            mask &= abs(current)<choice['width']
        if mask.any():
            picked={k:z[mask] for k,z in live.items()}
            if choice['stage']=='projection':
                features=design(picked,core['threshold'],context[mask,2:])
                raw=calibrated.decision_function(features)
            else:
                mode=choice['mode'].rsplit('_',1)[0]
                code=[view(mode,s) for s,k in zip(text,mask,strict=True) if k]
                mx=vectorizer(mode).transform(code)
                features=np.column_stack((design(picked,core['threshold']),calibrated['first'].decision_function(mx)))
                raw=calibrated['readout'].decision_function(features)
                extra_ops=2*mx.nnz
            weight=choice['strength']
            current[mask]=(1.-weight)*current[mask]+weight*raw-choice['cut']
            extra_ops+=2*features.size
            count=int(mask.sum())
    answer=np.column_stack((current>0.,canonical_mat(text)))
    cost['wall_seconds']=perf_counter()-t0
    cost['forward_seconds']+=perf_counter()-start
    cost['analytic_gflops']+=(ops['base_ops']+ops['extra_ops']+value['nnz']+2*x.size+extra_ops)/1e9
    return answer,current,mask,cost|{'new_rows':count,'routed':int(second.sum()),'scanner_calls':int(asked.sum())}


def main(sparse_only=False):
    plan=HERE/'selection.json'
    selection=json_at(plan)
    choice=selection['by_stage']['sparse'] if sparse_only else selection['selected']
    output=HERE/'public_sparse' if sparse_only else OUT
    if choice is None:
        raise ValueError('No supported cheap arm')
    candidate,added,stamp=weights(choice)
    old_path=R34/'selection_approx.json'
    old_cfg=json_at(old_path)['selected']
    fold,old_cal,_=heads(old_cfg,old_path)
    base=R34/'public_approx/candidate.npz'
    assert checksum(base)==json_at(R34/'public_approx/results.json')['comparison']['candidate']['predictions_sha256']
    with np.load(base) as z:
        expected,y=z['pred'],z['truth']
    text,y_true,_=load(PUBLIC)
    assert np.array_equal(y,y_true)
    ref32,coef,_=load_choice()
    model=f.pipe('official',f.small.tokenizer())
    common=(model,recipe(),packages(),ref32,coef,old_cfg,fold,old_cal)
    arms={False:[],True:[]}
    for k in (False,True):
        forward(text[:4096],*common,choice if k else None,added if k else None,candidate)
    saved={}
    for i in range(5):
        for k in ((False,True) if i%2==0 else (True,False)):
            forecast,margin,scope_,t=forward(text,*common,choice if k else None,added if k else None,candidate)
            if k in saved:
                assert np.array_equal(saved[k][0],forecast)
            saved[k]=forecast,margin,scope_
            if not k:
                assert np.array_equal(forecast,expected)
            else:
                assert np.array_equal(forecast[~scope_],expected[~scope_]) and np.array_equal(forecast[:,1],expected[:,1])
            score=f.competition_formula(f.metric(y,forecast)['mean']['f1'],t['forward_seconds'],t['analytic_gflops'])
            arms[k].append(t|{'score':score})
            print(i,'candidate' if k else 'reference',score,t,flush=True)
    report={}
    output.mkdir(exist_ok=True)
    for k,name in ((False,'reference'),(True,'candidate')):
        forecast,margin,scope_=saved[k]
        m=f.metric(y,forecast)
        times={field:median(u[field] for u in arms[k]) for field in arms[k][0] if field!='score'}
        file=output/f'{name}.npz'
        np.savez_compressed(file,pred=forecast,truth=y,margin=margin,scope=scope_)
        report[name]=dict(score=f.competition_formula(m['mean']['f1'],times['forward_seconds'],times['analytic_gflops']),
                          quality=m,cost=times,trials=arms[k],predictions_sha256=checksum(file))
    gold=y[:,0].astype(bool)
    before=expected[:,0]
    q,m,where=saved[True]
    changes=dict(frozen=f.comparison(gold,before,q[:,0]))
    for kind in ('tp','fp'):
        z=before.copy()
        bound=cut(gold[where],before[where],m[where],kind)
        z[where]=m[where]>bound
        changes[kind]=f.comparison(gold,before,z)
    paired=[b['score']-a['score'] for a,b in zip(arms[False],arms[True],strict=True)]
    write_json(output/'results.json',dict(comparison=report,quality=changes,paired=paired,selected=choice,
               receipt=stamp|{'selection_sha256':checksum(plan),'code_sha256':checksum(HERE/'live35.py'),
                              'public_sha256':checksum(PUBLIC),'reference_sha256':checksum(base)}))
    print(changes,paired,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sparse',action='store_true')
    main(cli.parse_args().sparse)
