"""Benchmark the compact token ranking head on the full signed public split."""
import argparse
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from sections34 import HERE, ROOT, f, checksum, json_at, write_json, support
from audit29 import packages, canonical_mat
from compare34 import design
from attention34 import TokenHead, select_tokens, SAMPLE, HIDDEN, N_HEADS
from disagreement import clip_logit
from evaluate_rules import infer
from first_hit32 import scan
from length_eval import recipe
from live32 import load_choice
from normalize_code import batch as canonicalize
from pipeline import PUBLIC, load
from positive_runtime import apply as correct_positive, scope
from predict_factor import full_context, lexical_features, ordered_batches, batch_plan, batch, small
from rank30 import rerank
from rule_model import pooling
from subgroups32 import features as first_rule_features
from experiment33 import cut
from prefilter34 import GateFilter
from defer34 import Deferred, operations

DEST=HERE/'public'


def heads(config):
    folder=HERE/config['stage']/'official'
    source=folder/'results.json'
    track=json_at(HERE/'selection.json')
    assert checksum(source)==track['receipts'][config['stage']]['official']
    here=HERE/'attention/official'
    assert checksum(here/'receipt.json')==json_at(source)['heads_sha256']
    stored=json_at(here/'receipt.json')
    weight=here/f"{config['mode']}.pt"
    assert checksum(weight)==stored['heads'][config['mode']]['sha256']
    attention=TokenHead(stored['width']).cuda().eval()
    attention.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
    file=folder/'calibrators.joblib'
    assert checksum(file)==json_at(source)['weights_sha256']
    readout=joblib.load(file)[config['mode'],config['source']]
    return attention,readout,dict(attention_sha256=checksum(weight),calibration_sha256=checksum(file))


@torch.inference_mode()
def encode(model,tok,texts,attention,filter_=None,trim=False,lazy=None):
    starts=perf_counter()
    full=np.empty((len(texts),2),np.float32)
    aux=np.empty((len(texts),22),np.float32)
    logits=np.full((len(texts),2),np.nan,np.float32)
    pay,gflops,added=0.,0.,0.
    scored=0
    def prepare(index):
        local=texts[index:index+2048]
        return full_context(local,tok),lexical_features(local)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset,(seq,side) in ordered_batches(range(0,len(texts),2048),prepare,True):
            full[offset:offset+len(seq),1]=side[:,:4].any(axis=1)
            lexical=torch.as_tensor(side,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                ids,mask=batch(group,seq,tok.pad_token_id)
                wall=perf_counter()
                states=model.encoder.token_states(ids,mask)
                first=model.encoder.readout(states,mask,lexical[group])
                proof=model.rule_head(pooling(states,mask))
                both=torch.cat((first,proof),1).float().sigmoid()
                sampled=(lambda z,m:(z,m.bool())) if trim and states.shape[1]<=SAMPLE else select_tokens
                width_seen=min(SAMPLE,states.shape[1]) if trim else SAMPLE
                rows=offset+np.asarray(group)
                if lazy is not None and lazy.capture(rows,states,mask):
                    more=both.new_full((len(group),2),float('nan'))
                    count=0
                elif filter_ is None:
                    sample,valid=sampled(states,mask)
                    more=attention(sample,valid).float()
                    count=len(group)
                else:
                    keep=filter_(both[:,:2],both[:,2:24]).nonzero().flatten()
                    count=keep.numel()
                    more=both.new_full((len(group),2),float('nan'))
                    if count:
                        sample,valid=sampled(states[keep],mask[keep])
                        more[keep]=attention(sample,valid).float()
                    added+=len(group)*80/1e9
                scored+=count
                raw=torch.cat((both,more),1).cpu().numpy()
                full[rows,0],aux[rows],logits[rows]=raw[:,0],raw[:,2:24],raw[:,24:]
                pay+=perf_counter()-wall
                n,width=len(group),model.width
                gflops+=small.operations(n,ids.shape[1],width)+n*(4*width*22+6*width)/1e9
                added+=operations(width,count,width_seen)
    return full,aux,logits,dict(wall_seconds=perf_counter()-starts,forward_seconds=pay,extra_gflops=added,
                                analytic_gflops=gflops+added,attention_rows=scored)


def forward(texts,model,core_values,cfg,reference_rule,coef,choice,attention,calibrator,filter_=None,trim=False,deferred=False):
    started=perf_counter()
    route=f.pinned()[0]
    lazy=Deferred(attention,model.encoder.width) if deferred and attention is not None else None
    if attention is None:
        raw,aux,payment=f.infer_student(model,texts,route,columns=22,return_all=True)
        payment['extra_gflops']=0.
        payment['attention_rows']=0
    else:
        raw,aux,context,payment=encode(model.encoder,model.tok,texts,attention,filter_,trim,lazy)
        active=(raw[:,0]>route['lower']) & (raw[:,0]<route['upper'])
        t=perf_counter()
        if active.any():
            second=canonicalize([s for s,k in zip(texts,active,strict=True) if k])
            pipe=SimpleNamespace(encoder=model.encoder.encoder,tok=model.tok,mode='control')
            raw2,price=infer(pipe,second,encoding=f.tokenize)
            raw[active,0]=.5*(raw[active,0]+raw2[:,0])
            payment['forward_seconds']+=price['forward_seconds']
            payment['analytic_gflops']+=price['analytic_gflops']
            payment['forward_seconds']+=perf_counter()-t-price['wall_seconds']
        payment['routed']=int(active.sum())
    clock=perf_counter()
    core,gate,regression=core_values
    values=np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
    est=f.estimate(regression,values)
    before,no,yes,asked=f.choose(values,est,gate,core['gate'])
    active=np.zeros(len(texts),bool)
    flag=np.zeros_like(active)
    hit=np.zeros((len(texts),2),np.int32)
    if asked.any():
        run,_=scan([s for s,k in zip(texts,asked,strict=True) if k],True)
        active[asked]=run[:,0]>=0
        flag[asked]=run[:,0]==1
        hit[asked]=run[:,1:]
    score=np.where(active,np.where(flag,yes,no),before).astype(np.float32)
    state=dict(p=score,w=est['w'],no=no,yes=yes,known=active,flag=flag)
    margin=np.full(len(texts),np.nan)
    bits,_,_,ops=rerank(texts,values,state,core['threshold'],cfg['variant'],cfg['weights'],cfg['section'],
                        cfg['section_head'],cfg['negative'],cfg['recall'],margin)
    old,adjusted,pos,price=correct_positive(texts,values,state,core['threshold'],bits,cfg['correction'],cfg['positive'])
    current=scope(texts,state,core['threshold'],None)[1]
    current[adjusted]=pos-cfg['correction']['cut']
    current[active & ~flag]=margin[active & ~flag]
    assert np.array_equal(current>0.,old)
    correction=flag & active & (abs(current)<reference_rule['width'])
    paired=first_rule_features(current[correction],hit[correction],reference_rule['style'])
    current[correction]+=reference_rule['strength']*(paired@coef)-reference_rule['cut']
    original=clip_logit(before.astype(np.float32).astype(np.float64))-clip_logit(np.asarray(np.float32(core['threshold']),dtype=np.float64))
    cases=dict(unqueried=original,queried=current,flag=np.where(active,flag.astype(np.int8),-1),old=asked)
    picked=support(cases)
    if choice['width'] is not None:
        picked &= abs(current)<choice['width']
    extra_ops=0
    if lazy is not None:
        n,more=lazy.finish(picked,context)
        payment['attention_rows']+=n
        payment['analytic_gflops']+=more
        payment['extra_gflops']+=more
    if deferred:
        payment['retained_bytes']=0 if lazy is None else lazy.bytes_used
    if attention is not None and picked.any():
        if not np.isfinite(context[picked]).all():
            raise ValueError('An eligible prediction was skipped by the early filter')
        pre=np.column_stack((values,est['w'],no,yes,before,
                            f.query_value(gate,before,no,yes,est['w'],core['gate']['base'])))
        cases.update(pre=pre,length=np.fromiter(map(len,texts),dtype=np.int32))
        matrix=design({k:v[picked] for k,v in cases.items()},core['threshold'],context[picked])
        w=choice['strength']
        current[picked]=(1.-w)*current[picked]+w*calibrator.decision_function(matrix)-choice['cut']
        extra_ops=4*matrix.size
    bits=np.column_stack((current>0.,canonical_mat(texts)))
    payment['forward_seconds']+=perf_counter()-clock
    payment['wall_seconds']=perf_counter()-started
    payment['analytic_gflops']+=(ops['base_ops']+ops['extra_ops']+price['nnz']+2*paired.size+extra_ops)/1e9
    return bits,picked,current,payment|{'new_rows':int(picked.sum()) if attention is not None else 0}


def benchmark(gated=False,pairs=5,trim=False,deferred=False):
    plan=HERE/'selection.json'
    choice=json_at(plan)['selected']
    if choice is None or choice['stage']!='attention_comparison':
        raise ValueError('No token ranking arm passed selection')
    attention,calibrator,receipt=heads(choice)
    control=ROOT/'round32/public'
    proof=json_at(control/'results.json')
    src=control/'candidate.npz'
    assert checksum(src)==proof['results']['candidate']['predictions_sha256'] and checksum(PUBLIC)==proof['public_sha256']
    with np.load(src) as x:
        old=x['pred']
        y=x['truth']
    texts,truth,_=load(PUBLIC)
    assert np.array_equal(y,truth)
    model=f.pipe('official',f.small.tokenizer())
    ref,coef,_=load_choice()
    core=recipe()
    filter_=GateFilter(core,f.pinned()[0]) if gated else None
    common=(model,core,packages(),ref,coef,choice)
    destination=HERE/'public_deferred' if deferred else (HERE/'public_trim' if trim else (HERE/'public_gated' if gated else DEST))
    expected=None
    if gated or trim or deferred:
        previous=DEST/'candidate.npz'
        assert checksum(previous)==json_at(DEST/'results.json')['comparison']['candidate']['predictions_sha256']
        with np.load(previous) as z:
            expected=z['pred']
    for head in (None,attention):
        forward(texts[:4096],*common,head,calibrator,filter_,trim,deferred)
    trials={'reference':[],'candidate':[]}
    finals={}
    for turn in range(pairs):
        for flag in ((False,True) if turn%2==0 else (True,False)):
            name='candidate' if flag else 'reference'
            value,scope_,rank,cost=forward(texts,*common,attention if flag else None,calibrator,filter_,trim,deferred)
            if name in finals:
                assert np.array_equal(value,finals[name][0])
            finals[name]=(value,scope_,rank)
            if flag:
                if expected is not None:
                    assert np.array_equal(expected,value)
                assert np.array_equal(value[~scope_],old[~scope_]) and np.array_equal(value[:,1],old[:,1])
            else:
                assert np.array_equal(old,value)
            score=f.competition_formula(f.metric(y,value)['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            trials[name].append(cost|{'score':score})
            print(turn,name,score,cost,flush=True)
    destination.mkdir(exist_ok=True)
    metrics={}
    for role,series in trials.items():
        bits,mask,rank=finals[role]
        cost={k:median(t[k] for t in series) for k in series[0] if k!='score'}
        measure=f.metric(y,bits)
        archive=destination/f'{role}.npz'
        np.savez_compressed(archive,pred=bits,scope=mask,rank=rank,truth=y)
        metrics[role]=dict(score=f.competition_formula(measure['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops']),
                           quality=measure,cost=cost,trials=series,predictions_sha256=checksum(archive))
    bits,mask,rank=finals['candidate']
    label=y[:,0].astype(bool)
    effects={'frozen':f.comparison(label,old[:,0],bits[:,0])}
    for k in ('fp','tp'):
        same=old[:,0].copy()
        threshold=cut(label[mask],same[mask],rank[mask],k)
        same[mask]=rank[mask]>threshold
        effects[k]=f.comparison(label,old[:,0],same)
    pair=[b['score']-a['score'] for a,b in zip(trials['reference'],trials['candidate'],strict=True)]
    write_json(destination/'results.json',dict(comparison=metrics,paired=pair,quality=effects,chosen=choice,
               receipt=receipt|dict(selection_sha256=checksum(plan),reference_sha256=checksum(src),
                                    public_sha256=checksum(PUBLIC),code_sha256=checksum(HERE/'live34.py'),
                                    prefilter_sha256=checksum(HERE/'prefilter34.py') if gated else None,
                                    deferred_sha256=checksum(HERE/'defer34.py') if deferred else None)))
    print(effects,pair,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--gated',action='store_true')
    cli.add_argument('--pairs',type=int,default=5)
    cli.add_argument('--trim-padding',action='store_true')
    cli.add_argument('--deferred',action='store_true')
    args=cli.parse_args()
    benchmark(args.gated,args.pairs,args.trim_padding,args.deferred)
