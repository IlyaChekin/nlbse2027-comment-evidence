"""Matched 64-token window comparison with the frozen submission pipeline."""
from contextlib import nullcontext
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'round55'))
from span55 import FIT, WIDTH, batch, checksum, f, full_context, json_at, lexical_features, load, pooling, readout, small, write_json
from live40 import anchor, pipeline
from live35 import forward
from source41 import parts
from pipeline import PUBLIC, load as public_load

SIZE=64
CONTENT=SIZE-2
LOWER=-3.8
MODES=('base','center','located')


def start_for(center,n):
    return max(1,min(center-CONTENT//2,n-CONTENT-1))


def features(score,local):
    return np.column_stack((score, local))


def session():
    _,runtime,baseline,args=pipeline()
    original=args[0].encoder
    tok=args[0].tok
    proof=FIT/'official/fit.json'
    head=readout(proof.parent,'span',original,json_at(proof))
    record=runtime.record

    def run(texts,mode,calibrator=None):
        total=len(texts)
        lengths=np.zeros(total,np.int32)
        peak=np.full(total,-1,np.int32)
        debt=0
        used=np.empty(0,np.int64)
        data=np.empty((0,3),np.float64)
        starts=np.empty(0,np.int32)
        window_seconds=0.
        applied=0
        num_fallback=0

        @torch.inference_mode()
        def find(ids,state,mask,base,aux):
            nonlocal debt
            with torch.autocast('cuda',dtype=torch.float16):
                prior=torch.stack((base[:,0],aux[:,0]),1)
                _,q=head(state,mask,prior)
                t=torch.arange(mask.size(1),device='cuda')[None,:]
                content=(t>0)&(t<mask.sum(1)[:,None]-1)
                peak[ids]=q.masked_fill(~content,-1.e4).argmax(1).cpu().numpy()
            debt+=mask.numel()*(2*state.shape[-1]*WIDTH+12*WIDTH+20)

        def capture(ids,state,mask,side,base,aux,pool):
            nonlocal num_fallback
            lengths[ids]=mask.sum(1).cpu().numpy()
            before=len(runtime.records)
            record(ids,state,mask,side,base,aux,pool)
            if mode=='located' and len(runtime.records)==before:
                find(ids,state,mask,base,aux)
                num_fallback+=len(ids)

        @torch.inference_mode()
        def stage(rows, rank, first, aux, ctx):
            nonlocal used,data,starts,debt,window_seconds,applied
            signed,scope,pay=baseline(rows,rank,first,aux,ctx)
            if mode=='base':
                return signed,scope,pay
            used=np.flatnonzero(rows['old']&(rows['flag']==0)&(signed<=0.)&(signed>LOWER)&(lengths>SIZE))
            if not len(used):
                return signed,scope,pay+debt
            t=perf_counter()
            if mode=='located':
                local=np.zeros(total,bool)
                local[used]=True
                for ids, pack in runtime.records:
                    ix=np.flatnonzero(local[ids])
                    if len(ix):
                        state,mask,_,main,rule,_=pack
                        find(ids[ix],state[ix],mask[ix],main[ix],rule[ix])
                assert (peak[used]>=1).all()
            sub=[texts[i] for i in used]
            entire=full_context(sub,tok)
            assert np.array_equal(lengths[used],[len(s) for s in entire])
            points=peak[used] if mode=='located' else (lengths[used]-1)//2
            starts=np.array([start_for(int(p),len(s)) for p,s in zip(points,entire,strict=True)],np.int32)
            windows=[[s[0],*s[u:u+CONTENT],s[-1]] for s,u in zip(entire,starts,strict=True)]
            assert all(len(s)==SIZE for s in windows)
            excerpt=tok.batch_decode([s[1:-1] for s in windows],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            side=torch.as_tensor(lexical_features(excerpt),device='cuda')
            predicted=np.empty((len(windows),2),np.float32)
            with torch.autocast('cuda',dtype=torch.float16):
                for offset in range(0,len(windows),128):
                    ids=list(range(offset,min(offset+128,len(windows))))
                    x,mask=batch(ids,windows,tok.pad_token_id)
                    h=original.encoder.token_states(x,mask)
                    v=original.encoder.readout(h,mask,side[ids])[:,0]
                    w=original.rule_head(pooling(h,mask))[:,0]
                    predicted[ids]=torch.stack((v,w),1).float().cpu().numpy()
                    debt+=1e9*small.operations(len(ids),SIZE,original.width)+len(ids)*original.width*100
            data=features(signed[used],predicted)
            result=signed.copy()
            if calibrator is not None:
                margin=calibrator.decision_function(data)
                accepted=margin>0.
                result[used[accepted]]=margin[accepted]
                applied=int(accepted.sum())
            window_seconds=perf_counter()-t
            return result,scope,pay+debt

        with patch.object(runtime,'record',capture) if mode!='base' else nullcontext():
            answer,_,_,cost=forward(texts,*args,stage)
        t=perf_counter()
        runtime.finish(cost)
        elapsed=perf_counter()-t
        cost['forward_seconds']+=elapsed
        cost['wall_seconds']+=elapsed
        cost.update(windows=len(used),tokens=len(used)*SIZE,window_seconds=window_seconds,
                    flipped=applied,eager_localizer_rows=num_fallback)
        return answer,used,data,starts,cost

    return run,dict(source_fit_sha256=checksum(proof),span_head_sha256=checksum(proof.parent/'span.pt'))


def matched(a,b):
    assert np.array_equal(a[1],b[1])
    assert a[-1]['windows']==b[-1]['windows'] and a[-1]['tokens']==b[-1]['tokens']
    assert a[-1]['scanner_calls']==b[-1]['scanner_calls']
    assert np.array_equal(a[0][:,1],b[0][:,1])


def main():
    result=HERE/'runs'
    result.mkdir(exist_ok=True)
    run,w=session()
    source,_,matrix=load('official')
    text,label,*_=f.inputs()
    cal,held=parts(source)
    sample=[text[i] for i in cal['ids']]
    left,right=(run(sample,x) for x in ('center','located'))
    matched(left,right)
    assert np.array_equal(left[0],right[0])
    truth=cal['y'][left[1]]
    assert np.unique(truth).size==2
    reg=LogisticRegression(C=1.,solver='liblinear',random_state=56).fit(np.r_[left[2],right[2]],np.tile(truth,2))
    config=dict(window_tokens=SIZE,content_tokens=CONTENT,minimum_margin=LOWER,scope='observed scanner negative and baseline margin in (-3.8,0]',
                calibration_rows=len(truth),calibration_positives=int(truth.sum()),coefficients=reg.coef_.tolist(),
                intercept=reg.intercept_.tolist(),threshold=0.,matrix_sha256=matrix,**w)
    write_json(result/'policy.json',config)
    cache={}
    phases={}
    archive,previous,public_truth,expected=anchor()
    public_x,public_y,_=public_load(PUBLIC)
    assert np.array_equal(public_truth,public_y)
    for phase,x,y,count in [('review',[text[i] for i in held['ids']],label[held['ids']],3),('public',public_x,public_y,5)]:
        for arm in MODES:
            run(x[:1024],arm,reg)
        costs={k:[] for k in MODES}
        remembered={}
        for i in range(count):
            order=(MODES[i%3:]+MODES[:i%3]) if i%2==0 else tuple(reversed(MODES[i%3:]+MODES[:i%3]))
            for arm in order:
                out=run(x,arm,reg)
                if arm in remembered:
                    assert np.array_equal(out[0],remembered[arm][0]) and np.array_equal(out[1],remembered[arm][1])
                remembered[arm]=out
                costs[arm].append(out[-1])
                print(phase,i,arm,out[-1]['windows'],round(out[-1]['window_seconds'],4),round(out[-1]['wall_seconds'],4),flush=True)
        matched(remembered['center'],remembered['located'])
        base=remembered['base'][0]
        if phase=='public':
            assert np.array_equal(base,expected)
        selected=remembered['center'][1]
        saved={phase+'_truth':y,phase+'_rows':selected}
        table={}
        for arm,record in remembered.items():
            p=record[0]
            assert np.array_equal(p[:,1],base[:,1])
            other=np.ones(len(p),bool)
            other[selected]=False
            assert np.array_equal(p[other],base[other])
            t={key:median(q[key] for q in costs[arm]) for key in costs[arm][0]}
            stat=f.metric(y,p)
            table[arm]=dict(metric=stat,change=f.comparison(y[:,0],base[:,0],p[:,0]),median=t,trials=costs[arm],
                score=f.competition_formula(stat['mean']['f1'],t['forward_seconds'],t['analytic_gflops']))
            saved[phase+'_'+arm]=p
            saved[phase+'_starts_'+arm]=record[3]
            saved[phase+'_features_'+arm]=record[2]
        phases[phase]=dict(arms=table,same_windows=int((remembered['center'][3]==remembered['located'][3]).sum()),
                           rows=len(y),paired_wall_difference=[b['wall_seconds']-a['wall_seconds']
                                   for a,b in zip(costs['center'],costs['located'],strict=True)])
        cache.update(saved)
    out=result/'predictions.npz'
    np.savez_compressed(out,**cache)
    write_json(result/'results.json',dict(phases=phases,policy_sha256=checksum(result/'policy.json'),
               predictions_sha256=checksum(out),baseline_archive_sha256=checksum(archive),
               baseline_predictions_sha256=checksum(previous),data_sha256=checksum(PUBLIC),
               code_sha256=checksum(Path(__file__))))


if __name__=='__main__':
    main()
