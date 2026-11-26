"""Check distilled readouts at the signed budget on displaced projects."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median

import joblib
import numpy as np
import torch

from distill import HERE, OUT, PARTS, STYLES, f, checksum, json_at, write_json
from reference29 import reference, match, strict_cut
from student_runtime import head, predict
from disagreement import clip_logit, trained_logistic
from train_delta import MODES as DELTAS

RESULTS=HERE/'comparisons'
WIDTHS=(.40,.9,1.8,None)


def load_heads(rotation,modes):
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    classifier={}
    records={}
    for name in modes:
        folder=OUT/rotation
        source='delta' if name in DELTAS else ('tuned' if name in STYLES[3:] else 'frozen')
        record=json_at(folder/f'{source}.json')
        path=folder/f'{name}.pt'
        if checksum(path)!=record['mode'][name]['sha256']:
            raise ValueError('Stored residual head changed')
        h=head(model.width).cuda().eval()
        h.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        classifier[name]=h
        records[name]=record['mode'][name]['sha256']
    return model,tok,classifier,records


def predictions(model,tok,ctx,heads):
    p,aux,extra,_=predict(model,tok,ctx.texts,tuple(heads.values()))
    old=ctx.frame['reference']
    if not np.allclose(aux[:,20],old[:,2],atol=.004,rtol=0.) or not np.allclose(aux[:,21],old[:,3],atol=.004,rtol=0.):
        raise ValueError('The original conditional head changed')
    anchor=clip_logit(aux[:,20].astype(np.float64))
    return {k:(extra[:,i].astype(np.float64),extra[:,i]-anchor) for i,k in enumerate(heads)}


def cost_of(model,tok,heads,texts):
    pair=next(iter(heads.values()))
    timings={False:[],True:[]}
    capture={False:(),True:(pair,)}
    for toggle in capture:
        predict(model,tok,texts,capture[toggle])
    for i in range(5):
        for toggle in capture if i%2==0 else reversed(capture):
            timings[toggle].append(predict(model,tok,texts,capture[toggle])[-1])
    delta=[b['forward_seconds']-a['forward_seconds'] for a,b in zip(timings[False],timings[True],strict=True)]
    return dict(seconds=max(0.,median(delta)),trial_delta=delta,
                extra_gflops=median(r['extra_gflops'] for r in timings[True]),
                turns={str(k):v for k,v in timings.items()})


def compare(rotation,stage):
    d,t,fit,cal,review,base,tbase,p,q=reference(rotation)
    modes=DELTAS if stage=='delta' else (STYLES if stage=='all' else STYLES[:3])
    model,tok,heads,records=load_heads(rotation,modes)
    a=predictions(model,tok,d,heads)
    b=a if d is t else predictions(model,tok,t,heads)
    dev,tst=d.frame['y'],t.frame['y']
    held=np.ones(len(tst),dtype=bool) if review is None else review
    budget_score={}
    if review is not None:
        scope=[s for s,y in zip(d.texts,cal,strict=True) if y]
        budget_score=cost_of(model,tok,heads,scope)
    out=RESULTS/stage/rotation
    out.mkdir(parents=True,exist_ok=True)
    predictions_out=dict(ids=t.frame['ids'][held],base=tbase[held],truth=tst[held],eligible=t.active[held])
    settings={}
    readouts={}
    for k in modes:
        aa,shift=a[k]
        bb,tshift=b[k]
        core=np.column_stack((p,aa,shift))
        test=np.column_stack((q,bb,tshift))
        use=fit&d.active
        h=trained_logistic(core[use],dev[use],c=.7)
        readouts[k]=None if k in DELTAS else h
        rd=p+shift if k in DELTAS else h.decision_function(core)
        rt=q+tshift if k in DELTAS else h.decision_function(test)
        for width,m in product(WIDTHS,(.35,.7)):
            ok=d.active.copy()
            on=t.active.copy()
            if width is not None:
                ok&=abs(p)<width
                on&=abs(q)<width
            present=on&held
            y=dev[ok&cal]
            if not len(y):
                continue
            ds=((1.-m)*p+m*rd)[ok]
            ts=((1.-m)*q+m*rt)[on]
            name=f'{k}_{width:g}_{m:g}' if width is not None else f'{k}_all_{m:g}'
            row=dict(mode=k,width=width,mix=m,cal_calls=len(y),test_calls=int((on&held).sum()),cuts={})
            if review is None:
                eq,per=match(tst,t.frame['project'],tbase,on,ts)
                row.update(matched=f.comparison(tst,tbase,eq),per=per)
                predictions_out[name+'_matched']=eq
            for arm in ('fp','tp','fixed'):
                cut=strict_cut(y,base[ok&cal],ds[cal[ok]],arm)
                dc=base.copy()
                tc=tbase.copy()
                dc[ok],tc[on]=ds>cut,ts>cut
                score=f.comparison(dev[cal],base[cal],dc[cal])
                assessment=dict(cut=cut,cal=score,frozen=f.comparison(tst[held],tbase[held],tc[held]))
                if budget_score:
                    debit=.2*f.PUBLIC_N/cal.sum()*(budget_score['seconds']/1000+budget_score['extra_gflops']/3e6)
                    assessment['proxy']=.3*(score['f1_after']-score['f1_before'])-debit
                row['cuts'][arm]=assessment
                predictions_out[name+'_'+arm]=tc[held]
            predictions_out[name+'_scope']=present[held]
            predictions_out[name+'_rank']=ts[held[on]]
            settings[name]=row
    path=out/'predictions.npz'
    np.savez_compressed(path,**predictions_out)
    weight=out/'readouts.joblib'
    joblib.dump(readouts,weight,compress=3)
    write_json(out/'results.json',dict(options=settings,heads=records,timing=budget_score,
               predictions_sha256=checksum(path),readout_sha256=checksum(weight),code_sha256=checksum(Path(__file__))))
    print(rotation,stage,len(settings),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    p.add_argument('stage',choices=('frozen','all','delta'))
    a=p.parse_args()
    compare(a.rotation,a.stage)
