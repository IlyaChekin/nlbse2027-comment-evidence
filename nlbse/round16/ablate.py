"""Ablate the local readout on the identical trained trunk."""

import argparse
import csv

import numpy as np
from scipy.special import expit, logit

from experiment import HERE,PROJECT,ROOT,comparison,infer,inputs,json_at,loaded,open_model,small,strict_budget
from evaluate import counts, exact_budget, raw
from pipeline import write_json


OUT=HERE/"ablation"
GAINS=(0.,.25,.5,1.,1.5,2.)


def measure(pipe,rows,raw_p,scope):
    pipe.encoder.evidence_gain=0.
    pipe.mode="control"
    baseline,_=infer(pipe,rows)
    before=baseline[:,0]
    after=raw_p[:,0]
    if not np.array_equal(baseline[:,1],raw_p[:,1]):
        raise ValueError(f"MAT changed during ablation in {scope}")
    return before, after


def mix(previous, updated, gain):
    if gain==0.:
        return previous
    if gain==1.:
        return updated
    p,q=logit(previous.clip(1e-6,1-1e-6)),logit(updated.clip(1e-6,1-1e-6))
    return expit(p+gain*(q-p))


def study(seed):
    output=OUT/str(seed)
    output.mkdir(parents=True,exist_ok=True)
    texts,y,projects,_,parts=inputs()
    dev,test=parts["dev"],parts["test"]
    tok=small.tokenizer()
    original_dev=raw("dev")["compact"]
    original_test=raw("test")["compact"]
    threshold=json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    budget_dev=counts(y[dev,0],original_dev>threshold)["fp"]
    budget_test=counts(y[test,0],original_test>threshold)["fp"]
    dev_text=[texts[i] for i in dev]
    test_text=[texts[i] for i in test]
    test_ref,_=exact_budget(original_test,y[test,0],budget_test)
    trial=PROJECT/str(seed)
    names=[k for k in json_at(trial/"policy.json")["dev"] if k!="control"]
    records={}
    for name in names:
        binder=open_model("projects",trial,name,tok)
        known=loaded(trial,"dev",name,dev,y[dev])
        off,on=measure(binder,dev_text,known,"dev")
        base,_=exact_budget(off,y[dev,0],budget_dev)
        policy={}
        for gain in GAINS:
            p=mix(off,on,gain)
            evaluated,_=exact_budget(p,y[dev,0],budget_dev)
            _,cut=strict_budget(p,y[dev,0],budget_dev)
            policy[str(gain)]=dict(threshold=cut,development=comparison(y[dev,0],base,evaluated))
        best=max(GAINS,key=lambda g:(policy[str(g)]["development"]["delta_tp"],g==1.))
        write_json(output/f"{name}_policy.json",dict(gain=best,dev_fp=budget_dev,grid=policy))
        off,on=measure(binder,test_text,loaded(trial,"test",name,test,y[test]),"test")
        base,_=exact_budget(off,y[test,0],budget_test)
        work={}
        for gain in (0.,1.,best) if best not in (0.,1.) else (0.,1.):
            p=mix(off,on,gain)
            same,_=exact_budget(p,y[test,0],budget_test)
            fixed=p>policy[str(gain)]["threshold"]
            work[str(gain)]=dict(matched=comparison(y[test,0],base,same),against_reference=comparison(y[test,0],test_ref,same),
                                 frozen=comparison(y[test,0],original_test>threshold,fixed))
            with (output/f"{name}_{gain:g}.csv").open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(("row","project","label","off_score","on_score","candidate_score","off_equal","candidate_equal","candidate_frozen"))
                w.writerows(zip(test,projects[test],y[test,0],off,on,p,base.astype(int),same.astype(int),fixed.astype(int),strict=True))
        log_difference=np.abs(logit(on.clip(1e-6,1.-1e-6))-logit(off.clip(1e-6,1.-1e-6)))
        records[name]=dict(selected=best,test=work,logit_change_quantiles=np.quantile(log_difference,[.5,.9,.99]).tolist())
        print(seed,name,best,records[name],flush=True)
    write_json(output/"results.json",records)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True)
    study(p.parse_args().seed)
