"""Reallocate the finite short-function false-positive budget using a two-sided readout."""
import csv
import json

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bindings
from evaluate import OUT as STUDY, cohort, counts, exact_budget, inputs, raw, summarize_row
from rotate import HERE, checksum
from pipeline import write_json

OUT=HERE/"redistribute"
CANDIDATES=("logits","token","bindings","grammar_bindings","interface","interface_bindings","all")


def design(method,p,features,length):
    names=() if method=="logits" else ("token","grammar_bindings","interface_bindings") if method=="all" else (method,)
    columns=np.column_stack([p,*[features[k] for k in names]])
    return np.column_stack((logit(columns.clip(1e-6,1-1e-6)),np.log1p(length)))


def model_for():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.17,max_iter=400))


def blend(a,b,mix):
    return expit((1-mix)*logit(a.clip(1e-6,1-1e-6))+mix*logit(b.clip(1e-6,1-1e-6)))


def samples(role):
    text,y,project,paths,parts=inputs()
    ids=parts[role]
    p=raw(role)
    short,f=bindings.posterior(role,ids,y)
    length=np.fromiter((len(text[i]) for i in ids),dtype=np.int32,count=len(ids))
    return y[ids,0],p,short,f,length


def tune():
    y0,p0,s0,f0,chars0=samples("probe")
    y1,p1,s1,f1,chars1=samples("dev")
    nums=p1["compact"][s1]
    target=cohort("joint_low",s1,p1)[s1]
    base=float(json.loads((STUDY/"policy.json").read_text())["compact_threshold"])
    old=nums>base
    budget=counts(y1[s1],old)["fp"]
    rank,_=exact_budget(nums,y1[s1],budget)
    configs={}
    weights={}
    for k in CANDIDATES:
        features=design(k,p0["compact"],f0,chars0)
        model=model_for().fit(features[s0],y0[s0])
        q=model.predict_proba(design(k,p1["compact"],f1,chars1)[s1])[:,1]
        candidates=[]
        for mix in (0.,.12,.25,.4,.55,.7,.85,1.):
            p=blend(nums,q,mix)
            selected,t=exact_budget(p,y1[s1],budget)
            values=summarize_row(y1[s1],rank,selected,target)
            candidates.append(dict(mix=mix,threshold=t,development=values))
        rule=max(candidates,key=lambda x:(x["development"]["delta_tp"],x["development"]["repaired_in_target"],-x["mix"]))
        configs[k]=dict(chosen=rule,grid=candidates)
        weights[k]=model
        print(k,rule,flush=True)
    winner=max(configs,key=lambda k:(configs[k]["chosen"]["development"]["delta_tp"],
               configs[k]["chosen"]["development"]["repaired_in_target"]))
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/"heads.joblib"
    joblib.dump(weights,dest,compress=3)
    write_json(OUT/"policy.json",dict(chosen=winner,settings=configs,base_threshold=base,
                                     sha256=checksum(dest),development_short_fp=budget))


def test():
    catalog=json.loads((OUT/"policy.json").read_text())
    artifact=OUT/"heads.joblib"
    if catalog["sha256"]!=checksum(artifact):
        raise ValueError("The cross-project readouts changed")
    y,p,short,features,length=samples("test")
    _,_,project,_,parts=inputs()
    group=project[parts["test"]]
    probability=p["compact"][short]
    base=p["compact"]>catalog["base_threshold"]
    target=cohort("joint_low",short,p)
    budget=counts(y[short],base[short])["fp"]
    base_short,_=exact_budget(probability,y[short],budget)
    baseline=base.copy()
    baseline[short]=base_short
    output={}
    for name,model in joblib.load(artifact).items():
        recipe=catalog["settings"][name]["chosen"]
        score=model.predict_proba(design(name,p["compact"],features,length)[short])[:,1]
        posterior=blend(probability,score,recipe["mix"])
        equal=base.copy()
        equal[short],_=exact_budget(posterior,y[short],budget)
        frozen=base.copy()
        frozen[short]=posterior>recipe["threshold"]
        subset={}
        for project in np.unique(group):
            zz=group==project
            subset[project]=summarize_row(y[zz],base[zz],frozen[zz],target[zz])
        output[name]=dict(exact_fp=summarize_row(y,baseline,equal,target),
            exact_fp_short=summarize_row(y[short],base_short,equal[short],target[short]),
            frozen=summarize_row(y,base,frozen,target),frozen_by_project=subset)
        p_full=p["compact"].astype(np.float64).copy()
        p_full[short]=posterior
        with (OUT/f"{name}.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","label","short","joint_low","base_score","candidate_score","base_pred","base_exact_fp","pred_dev_cut","pred_exact_fp"))
            w.writerows(zip(parts["test"],group,y,short.astype(int),target.astype(int),p["compact"],p_full,
                            base.astype(int),baseline.astype(int),frozen.astype(int),equal.astype(int),strict=True))
        print(name,output[name]["exact_fp"],output[name]["frozen"],flush=True)
    write_json(OUT/"analysis.json",dict(chosen=catalog["chosen"],test_short_fp=budget,results=output))


if __name__=="__main__":
    tune()
    test()
