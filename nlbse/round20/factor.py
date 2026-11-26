"""Compare conditional predictions with a value-of-information rule gate."""
import argparse
from dataclasses import asdict
import time
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from evaluate_rules import (HERE, ROOT, R19, WEAK, SOURCE, SEED, PUBLIC_N, SECONDS_PER_RULE_SCAN,
    base, benchmark, comparison, counts, encoder_out, exact_budget, infer_student,
    inputs, json_at, open_student, per_project, pinned, raw_rule_counts, run, scanner, small,
    strict_budget, tokenize, write_json)
from runtime import combine, refine
from pipeline import metric
from score_runtime import competition_formula
from students import dataset
from subsets import numeric_counts
from conditional import Policy,choose,estimate,features,probabilities,query_value,valid,weak


OUT=HERE/"factor"
CACHE=OUT
NUMERIC=False


POLICIES=[
    Policy(mix,fraction,method) for mix in (.50,1.)
    for fraction in (0.,.06,.12,.24) for method in ("value","margin")
    if fraction or method=="value"
]


def pipe(rotation,tok):
    return SimpleNamespace(encoder=open_student(rotation,SEED,"factor",tok),tok=tok,mode="control")


def factor_data(rotation,text,tok):
    p,a,c=infer_student(pipe(rotation,tok),text,pinned()[0],columns=22,return_all=True)
    result=np.column_stack((p[:,0],a[:,0],a[:,-2:]))
    return result,p[:,1]>.5,c


def fit(values,raw,y,train):
    v=train & valid(raw)
    fired=weak(raw)
    return {kind:LogisticRegression(C=.3,max_iter=350,random_state=42).fit(
            features(values[mask],kind),target[mask])
        for kind,mask,target in (
            ("w",v,fired),("negative",v & ~fired,y),("positive",v & fired,y)
        )}


def calibrate(values,est,policy,y,budget):
    prior,no,yes,w=probabilities(values,est,policy)
    _,t=strict_budget(prior,y,budget)
    if not policy.fraction:
        return {"utility":2.,"base":t}
    value=query_value(policy,prior,no,yes,w,t)
    return {"utility":float(np.quantile(value,1-policy.fraction)),"base":t}


def apply(values,est,policy,gate,raw):
    prior,no,yes,active=choose(values,est,policy,gate)
    output=prior.astype(np.float32)
    observed=active & valid(raw)
    output[observed]=np.where(weak(raw)[observed],yes[observed],no[observed])
    return output,active


def collect(rotation):
    target=CACHE/rotation
    target.mkdir(parents=True,exist_ok=True)
    texts,y,prj,_,_=inputs()
    tok=small.tokenizer()
    with np.load(R19/f"{rotation}.npz") as f:
       for part in ("dev","test"):
          ids=f[part]
          values,mat,cost=factor_data(rotation,[texts[i] for i in ids],tok)
          np.savez_compressed(target/f"{part}.npz",ids=ids,values=values,mat=mat.astype(bool),
                              model_cost=np.asarray([cost["forward_seconds"],cost["analytic_gflops"]]))
          print(rotation,part, len(ids),cost,flush=True)


def evaluate(rotation):
    target=OUT/rotation
    target.mkdir(parents=True,exist_ok=True)
    cfg=json_at(WEAK/rotation/"policy.json")
    reference_head=joblib.load(WEAK/rotation/"models.joblib")
    y=inputs()[1]
    prj=inputs()[2]
    frames={}
    for part in ("dev","test"):
        with np.load(CACHE/rotation/f"{part}.npz") as f,np.load(HERE/rotation/f"{part}_scores.npz") as g:
            ids=f["ids"]
            if not np.array_equal(ids,g["ids"]):
                raise ValueError("Factorized outputs changed the selected rows")
            raw=g["raw"]
            expanded=np.column_stack((raw,numeric_counts()[ids])) if NUMERIC else raw
            frames[part]=dict(ids=ids,values=f["values"],raw=expanded,y=y[ids,0],project=prj[ids],
                              reference=combine(g["reference"],g["raw"],"linear_medium",reference_head))
    d,t=frames["dev"],frames["test"]
    train=np.isin(d["ids"],cfg["fit_rows"])
    cal=np.isin(d["ids"],cfg["calibration_rows"])
    models=fit(d["values"],d["raw"],d["y"],train)
    e=estimate(models,d["values"])
    et=estimate(models,t["values"])
    estimated={k:v[cal] for k,v in e.items()}
    cut=cfg["candidates"]["linear_medium"]["cutoff"]
    dev_zero=d["reference"]>cut
    test_zero=t["reference"]>cut
    fp=counts(d["y"][cal],dev_zero[cal])["fp"]
    t_fp=counts(t["y"],test_zero)["fp"]
    rank,_=exact_budget(t["reference"],t["y"],t_fp)
    result={}
    for policy in POLICIES:
        gate=calibrate(d["values"][cal],estimated,policy,d["y"][cal],fp)
        a,_=apply(d["values"],e,policy,gate,d["raw"])
        b,threshold=strict_budget(a[cal],d["y"][cal],fp)
        z,scanned=apply(t["values"],et,policy,gate,t["raw"])
        ticket,_=exact_budget(z,t["y"],t_fp)
        delta,per=per_project(t["y"],t["project"],t["reference"],z,test_zero)
        row=dict(policy=asdict(policy),gate=gate,threshold=threshold,
                 calibration=comparison(d["y"][cal],dev_zero[cal],b),
                 matched=comparison(t["y"],rank,ticket),
                 frozen=comparison(t["y"],test_zero,z>threshold),
                 project=per,project_tp=delta,scanned=int(scanned.sum()))
        result[policy.name]=row
    selected=max(result,key=lambda k:.3*result[k]["calibration"]["f1_after"]-.0002*PUBLIC_N*SECONDS_PER_RULE_SCAN*result[k]["policy"]["fraction"])
    safe=valid(t["raw"])
    audit=dict(rule_ap=float(average_precision_score(weak(t["raw"])[safe],et["w"][safe])),
               rule_brier=float(brier_score_loss(weak(t["raw"])[safe],et["w"][safe])))
    write_json(target/"study.json",dict(options=result,selected=selected,audit=audit,test_fp=t_fp,dev_fp=fp))
    print(rotation,selected,result[selected]["project_tp"],result[selected]["scanned"],flush=True)


def official_tune():
    folder=OUT/"official"
    folder.mkdir(exist_ok=True)
    samples=[json_at(OUT/r/"study.json")["options"] for r in base.ROTATIONS]
    supported={p.name for p in POLICIES if min(v[p.name]["project_tp"] for v in samples)>=-4
               and sum(v[p.name]["project_tp"] for v in samples)>=0}
    _,_,text,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if not np.array_equal(ids,f["rows"]):
            raise ValueError("Official calibration group changed")
        groups=f["folds"]
    cal=groups==3
    rev=groups==4
    tok=small.tokenizer()
    old,_,cost=encoder_out("official",text,"reference",tok)
    value,mat,student_cost=factor_data("official",text,tok)
    if not np.array_equal(old[:,1]>.5,mat):
        raise ValueError("Factorized output changed the deterministic MAT rule")
    verification=json_at(HERE/"scan_equivalence.json")
    source=base.checksum(HERE/"scan.mjs")
    if verification["changes"] or verification["program_sha256"]!=source:
        raise ValueError("The fast exact rule scanner has not passed its equivalence check")
    timing=json_at(HERE/"scan_timing.json")["timings"]
    acceleration=float(np.median(timing["fast"])/np.median(timing["original"]))
    raw=raw_rule_counts()[ids]
    original=json_at(SOURCE/"policy.json")
    old_heads=joblib.load(SOURCE/"models.joblib")
    ref_score=combine(old[:,0],raw,"linear_medium",old_heads)
    threshold=original["modes"]["linear_medium"]["threshold"]
    ref_bits=ref_score>threshold
    fp=counts(y[cal,0],ref_bits[cal])["fp"]
    observed=np.column_stack((raw,numeric_counts()[ids])) if NUMERIC else raw
    if NUMERIC:
        recipe=json_at(HERE/"subsets_numeric/cost_model.json")
        base_profile=json_at(HERE/"subsets/cost_model.json")
        def per_row(data):
            return (np.median(data["runs"]["all"])-np.median(data["runs"]["startup"]))/len(data["probe_rows"])
        acceleration=float(per_row(recipe)/per_row(base_profile))
    heads=fit(value,observed,y[:,0],groups<3)
    e=estimate(heads,value)
    est={k:v[cal] for k,v in e.items()}
    live=config_proxy(y,np.column_stack((ref_bits,mat)),cal,cost,((old[:,0]>.08)&(old[:,0]<.96)).mean())
    picks={"reference":dict(threshold=threshold,proxy=live,scans=float(((old[:,0]>.08)&(old[:,0]<.96)).mean()))}
    for policy in POLICIES:
        if policy.name not in supported:
            continue
        gate=calibrate(value[cal],est,policy,y[cal,0],fp)
        score,scan=apply(value,e,policy,gate,observed)
        _,cut=strict_budget(score[cal],y[cal,0],fp)
        bits=np.column_stack((score>cut,mat))
        proxy=config_proxy(y,bits,cal,student_cost,scan.mean(),acceleration)
        picks[policy.name]=dict(policy=asdict(policy),gate=gate,threshold=cut,proxy=proxy,
            calibration=comparison(y[cal,0],ref_bits[cal],bits[cal,0]),
            review=comparison(y[rev,0],ref_bits[rev],bits[rev,0]),scans=float(scan.mean()))
    selected=max(picks,key=lambda n:picks[n]["proxy"])
    joblib.dump(heads,folder/"heads.joblib",compress=3)
    program=HERE/("scan_extended.mjs" if NUMERIC else "scan.mjs")
    write_json(folder/"policy.json",dict(selected=selected,modes=picks,fp=fp,supported=sorted(supported),
                 scanner_sha256=source,estimate_scan_ratio=acceleration,numeric=NUMERIC,
                 program_sha256=base.checksum(program)))
    print("selected",selected,picks[selected],flush=True)


def config_proxy(y,bits,cal,model_cost,fraction,relative=1.):
    m=metric(y[cal],bits[cal])["mean"]["f1"]
    tm=model_cost["forward_seconds"]*PUBLIC_N/len(y)+fraction*PUBLIC_N*SECONDS_PER_RULE_SCAN*relative
    return competition_formula(m,tm,model_cost["analytic_gflops"]*PUBLIC_N/len(y))


def public():
    folder=OUT/"official"
    info=json_at(folder/"policy.json")
    if base.checksum(HERE/"scan.mjs")!=info["scanner_sha256"]:
        raise ValueError("The exact scanner changed between calibration and evaluation")
    program=HERE/("scan_extended.mjs" if NUMERIC else "scan.mjs")
    if base.checksum(program)!=info["program_sha256"] or NUMERIC!=info["numeric"]:
        raise ValueError("The frozen diagnostic rule set changed")
    route,_=pinned()
    tok=small.tokenizer()
    reference=base.open_encoder("official",SEED,"reference",tok)
    original=json_at(SOURCE/"policy.json")
    original_heads=joblib.load(SOURCE/"models.joblib")
    model=pipe("official",tok) if info["selected"]!="reference" else None
    heads=joblib.load(folder/"heads.joblib")
    def runner(name,texts):
        if name=="reference":
            p,cost=run(reference,texts,route,tokenize)
            start=time.perf_counter()
            scanned=int(((p[:,0]>.08)&(p[:,0]<.96)).sum())
            p[:,0]=refine(texts,p[:,0],original,original_heads)
        else:
            p,aux,cost=infer_student(model,texts,route,columns=22,return_all=True)
            start=time.perf_counter()
            value=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
            policy=Policy(**info["modes"][name]["policy"])
            collected=estimate(heads,value)
            score,no,yes,active=choose(value,collected,policy,info["modes"][name]["gate"])
            scanned=int(active.sum())
            if scanned:
                cols,raw=scanner([t for t,a in zip(texts,active,strict=True) if a],program=program)
                expected=original["rule_columns"]+(["rule_195"] if NUMERIC else [])
                if cols!=expected:
                    raise ValueError("Observed outcomes came from a different rule set")
                score[active]=np.where(valid(raw),np.where(weak(raw),yes[active],no[active]),score[active])
            p[:,0]=score
        added=time.perf_counter()-start
        cost["wall_seconds"]+=added
        cost["forward_seconds"]+=added
        cost["scanned"]=scanned
        return p,cost
    models=list(dict.fromkeys(("reference",info["selected"])))
    benchmark(folder,models,info,runner,reference=SOURCE/"linear_medium_public.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("collect","project","tune","public"))
    p.add_argument("--rotation",choices=base.ROTATIONS,default="test_a")
    p.add_argument("--numeric",action="store_true")
    a=p.parse_args()
    if a.numeric:
        NUMERIC=True
        OUT=HERE/"factor_numeric"
    if a.stage in ("collect","project"):
        {"collect":collect,"project":evaluate}[a.stage](a.rotation)
    else:
        {"tune":official_tune,"public":public}[a.stage]()
