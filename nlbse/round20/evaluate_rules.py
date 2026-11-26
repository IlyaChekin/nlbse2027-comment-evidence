"""Calibrate rule-guided routing before public measurements."""
import argparse
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

from student import HERE,ROOT,R19,WEAK,base,open_student
from transfer import benchmark,comparison,counts,exact_budget,infer,inputs,json_at,pinned,run
from transfer import small,strict_budget,tokenize,write_json
from score_runtime import competition_formula
from students import dataset

sys.path.insert(0,str(WEAK))
from runtime import combine,scanner,selected_rows


SOURCE = WEAK/"official"
FINAL = HERE/"official"
PUBLIC_N = 30655
SECONDS_PER_RULE_SCAN = (8.1138-1.7806)/5963
SEED = 42


@dataclass(frozen=True)
class Option:
    name: str
    estimate: str = "p"
    low_fraction: float = 0.
    skip_medium: float = 0.
    boost: float = 6.
    use_exact: bool = True


OPTIONS = [Option("reference")]
for estimator in ("p","probe","joint"):
    for fraction in (.025,.075,.15,.30):
        for shift in (5.,7.):
            OPTIONS.append(Option(f"rescue_{estimator}_{fraction:g}_{shift:g}",estimator,fraction,0.,shift))
for estimator in ("probe","joint"):
    for fraction in (.25,.50,.70):
        OPTIONS.append(Option(f"skip_{estimator}_{fraction:g}",estimator,0.,fraction))
    for fraction in (.075,.15):
        OPTIONS.append(Option(f"both_{estimator}_{fraction:g}",estimator,fraction,.50,7.))
    for fraction in (.025,.075,.15):
        OPTIONS.append(Option(f"direct_{estimator}_{fraction:g}",estimator,fraction,0.,6.,False))


@lru_cache(maxsize=1)
def raw_rule_counts():
    path = WEAK/"train_parallel.csv"
    header = json_at(path.with_suffix(".json"))
    if base.checksum(path)!=header["sha256"]:
        raise ValueError("Exact rule labels changed")
    raw = np.loadtxt(path,skiprows=1,delimiter=",",dtype=np.int32)
    if not np.array_equal(raw[:,0],np.arange(header["rows"])):
        raise ValueError("Exact rule labels were reordered")
    return raw[:,1:]


def encoder_out(rotation,texts,mode,tok):
    if mode in ("pooled","reference"):
        pipe = base.open_encoder(rotation,SEED,mode,tok)
        route,_ = pinned()
        p,cost = run(pipe,texts,route,tokenize)
        return p,None,cost
    model = open_student(rotation,SEED,mode,tok)
    return infer_student(SimpleNamespace(encoder=model,tok=tok,mode="control"),texts,pinned()[0])


def infer_student(pipe,texts,route,columns=20,return_all=False):
    start=time.perf_counter()
    aux = np.empty((len(texts),columns),dtype=np.float32)
    p,cost = infer(pipe,texts,encoding=tokenize,aux_out=aux)
    base_pipe=SimpleNamespace(encoder=pipe.encoder.encoder,tok=pipe.tok,mode="control")
    active=(p[:,0]>route["lower"]) & (p[:,0]<route["upper"])
    if active.any():
        canonicalize_at=time.perf_counter()
        from normalize_code import batch as canonicalize
        bodies=canonicalize([s for s,enable in zip(texts,active,strict=True) if enable])
        cost["forward_seconds"]+=time.perf_counter()-canonicalize_at
        q,other=infer(base_pipe,bodies,encoding=tokenize)
        p[active,0]=.5*(p[active,0]+q[:,0])
        for name in ("forward_seconds","analytic_gflops"):
            cost[name]+=other[name]
    cost.update(wall_seconds=time.perf_counter()-start,routed=int(active.sum()))
    return p,(aux if return_all else aux[:,0]),cost


def dev_test(rotation):
    texts, y, project, _, _ = inputs()
    folder = HERE/rotation
    tok = small.tokenizer()
    raw = raw_rule_counts()
    cache = {}
    with np.load(R19/f"{rotation}.npz") as f:
        for part in ("dev","test"):
            ids = f[part]
            sample = [texts[i] for i in ids]
            values = {}
            for name in ("reference","pooled","probe","joint"):
                pred,aux,cost = encoder_out(rotation,sample,name,tok)
                values[name] = pred[:,0]
                if aux is not None:
                    values[name+"_q"] = aux
                values[name+"_cost"] = np.array([cost["forward_seconds"],cost["analytic_gflops"]])
            cache[part] = dict(ids=ids,y=y[ids,0],project=project[ids],raw=raw[ids],**values)
            np.savez_compressed(folder/f"{part}_scores.npz",**cache[part])
            print(rotation,part,len(ids),flush=True)
    heads = joblib.load(WEAK/rotation/"models.joblib")
    examine(rotation,cache,heads)


def quantile(x,q,default):
    return float(np.quantile(x,q)) if len(x) else default


def thresholds(option,p,q):
    return {
        "low":quantile(q[p<=.08],1-option.low_fraction,2.) if option.low_fraction else 2.,
        "medium":quantile(q[selected_rows(p,"linear_medium")],option.skip_medium,-1.) if option.skip_medium else -1.,
    }


def masks(option,p,q,gate):
    middle = selected_rows(p,"linear_medium") & (q>=gate["medium"])
    low = (p<=.08) & (q>gate["low"]) if option.low_fraction else np.zeros(len(p),dtype=bool)
    exact = middle | (low if option.use_exact else False)
    return middle,low,exact


def apply(option,p,q,gate,raw,heads):
    middle,low,exact = masks(option,p,q,gate)
    rule_input = np.zeros_like(raw)
    rule_input[middle] = raw[middle]
    combined = combine(p,rule_input,"linear_medium",heads).astype(np.float32)
    evidence = (raw>0).any(1) if option.use_exact else q
    extra = low & (evidence>0)
    moved = expit(logit(p[extra].clip(1e-5,1-1e-5))+option.boost*evidence[extra])
    combined[extra] = np.maximum(combined[extra],moved)
    return combined,exact,low


def reference_score(frame,heads):
    p = frame["reference"]
    return combine(p,frame["raw"],"linear_medium",heads).astype(np.float32)


def aux_scores(frame,option):
    return frame["reference"] if option.estimate=="p" else frame[option.estimate+"_q"]


def per_project(y,prj,reference,candidate,reference_bits):
    delta = 0
    separate = {}
    for name in sorted(set(prj)):
        m = prj==name
        n = counts(y[m],reference_bits[m])["fp"]
        a,_ = exact_budget(reference[m],y[m],n)
        b,_ = exact_budget(candidate[m],y[m],n)
        item = comparison(y[m],a,b)
        if item["delta_fp"]:
            raise ValueError("Fixed-FP project comparison changed by policy")
        separate[name] = item
        delta += item["delta_tp"]
    return delta,separate


def examine(rotation,cache,heads):
    dev,test = cache["dev"],cache["test"]
    folder = HERE/rotation
    old = reference_score(dev,heads)
    ref_threshold = json_at(WEAK/rotation/"policy.json")["candidates"]["linear_medium"]["cutoff"]
    bits = old>ref_threshold
    fp = counts(dev["y"],bits)["fp"]
    base_test=reference_score(test,heads)
    reference_test=base_test>ref_threshold
    budget_test=counts(test["y"],reference_test)["fp"]
    base_equal,_=exact_budget(base_test,test["y"],budget_test)
    config = {}
    rankings = {}
    net = {}
    p_test=test["reference"]
    p_dev=dev["reference"]
    y_rule=(test["raw"]>0).any(1)
    low = p_test<=.08
    needed = low & y_rule & (test["y"]==1)
    base_budget=counts(test["y"],p_test>.50)["fp"]
    ctrl_bits,_=exact_budget(p_test,test["y"],base_budget)
    for name in ("pooled","probe","joint"):
        at,_=exact_budget(test[name],test["y"],base_budget)
        net[name]=comparison(test["y"],ctrl_bits,at)
        if name+"_q" in test:
            rankings[name]=float(average_precision_score(y_rule[low],test[name+"_q"][low]))
    rankings["p"]=float(average_precision_score(y_rule[low],p_test[low]))
    import csv
    with (folder/"policies.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(("name","dev_proxy","calls","low_calls","low_true_rule","low_false_rule",
                         "delta_tp","project_tp","frozen_tp","frozen_fp"))
        for option in OPTIONS:
            gate=thresholds(option,p_dev,aux_scores(dev,option))
            p,q,l=apply(option,p_dev,aux_scores(dev,option),gate,dev["raw"],heads)
            m,cut=strict_budget(p,dev["y"],fp)
            selected_cut=ref_threshold if option.name=="reference" else cut
            quality=comparison(dev["y"],bits,p>selected_cut)["f1_after"]
            proxy=.3*quality - .0002*SECONDS_PER_RULE_SCAN*PUBLIC_N*q.mean()
            candidate,asked,asked_low=apply(option,p_test,aux_scores(test,option),gate,test["raw"],heads)
            rank,_=exact_budget(candidate,test["y"],budget_test)
            result=comparison(test["y"],base_equal,rank)
            fixed=comparison(test["y"],reference_test,candidate>selected_cut)
            delta,per=per_project(test["y"],test["project"],base_test,candidate,reference_test)
            row=dict(option=asdict(option),gate=gate,threshold=selected_cut,proxy=proxy,
                     fixed=fixed,matched=result,project=per,delta_project_tp=delta,
                     scan_count=int(asked.sum()),low_count=int(asked_low.sum()),
                     low_true_rules=int((asked_low & needed).sum()),low_false_rules=int((asked_low & y_rule & ~test["y"].astype(bool)).sum()))
            config[option.name]=row
            writer.writerow((option.name,proxy,row["scan_count"],row["low_count"],row["low_true_rules"],
                             row["low_false_rules"],result["delta_tp"],delta,fixed["delta_tp"],fixed["delta_fp"]))
    chosen=max(config,key=lambda name:config[name]["proxy"])
    write_json(folder/"study.json",dict(selected=chosen,rule_ap_low=rankings,neural_matched=net,policies=config,
               test_fp=budget_test,dev_fp=fp))
    print(rotation,"selected",chosen,"rule AP",rankings,"neural", {n:r["delta_tp"] for n,r in net.items()},flush=True)


def allowed_options():
    outcomes=[json_at(HERE/r/"study.json")["policies"] for r in base.ROTATIONS]
    allowed=[]
    for option in OPTIONS:
        per=[report[option.name] for report in outcomes]
        if option.name=="reference" or (sum(p["delta_project_tp"] for p in per)>=0 and min(p["delta_project_tp"] for p in per)>=-3):
            allowed.append(option)
    return allowed


def official_tune():
    _, _, text, y, ids, _ = dataset(full=True)
    ids=np.asarray(ids)
    with np.load(ROOT/"round8/frozen/holdout.npz") as folds:
        if not np.array_equal(ids,folds["rows"]):
            raise ValueError("Official holdout files changed")
        calibration=folds["folds"]==3
        review=folds["folds"]==4
    tok=small.tokenizer()
    raw=raw_rule_counts()[ids]
    heads=joblib.load(SOURCE/"models.joblib")
    reference=json_at(SOURCE/"policy.json")
    p,_,baseline_cost=encoder_out("official",text,"reference",tok)
    old=combine(p[:,0],raw,"linear_medium",heads).astype(np.float32)
    threshold=reference["modes"]["linear_medium"]["threshold"]
    original=old>threshold
    budget=counts(y[calibration,0],original[calibration])["fp"]
    aux={}
    times={}
    for name in ("probe","joint"):
        _,aux[name],times[name]=encoder_out("official",text,name,tok)
    options=allowed_options()
    output={}
    for option in options:
        scores=aux.get(option.estimate,p[:,0])
        gate=thresholds(option,p[calibration,0],scores[calibration])
        s,asked,_=apply(option,p[:,0],scores,gate,raw,heads)
        _,cut=strict_budget(s[calibration],y[calibration,0],budget)
        if option.name=="reference":
            cut=threshold
        b=s>cut
        cal=comparison(y[calibration,0],original[calibration],b[calibration])
        rev=comparison(y[review,0],original[review],b[review])
        f1=(cal["f1_after"]+2*np.count_nonzero(y[calibration,1] & (p[calibration,1]>.5))/
            max(int(y[calibration,1].sum())+int((p[calibration,1]>.5).sum()),1))/2
        cpu=float(asked.mean())*PUBLIC_N*SECONDS_PER_RULE_SCAN
        neural=times.get(option.estimate,{"forward_seconds":0.,"analytic_gflops":0.}).copy()
        if option.estimate=="probe":
            for k in neural:
                neural[k]=max(0.,neural[k]-baseline_cost[k])
        forward=cpu+(baseline_cost["forward_seconds"]+neural["forward_seconds"])*PUBLIC_N/len(text)
        ops=(baseline_cost["analytic_gflops"]+neural["analytic_gflops"])*PUBLIC_N/len(text)
        proxy=competition_formula(f1,forward,ops)
        output[option.name]=dict(option=asdict(option),gate=gate,threshold=cut,proxy=proxy,
                                 cal=cal,review=rev,projected_scans=float(asked.mean())*PUBLIC_N)
    chosen=max(output,key=lambda k:output[k]["proxy"])
    FINAL.mkdir(exist_ok=True)
    write_json(FINAL/"policy.json",dict(selected=chosen,modes=output,reference_sha256=base.checksum(SOURCE/"policy.json"),
               selection="internal fold 3",review="internal fold 4"))
    print("selected",chosen,output[chosen],flush=True)


def official_public():
    info=json_at(FINAL/"policy.json")
    rule=json_at(SOURCE/"policy.json")
    if base.checksum(SOURCE/"policy.json")!=info["reference_sha256"]:
        raise ValueError("The frozen round 19 policy changed")
    heads=joblib.load(SOURCE/"models.joblib")
    tok=small.tokenizer()
    ref=base.open_encoder("official",SEED,"reference",tok)
    aux={}
    pick=info["modes"][info["selected"]]["option"]
    if pick["estimate"]!="p":
        m=open_student("official",SEED,pick["estimate"],tok)
        aux[pick["estimate"]]=SimpleNamespace(encoder=m,tok=tok,mode="control")
    def runner(name,texts):
        entry=info["modes"][name]
        option=Option(**entry["option"])
        if option.estimate=="probe":
            p,q,cost=infer_student(aux["probe"],texts,rule["route"])
        else:
            p,cost=run(ref,texts,rule["route"],tokenize)
            q=p[:,0]
        if option.estimate=="joint":
            computed=np.empty((len(texts),20),dtype=np.float32)
            _,extra=infer(aux[option.estimate],texts,encoding=tokenize,aux_out=computed)
            for k in ("wall_seconds","forward_seconds","analytic_gflops"):
                cost[k]+=extra[k]
            q=computed[:,0]
        start=time.perf_counter()
        _,_,search=masks(option,p[:,0],q,entry["gate"])
        raw=np.zeros((len(texts),19),dtype=np.int32)
        if search.any():
            columns,values=scanner([t for t,ask in zip(texts,search,strict=True) if ask])
            if columns!=rule["rule_columns"]:
                raise ValueError("Exact rule scan columns changed")
            raw[search]=values
        p[:,0],_,_=apply(option,p[:,0],q,entry["gate"],raw,heads)
        tail=time.perf_counter()-start
        cost["wall_seconds"]+=tail
        cost["forward_seconds"]+=tail
        cost["scanned"]=int(search.sum())
        return p,cost
    names=list(dict.fromkeys(("reference",info["selected"])))
    benchmark(FINAL,names,info,runner,reference=SOURCE/"linear_medium_public.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("project","tune","public"))
    p.add_argument("--rotation",choices=base.ROTATIONS,default="test_a")
    a=p.parse_args()
    if a.stage=="project":
        dev_test(a.rotation)
    else:
        {"tune":official_tune,"public":official_public}[a.stage]()
