"""Transfer the selected local-token readout and profile the full public set."""

import argparse
import csv
from hashlib import sha256
from statistics import median
import time

import numpy as np

from experiment import (HERE,ROOT,OFFICIAL,PROJECT,checkpoint,fit,infer,open_model,small,json_at,
                        strict_budget,comparison)
from evaluate import counts, exact_budget
from rotate import checksum, read_partition
from pipeline import PUBLIC, TRAIN, load, metric, write_json
from normalize_code import batch as canonicalize
from score_runtime import competition_formula
from round2_report import bootstrap_delta
from students import dataset, group_codes


SOURCE=ROOT/"round15/official_distill"
SEEDS=(42,101)


def old_policy():
    route=json_at(ROOT/"round12/efficient_aug/policy.json")["selected"]["selective"]
    threshold=json_at(SOURCE/"policy.json")["records"]["original_teacher"]["threshold"]
    return route,threshold


def choose_head():
    dev=[json_at(PROJECT/str(seed)/"policy.json")["dev"] for seed in SEEDS]
    keys=[name for name in dev[0] if name!="control"]
    if any(set(row)!=set(dev[0]) for row in dev):
        raise ValueError("Seeds did not run the same architectures")
    choice=max(keys,key=lambda name:sum(r[name]["matched"]["delta_tp"]-r["control"]["matched"]["delta_tp"] for r in dev))
    return choice,{name:[row[name]["matched"]["delta_tp"] for row in dev] for name in dev[0]}


def run(pipe,texts,route,encoding=None):
    began=time.perf_counter()
    p,cost=infer(pipe,texts,encoding=encoding)
    active=(p[:,0]>route["lower"])&(p[:,0]<route["upper"])
    if active.any():
        before=time.perf_counter()
        bodies=canonicalize([s for s,b in zip(texts,active,strict=True) if b])
        conversion=time.perf_counter()-before
        q,other=infer(pipe,bodies,encoding=encoding)
        p[active,0]=.5*(p[active,0]+q[:,0])
        cost["analytic_gflops"]+=other["analytic_gflops"]
        cost["forward_seconds"]+=other["forward_seconds"]+conversion
    cost.update(wall_seconds=time.perf_counter()-began,routed=int(active.sum()))
    return p,cost


def fit_official():
    name,source=choose_head()
    OFFICIAL.mkdir(exist_ok=True)
    write_json(OFFICIAL/"transfer.json",dict(head=name,selection="mean matched true positives vs matched continuation on project dev",seeds=SEEDS,dev=source))
    fit("official",OFFICIAL,("control",name),42)


def heldout(include_test=True):
    _,_,texts,y,ids,_=dataset(full=True)
    ids=np.asarray(ids)
    r13=read_partition()
    source=np.concatenate((r13["probe"],r13["test"])) if include_test else r13["probe"]
    safe=np.isin(ids,source)
    if np.isin(ids[safe],np.concatenate((r13["fit"],r13["dev"]))).any():
        raise ValueError("Reserved calibration overlaps project fitting or head selection")
    _,_,metadata=load(TRAIN,audit=True)
    files=[(metadata["Projectname"][i],metadata["Filepath"][i]) for i in ids[safe]]
    group=[int.from_bytes(sha256(f"NLBSE-R16-42\0{p}\0{f}".encode()).digest()[:4],"big")%5<3 for p,f in files]
    local=list(np.nonzero(safe)[0])
    cal,review=np.zeros(len(ids),dtype=bool),np.zeros(len(ids),dtype=bool)
    cal[local]=group
    review[local]=~np.asarray(group,dtype=bool)
    if not cal.any() or not review.any():
        raise ValueError("Missing independent official internal groups")
    return [texts[i] for i in local],y[safe],ids[safe],cal[safe],review[safe]


def tune():
    texts,y,ids,cal,review=heldout()
    route,threshold=old_policy()
    mode=json_at(OFFICIAL/"transfer.json")["head"]
    tok=small.tokenizer()
    p,cost=run(open_model("official",OFFICIAL,"reference",tok),texts,route)
    p0=p[:,0]>threshold
    fp=counts(y[cal,0],p0[cal])["fp"]
    eq,_=strict_budget(p[cal,0],y[cal,0],fp)
    scored={}
    for name in ("reference","control",mode):
        q,profile=(p,cost) if name=="reference" else run(open_model("official",OFFICIAL,name,tok),texts,route)
        if not np.array_equal(p[:,1],q[:,1]):
            raise ValueError("Local evidence altered the full-comment MAT rule")
        matched,cut=strict_budget(q[cal,0],y[cal,0],fp)
        if name=="reference":
            cut=threshold
        before = p0[cal]
        bits=q[cal]>(cut,.5)
        quality=metric(y[cal],bits)
        # Compare deterministic counted operations here. Public trials measure both time and operations.
        proxy=competition_formula(quality["mean"]["f1"],0.,profile["analytic_gflops"]*30655/len(y))
        scored[name]=dict(threshold=cut,matched=comparison(y[cal,0],eq,matched),
                          calibration=comparison(y[cal,0],before,bits[:,0]),
                          review=comparison(y[review,0],p0[review],q[review,0]>cut),
                          projected_cost_score=proxy,cost=profile)
        print(name,scored[name],flush=True)
    winner=max(scored,key=lambda key:(scored[key]["projected_cost_score"],key=="reference"))
    write_json(OFFICIAL/"policy.json",dict(selected=winner,head=mode,candidates=scored,base_sha256=checkpoint("official")[1],
               cal_rows=ids[cal].tolist(),review_rows=ids[review].tolist(),fp=fp,route=route))


def write_public(path,ids,y,p,bits,equal,ref):
    with path.open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("row","label_vulnerability","label_mat","probability","vulnerability","mat","equal_fp","reference_equal_fp"))
        w.writerows(zip(ids,y[:,0],y[:,1],p[:,0],bits[:,0].astype(int),bits[:,1].astype(int),equal.astype(int),ref.astype(int),strict=True))


def public():
    cfg=json_at(OFFICIAL/"policy.json")
    winner=cfg["selected"]
    variants=list(dict.fromkeys(("reference",winner)))
    texts,y,meta=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    with (SOURCE/"original_teacher_public.csv").open(newline="") as stream:
        archive=list(csv.DictReader(stream))
    old=np.asarray([[int(r[k]) for k in ("vulnerability","mat")] for r in archive],dtype=bool)
    if [int(row["row"]) for row in archive]!=ids:
        raise ValueError("Public archive row IDs changed")
    groups=group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    tok=small.tokenizer()
    pipes={k:open_model("official",OFFICIAL,k,tok) for k in variants}
    for p in pipes.values():
        run(p,texts[:2048],cfg["route"])
    trials={k:[] for k in variants}
    scores,outputs={},{}
    for i in range(3):
        for name in variants if i%2==0 else reversed(variants):
            p,c=run(pipes[name],texts,cfg["route"])
            bits=p>(cfg["candidates"][name]["threshold"],.5)
            if name=="reference" and not np.array_equal(bits,old):
                raise ValueError(f"Round 15 reference changed: {np.count_nonzero(bits!=old)} decisions")
            if name in outputs and not np.array_equal(outputs[name],bits):
                raise ValueError("Local-head predictions changed between timing repetitions")
            trials[name].append(c)
            scores[name],outputs[name]=p,bits
            print(name,c,flush=True)
    before=old[:,0]
    fp=counts(y[:,0],before)["fp"]
    ref,_=exact_budget(scores["reference"][:,0],y[:,0],fp)
    report={}
    for name,runs in trials.items():
        p,bits=scores[name],outputs[name]
        if not np.array_equal(bits[:,1],old[:,1]):
            raise ValueError("MAT decisions changed during the public comparison")
        cost={k:median(x[k] for x in runs) for k in runs[0]}
        m=metric(y,bits)
        equal,_=exact_budget(p[:,0],y[:,0],fp)
        report[name]=dict(quality=m,cost=cost,trials=runs,score=competition_formula(m["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                          matched=comparison(y[:,0],ref,equal),frozen=comparison(y[:,0],before,bits[:,0]),
                          paired_interval=bootstrap_delta(y,old,bits,groups))
        write_public(OFFICIAL/f"{name}_public.csv",ids,y,p,bits,equal,ref)
    write_json(OFFICIAL/"results.json",report)
    print({k:(r["score"],r["matched"]["delta_tp"]) for k,r in report.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fit","tune","public"))
    {"fit":fit_official,"tune":tune,"public":public}[p.parse_args().stage]()
