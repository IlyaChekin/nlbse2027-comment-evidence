"""Select exact rule subsets under observed CPU costs and independent FP budgets."""
import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import os
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluate_rules import (HERE, ROOT, R19, WEAK, SOURCE, SEED, PUBLIC_N, base,
    benchmark, combine, comparison, counts, encoder_out, exact_budget, inputs, json_at,
    per_project, pinned, raw_rule_counts, run, scanner, selected_rows, small,
    strict_budget, tokenize, write_json)
from runtime import get_features, scan_chunk
from pipeline import metric
from score_runtime import competition_formula
from students import dataset


OUT=HERE/"subsets"
PROGRAM=None
INCLUDE_NUMERIC=False
GATES=((0.,1.),(.025,.985),(.08,.96),(.16,.90),(.40,.96))
TRAIN_FILE=WEAK/"train_parallel.json"


@dataclass(frozen=True)
class Spec:
    k:int
    left:float
    right:float
    order:str

    @property
    def name(self):
        return f"{self.order}_{self.k}_{self.left:g}_{self.right:g}"


def column_names():
    keys=[name.removeprefix("rule_") for name in json_at(TRAIN_FILE)["columns"]]
    return keys+["195"] if INCLUDE_NUMERIC else keys


@lru_cache(maxsize=1)
def numeric_counts():
    path=HERE/"extended_counts.csv"
    expected=json_at(path.with_suffix(".json"))
    if base.checksum(path)!=expected["sha256"] or base.checksum(HERE/"scan.mjs")!=expected["implementation_sha256"]:
        raise ValueError("The bounded rule features changed")
    rows=np.loadtxt(path,skiprows=1,delimiter=",",dtype=np.int32)
    if not np.array_equal(rows[:,0],np.arange(expected["rows"])):
        raise ValueError("The bounded rule features changed row order")
    return rows[:,1]


def extended(raw,ids):
    if not INCLUDE_NUMERIC:
        return raw
    return np.column_stack((raw,numeric_counts()[ids]))


def profile():
    OUT.mkdir(exist_ok=True)
    rule=column_names()
    text,_,_,_,split=inputs()
    rng=np.random.default_rng(84)
    ids=np.sort(rng.choice(split["fit"],900,replace=False))
    rows=[text[i] for i in ids]
    timings={}
    for tag in (*rule,"all","startup"):
        seq=rows[:2] if tag=="startup" else rows
        runs=[]
        for _ in range(3):
            start=time.perf_counter()
            scan_chunk(seq,None if tag in ("all","startup") else (tag,),PROGRAM)
            runs.append(time.perf_counter()-start)
        timings[tag]=runs
        print(tag,runs,flush=True)
    write_json(OUT/"cost_model.json",dict(probe_rows=ids.tolist(),runs=timings,node=os.environ.get("NLBSE_NODE","node")))


def rates(folder=None):
    stat=json_at((folder or OUT)/"cost_model.json")
    t={k:float(np.median(v)) for k,v in stat["runs"].items()}
    total=(t.pop("all")-t["startup"])/len(stat["probe_rows"])
    t={k:max(.002,t[k]-t["startup"]) for k in t if k!="startup"}
    scale=total/sum(t.values())
    return {k:v*scale for k,v in t.items()}, total


def model(p,raw,y):
    x=get_features(p,raw)[2]
    return LogisticRegression(C=.3,max_iter=500,random_state=42).fit(x,y)


def predict(p,raw,head):
    return combine(p,raw,"linear_all",{"linear":head}).astype(np.float32)


def order_rules(p,raw,y,columns,rr):
    n=counts(y,p>.59)["fp"]
    old,_=exact_budget(p,y,n)
    gain=[]
    for i,key in enumerate(columns):
        matrix=raw[:,i:i+1]
        head=model(p,matrix,y)
        score=predict(p,matrix,head)
        candidate,_=exact_budget(score,y,n)
        extra=comparison(y,old,candidate)["delta_tp"]
        gain.append(float(extra))
    return {
        "quality":sorted(range(len(columns)),key=lambda i:(gain[i],-rr[columns[i]]),reverse=True),
        "efficient":sorted(range(len(columns)),key=lambda i:gain[i]/rr[columns[i]],reverse=True)
    },dict(zip(columns,gain,strict=True))


def build(frame,training,calibration,reference,prior_threshold,keys):
    p,y,raw=frame["reference"],frame["y"],frame["raw"]
    rr,_=rates()
    ordered,gain=order_rules(p[training],raw[training],y[training],keys,rr)
    budget=counts(y[calibration],reference[calibration]>prior_threshold)["fp"]
    configs={}
    ref_bits=reference[calibration]>prior_threshold
    heads={}
    for order,index in ordered.items():
        for k in dict.fromkeys((3,6,10,15,19,len(keys))):
            cols=sorted(index[:k])
            headkey=f"{order}/{k}"
            head=model(p[training],raw[training][:,cols],y[training])
            heads[headkey]=head
            score=predict(p,raw[:,cols],head)
            cpu_unit=sum(rr[keys[i]] for i in cols)
            for low,high in GATES:
                spec=Spec(k,low,high,order)
                active=(p>low)&(p<high)
                fused=np.where(active,score,p)
                b,cut=strict_budget(fused[calibration],y[calibration],budget)
                quality=comparison(y[calibration],ref_bits,b)
                frac=float(active[calibration].mean())
                proxy=.3*quality["f1_after"]-.0002*PUBLIC_N*cpu_unit*frac/4
                configs[spec.name]=dict(spec=asdict(spec),columns=cols,head=headkey,threshold=cut,proxy=proxy,
                                        cal=quality,cpu_unit=cpu_unit,scan_fraction=frac)
    return heads,configs,gain


def use(frame,entry,head):
    spec=Spec(**entry["spec"])
    p=frame["reference"]
    active=(p>spec.left)&(p<spec.right)
    score=predict(p,frame["raw"][:,entry["columns"]],head)
    return np.where(active,score,p),active


def project(rotation):
    target=OUT/rotation
    target.mkdir(parents=True,exist_ok=True)
    columns=column_names()
    old_cfg=json_at(WEAK/rotation/"policy.json")
    threshold=old_cfg["candidates"]["linear_medium"]["cutoff"]
    original=joblib.load(WEAK/rotation/"models.joblib")
    frames={}
    for group in ("dev","test"):
        with np.load(HERE/rotation/f"{group}_scores.npz") as f:
            frames[group]={name:f[name] for name in ("ids","reference","y","raw","project")}
        q=frames[group]
        q["raw"]=extended(q["raw"],q["ids"])
    dev,test=frames["dev"],frames["test"]
    training=np.isin(dev["ids"],old_cfg["fit_rows"])
    calibration=np.isin(dev["ids"],old_cfg["calibration_rows"])
    def old(frame):
        return combine(frame["reference"],frame["raw"][:,:19],"linear_medium",original)
    old_dev=old(dev)
    old_test=old(test)
    heads,options,gain=build(dev,training,calibration,old_dev,threshold,columns)
    bits=old_test>threshold
    budget=counts(test["y"],bits)["fp"]
    ref,_=exact_budget(old_test,test["y"],budget)
    for name,entry in options.items():
        score,active=use(test,entry,heads[entry["head"]])
        equal,_=exact_budget(score,test["y"],budget)
        delta,_=per_project(test["y"],test["project"],old_test,score,bits)
        entry.update(heldout=comparison(test["y"],ref,equal),by_project_tp=delta,
                     frozen=comparison(test["y"],bits,score>entry["threshold"]),test_scans=int(active.sum()))
    chosen=max(options,key=lambda k:options[k]["proxy"])
    write_json(target/"study.json",dict(selected=chosen,method="calibration only",options=options,gain=gain))
    print(rotation,chosen,options[chosen]["heldout"]["delta_tp"],options[chosen]["by_project_tp"],flush=True)


def tune():
    folder=OUT/"official"
    folder.mkdir(exist_ok=True)
    data=[json_at(OUT/r/"study.json")["options"] for r in base.ROTATIONS]
    supported={k for k in data[0] if min(d[k]["by_project_tp"] for d in data)>=-4 and
               sum(d[k]["by_project_tp"] for d in data)>=0}
    _,_,text,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if not np.array_equal(f["rows"],ids):
            raise ValueError("Official calibration rows changed")
        group=f["folds"]
    p,_,cost=encoder_out("official",text,"reference",small.tokenizer())
    raw=raw_rule_counts()[ids]
    frame=dict(reference=p[:,0],y=y[:,0],raw=extended(raw,ids))
    old_cfg=json_at(SOURCE/"policy.json")
    original=joblib.load(SOURCE/"models.joblib")
    old=combine(p[:,0],raw,"linear_medium",original)
    threshold=old_cfg["modes"]["linear_medium"]["threshold"]
    keys=column_names()
    calibration=group==3
    review=group==4
    heads,options,gain=build(frame,group<3,calibration,old,threshold,keys)
    _,full_cost=rates(HERE/"subsets")
    fp=counts(frame["y"][calibration],old[calibration]>threshold)["fp"]
    baseline_fraction=float(selected_rows(p[:,0],"linear_medium").mean())
    result={"reference":dict(threshold=threshold,cpu_unit=full_cost,scan_fraction=baseline_fraction,columns=list(range(19)))}
    for name in supported:
        result[name]=options[name]
    for name,entry in result.items():
        s,asked=(old,selected_rows(p[:,0],"linear_medium")) if name=="reference" else use(frame,entry,heads[entry["head"]])
        b=np.column_stack((s>entry["threshold"],p[:,1]>.5))
        grade=metric(y[calibration],b[calibration])["mean"]["f1"]
        gpu=cost["forward_seconds"]*PUBLIC_N/len(p)
        gflops=cost["analytic_gflops"]*PUBLIC_N/len(p)
        # Rule costs are measured on a single CPU, and normalized to the six-worker full-scanner reference.
        scale=(8.113792-1.780551)/(full_cost*5963)
        forward=gpu+entry["cpu_unit"]*scale*asked.mean()*PUBLIC_N
        entry.update(proxy=competition_formula(grade,forward,gflops),
                     review=comparison(y[review,0],old[review]>threshold,b[review,0]))
    best=max(result,key=lambda k:result[k]["proxy"])
    joblib.dump(heads,folder/"heads.joblib",compress=3)
    write_json(folder/"policy.json",dict(selected=best,modes=result,allowed=sorted(supported),
               fp=fp, columns=keys, gain=gain,program_sha256=base.checksum(PROGRAM) if PROGRAM else None,
               scanner_sha256=base.checksum(HERE/"scan.mjs") if PROGRAM else None))
    print("selected",best,result[best],flush=True)


def public():
    folder=OUT/"official"
    policy=json_at(folder/"policy.json")
    if PROGRAM and (base.checksum(PROGRAM)!=policy["program_sha256"] or
                    base.checksum(HERE/"scan.mjs")!=policy["scanner_sha256"]):
        raise ValueError("The selected rule implementation changed")
    tok=small.tokenizer()
    route,_=pinned()
    pipe=base.open_encoder("official",SEED,"reference",tok)
    original=joblib.load(SOURCE/"models.joblib")
    heads=joblib.load(folder/"heads.joblib")
    rule_columns=[f"rule_{key}" for key in policy["columns"]]
    def runner(name,texts):
        p,cost=run(pipe,texts,route,tokenize)
        at=time.perf_counter()
        entry=policy["modes"][name]
        prior=name=="reference"
        if prior:
            active=selected_rows(p[:,0],"linear_medium")
        else:
            spec=Spec(**entry["spec"])
            active=(p[:,0]>spec.left)&(p[:,0]<spec.right)
        indexes=entry["columns"]
        valid=[rule_columns[i] for i in indexes]
        raw=np.zeros((len(p),len(indexes)),dtype=np.int32)
        if active.any():
            found,x=scanner([t for t,a in zip(texts,active,strict=True) if a],
                            None if prior else tuple(k.removeprefix("rule_") for k in valid),
                            None if prior else PROGRAM)
            if found!=valid:
                raise ValueError("Subset scanner changed feature order")
            raw[active]=x
        if prior:
            p[:,0]=combine(p[:,0],raw,"linear_medium",original)
        else:
            p[active,0]=predict(p[active,0],raw[active],heads[entry["head"]])
        added=time.perf_counter()-at
        cost["wall_seconds"]+=added
        cost["forward_seconds"]+=added
        cost["scanned"]=int(active.sum())
        return p,cost
    names=list(dict.fromkeys(("reference",policy["selected"])))
    benchmark(folder,names,policy,runner,reference=SOURCE/"linear_medium_public.csv")


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument("stage",choices=("profile","project","tune","public"))
    q.add_argument("--rotation",choices=base.ROTATIONS,default="test_a")
    q.add_argument("--implementation",choices=("original","fast","numeric"),default="original")
    a=q.parse_args()
    if a.implementation!="original":
        OUT=HERE/f"subsets_{a.implementation}"
        INCLUDE_NUMERIC=a.implementation=="numeric"
        PROGRAM=HERE/("scan_extended.mjs" if INCLUDE_NUMERIC else "scan.mjs")
        GATES=(*GATES,(.01,.995))
    if a.stage=="project":
        project(a.rotation)
    else:
        {"profile":profile,"tune":tune,"public":public}[a.stage]()
