"""Measure one frozen section-scoring correction end to end on the public dataset."""
import csv
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np

from view_scores import compute
from length_eval import HERE,ROOT,R24,PARTS,f,REFERENCE,recipe,log_odds,checksum,json_at,write_json
from fuse_sparse import aligned,calibrate_sparse,proposals,apply as sparse_apply
from union_scanner import scan,PROGRAM
from layout_official import recover,load,PUBLIC

VIEWS=HERE/"views"
OUT=VIEWS/"public"


def select():
    groups=[json_at(VIEWS/k/"results.json")["results"] for k in PARTS]
    paper=json_at(VIEWS/"official/results.json")
    support={}
    for key,item in paper["results"].items():
        gains=[q[key]["matched"]["delta_tp"] for q in groups]
        if min(gains)>=0 and max(gains)>0 and item["proxy"]>0:
            support[key]=dict(**item,project_tp=gains)
    picked=max(support,key=lambda k:support[k]["proxy"],default=REFERENCE)
    OUT.mkdir(exist_ok=True)
    record=dict(selected=picked,support=support,length_cutoff=paper["length_cutoff"],
                source_policy_sha256=checksum(VIEWS/"official/results.json"),
                calibrator_sha256=paper["model_sha256"],scoped_source_sha256=checksum(R24/"official/tuned/policy.json"))
    write_json(OUT/"policy.json",record)
    return record


def adjust(texts,values,state,threshold,base_variant,package,option=None,length_cut=None,head=None):
    active=state["known"]&~state["flag"]
    p=state["p"].copy()
    original=p>threshold
    touched=np.zeros(len(p),dtype=bool)
    nnz=0
    if not active.any():
        return original,p,touched,nnz,0.
    local=[s for s,on in zip(texts,active,strict=True) if on]
    compact={k:v[active] for k,v in state.items()}
    raw,nnz=aligned(package["expert"],local,"token")
    sparse=calibrate_sparse(package["oof"],raw)
    recalib=proposals({REFERENCE:values[active]},compact,sparse,package["combiner"],threshold)[0]
    _,score=sparse_apply(compact,recalib,threshold,"blend","no",.5)
    cutoff=base_variant["threshold"]
    new=score>cutoff
    p[active]=score
    original[active]=new
    start=time.perf_counter()
    if option:
        chose=np.ones(len(local),dtype=bool)
        branch=option["branch"]
        if branch!="all":
            length=np.fromiter((len(s) for s in local),dtype=np.int32)
            chose &= (length<=length_cut)==(branch=="short")
        if option["distance"] is not None:
            chose &= abs(score-cutoff)<option["distance"]
        if chose.any():
            extra,total=compute([s for s,on in zip(local,chose,strict=True) if on],
                                 option["region"],package["expert"][0])
            x=np.column_stack((f.features(values[active][chose],"negative"),log_odds(sparse[chose,0]),
                                log_odds(score[chose]),extra))
            w=option["damping"]
            new_score=(1.-w)*score[chose]+w*head.predict_proba(x)[:,1]
            original[np.flatnonzero(active)[chose]]=new_score>option["threshold"]
            p[np.flatnonzero(active)[chose]]=new_score
            nnz+=total
        touched[active]=chose
    return original,p,touched,nnz,time.perf_counter()-start


def run():
    saved=select()
    selected=saved["selected"]
    print("selected",selected,flush=True)
    if selected==REFERENCE:
        return
    option=saved["support"][selected]
    package_file=R24/"official/tuned/models.joblib"
    config=json_at(R24/"official/tuned/policy.json")
    if checksum(package_file)!=config["models_sha256"] or checksum(VIEWS/"official/calibrator.joblib")!=saved["calibrator_sha256"]:
        raise ValueError("Selected learned models changed")
    package=joblib.load(package_file)["token"]
    h=joblib.load(VIEWS/"official/calibrator.joblib")[option["region"]]
    base=config["candidates"][config["selected"]]
    ref,policy,args=recipe()
    text,y,_=load(PUBLIC)
    pub=json_at(R24/"public/results.json")
    pfile=R24/"public"/f"{pub['selected'].replace(':','_')}.csv"
    if checksum(pfile)!=pub["results"][pub["selected"]]["predictions_sha256"]:
        raise ValueError("Round 24 public predictions changed")
    with pfile.open(newline="") as stream:
        archive=np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")]
                            for r in csv.DictReader(stream)],dtype=bool)
    model=f.pipe("official",f.small.tokenizer())
    def forward(which,texts):
        p,aux,cost=f.infer_student(model,texts,f.pinned()[0],columns=22,return_all=True)
        start=time.perf_counter()
        values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
        est=f.estimate(args,values)
        score,no,yes,ask=f.choose(values,est,policy,ref["gate"])
        flags=np.zeros(len(p),dtype=bool)
        found=np.zeros_like(flags)
        if ask.any():
            outcomes=scan([s for s,k in zip(texts,ask,strict=True) if k])
            found[ask]=outcomes>=0
            flags[ask]=outcomes==1
            score[found]=np.where(flags[found],yes[found],no[found])
        state=dict(p=score.astype(np.float32),known=found,flag=flags,w=est["w"],no=no,yes=yes)
        mark=recover(texts,p[:,1]>.5)
        estimate,proba,touched,nnz,added=adjust(texts,values,state,ref["threshold"],base,
                                      package,option if which!=REFERENCE else None,saved["length_cutoff"],h)
        elapsed=time.perf_counter()-start
        cost["wall_seconds"]+=elapsed
        cost["forward_seconds"]+=elapsed
        cost["analytic_gflops"]+=4*nnz/1e9
        cost.update(segment_seconds=added,segment_calls=int(touched.sum()),sparse_calls=int((found&~flags).sum()))
        return np.column_stack((estimate,mark)),proba,touched,cost
    labels=(REFERENCE,selected)
    trials={name:[] for name in labels}
    result={}
    for name in labels:
        forward(name,text[:2048])
    for attempt in range(5):
        for name in labels if attempt%2==0 else reversed(labels):
            bits,p,active,cost=forward(name,text)
            if name==REFERENCE and not np.array_equal(bits,archive):
                raise ValueError("Live reference differs from the verified Round 24 archive")
            if np.any(bits[~active]!=archive[~active]):
                raise ValueError("The frozen correction changed any unrelated decision")
            if name in result and not np.array_equal(bits,result[name][0]):
                raise ValueError("Repeated decisions changed")
            result[name]=(bits,p,active)
            trials[name].append(cost)
            print(name,cost,flush=True)
    stats={}
    for name,(bits,p,active) in result.items():
        quality=f.metric(y,bits)
        cost={k:median(x[k] for x in trials[name]) for k in trials[name][0]}
        score=f.competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])
        old=archive[:,0].copy()
        fp=int((old[active]&(y[active,0]==0)).sum())
        if active.any():
            old[active],_=f.exact_budget(p[active],y[active,0],fp)
        file=OUT/f"{name}.csv"
        with file.open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(("row","label_vulnerability","label_mat","pred_vulnerability","pred_MAT","probability","matched"))
            writer.writerows((i,*yy,*vv,pp,tie) for i,(yy,vv,pp,tie) in enumerate(
                zip(y,bits.astype(int),p,old.astype(int),strict=True)))
        stats[name]=dict(score=score,quality=quality,cost=cost,trials=trials[name],
             frozen=f.comparison(y[:,0],archive[:,0],bits[:,0]),equal_fp=f.comparison(y[:,0],archive[:,0],old),
             predictions_sha256=checksum(file))
    write_json(OUT/"results.json",dict(selected=selected,results=stats,public_sha256=checksum(PUBLIC),
             policy_sha256=checksum(OUT/"policy.json"),scanner_sha256=checksum(PROGRAM)))
    print({k:v["score"] for k,v in stats.items()},flush=True)


if __name__=="__main__":
    run()
