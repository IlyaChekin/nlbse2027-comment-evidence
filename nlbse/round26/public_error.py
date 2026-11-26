"""Benchmark a single error-correction policy frozen on both project rotations and internal files."""
import csv
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np

from disagreement import HERE,ROOT,R24,R25,PARTS,f,checksum,json_at,write_json
from length_eval import REFERENCE
sys.path.insert(0,str(R25))
from section_runtime import compute
from error_runtime import correct
from internal_sparse import recipe
from union_scanner import scan,PROGRAM
from layout_official import recover,load,PUBLIC

OUT=HERE/"public"


def select():
    score={}
    for family in ("disagreement","ranking"):
        version=HERE/family
        a,b,c=[json_at(version/part/"results.json") for part in (*PARTS,"official")]
        for name,info in c["options"].items():
            gains=[x["options"][name]["matched"]["delta_tp"] for x in (a,b)]
            if min(gains)>=0 and max(gains)>0:
                score[name]=dict(**info,gains=gains,family=family,source_sha256=c["weights_sha256"])
    eligible={k:q for k,q in score.items() if q["proxy"]>0}
    best=max(eligible,key=lambda k:eligible[k]["proxy"],default=REFERENCE)
    report=dict(selected=best,supported=score)
    OUT.mkdir(exist_ok=True)
    write_json(OUT/"policy.json",report)
    print("selected",best,flush=True)
    return report


def main():
    policy=select()
    selected=policy["selected"]
    if selected==REFERENCE:
        return
    option=policy["supported"][selected]
    new=HERE/option["family"]/"official/models.joblib"
    if checksum(new)!=option["source_sha256"]:
        raise ValueError("Stored error-specific heads changed")
    extra=joblib.load(new)[option["mode"]]
    base_dir=R24/"official/tuned"
    first=json_at(base_dir/"policy.json")
    if checksum(base_dir/"models.joblib")!=first["models_sha256"]:
        raise ValueError("Round 24 ranker changed")
    package=joblib.load(base_dir/"models.joblib")["token"]
    base=first["candidates"][first["selected"]]
    section_dir=R25/"views"
    prev=json_at(section_dir/"public/policy.json")
    section=prev["support"][prev["selected"]]
    if checksum(section_dir/"official/calibrator.joblib")!=prev["calibrator_sha256"]:
        raise ValueError("Round 25 calibrator changed")
    section_head=joblib.load(section_dir/"official/calibrator.joblib")[section["region"]]
    bundle=json_at(section_dir/"public/results.json")
    archive_path=section_dir/"public"/f"{bundle['selected']}.csv"
    if checksum(archive_path)!=bundle["results"][bundle["selected"]]["predictions_sha256"]:
        raise ValueError("Round 25 verified predictions changed")
    texts,y,_=load(PUBLIC)
    with archive_path.open(newline="") as stream:
        archive=np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")] for r in csv.DictReader(stream)],
                           dtype=bool)
    ref,policy0,heads=recipe()
    model=f.pipe("official",f.small.tokenizer())
    def forward(name,text):
        p,aux,cost=f.infer_student(model,text,f.pinned()[0],columns=22,return_all=True)
        start=time.perf_counter()
        values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
        est=f.estimate(heads,values)
        score,no,yes,ask=f.choose(values,est,policy0,ref["gate"])
        flags=np.zeros(len(p),dtype=bool)
        found=np.zeros_like(flags)
        if ask.any():
            outcome=scan([s for s,k in zip(text,ask,strict=True) if k])
            found[ask]=outcome>=0
            flags[ask]=outcome==1
            score[found]=np.where(flags[found],yes[found],no[found])
        state=dict(p=score.astype(np.float32),known=found,flag=flags,w=est["w"],no=no,yes=yes)
        mt=recover(text,p[:,1]>.5)
        labels,q,active,info=correct(text,values,state,ref["threshold"],base,package,section,section_head,
                            None if name==REFERENCE else option,extra)
        end=time.perf_counter()
        cost["wall_seconds"]+=end-start
        cost["forward_seconds"]+=end-start
        cost["analytic_gflops"]+=(info["scoped_features"]+info["additional_features"])/1e9
        cost.update(info,extra_scope=int(active.sum()))
        return np.column_stack((labels,mt)),q,active,cost
    names=(REFERENCE,selected)
    curves={}
    trials={k:[] for k in names}
    for n in names:
        forward(n,texts[:2048])
    for batch in range(5):
        for n in names if batch%2==0 else reversed(names):
            bits,p,active,cost=forward(n,texts)
            if (n==REFERENCE and not np.array_equal(archive,bits)) or np.any(bits[~active]!=archive[~active]):
                raise ValueError("Baseline predictions or an unrelated row changed")
            if n in curves and not np.array_equal(bits,curves[n][0]):
                raise ValueError("Repeated output changed")
            trials[n].append(cost)
            curves[n]=bits,p,active
            print(n,cost,flush=True)
    result={}
    truth=y[:,0]
    for name,(bits,p,active) in curves.items():
        cost={k:median(x[k] for x in trials[name]) for k in trials[name][0]}
        quality=f.metric(y,bits)
        total=f.competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])
        matched=archive[:,0].copy()
        nfp=int((matched[active]&(truth[active]==0)).sum())
        if active.any():
            matched[active],_=f.exact_budget(p[active],truth[active],nfp)
        file=OUT/f"{name}.csv"
        with file.open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","label_vulnerability","label_mat","pred_vulnerability","pred_MAT","margin","equal_fp","scope"))
            w.writerows((i,*yy,*v,pps,dg,sc) for i,(yy,v,pps,dg,sc) in enumerate(
                            zip(y,bits.astype(int),p,matched.astype(int),active.astype(int),strict=True)))
        result[name]=dict(score=total,quality=quality,cost=cost,trials=trials[name],
                     frozen=f.comparison(truth,archive[:,0],bits[:,0]),matched=f.comparison(truth,archive[:,0],matched),
                     predictions_sha256=checksum(file))
    write_json(OUT/"results.json",dict(selected=selected,results=result,policy_sha256=checksum(OUT/"policy.json"),
                  public_sha256=checksum(PUBLIC),scanner_sha256=checksum(PROGRAM)))
    print({k:v["score"] for k,v in result.items()},flush=True)


if __name__=="__main__":
    main()
