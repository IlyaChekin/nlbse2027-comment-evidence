"""Measure short-function recovery at a fixed false-positive count and at frozen cutoffs."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from rotate import HERE, ROOT, checksum, family, read_partition
import bindings
from pipeline import TRAIN, load, write_json
from project_heads import calibration
from students import group_codes
from round2_report import bootstrap_delta

OUT=HERE/"study"
NAMES=("logits","token","grammar","bindings","grammar_bindings","token_bindings","interface","interface_bindings")
COHORTS=("joint_low","compact_low")


@lru_cache(maxsize=1)
def inputs():
    texts,y,fields=load(TRAIN,audit=True)
    projects=np.asarray([family(r) for r in fields["Projectname"]])
    return texts,y,projects,fields["Filepath"],read_partition()


def raw(role):
    _,y,_,_,roles=inputs()
    ids=roles[role]
    result={}
    for name in ("compact","unix"):
        p=HERE/"backbones"/name/f"{role}.npz"
        if checksum(p)!=json.loads(p.with_suffix(".json").read_text())["sha256"]:
            raise ValueError("Base model predictions changed")
        with np.load(p) as x:
            if not np.array_equal(x["ids"],ids) or not np.array_equal(x["y"],y[ids]):
                raise ValueError("Scores and test examples are misaligned")
            result[name]=x["probabilities"][:,0]
    return result


def cohort(k,small,p):
    chosen=small&(p["compact"]<.05)
    if k=="joint_low":
        chosen&=p["unix"]<.05
    return chosen


def matrix(name,p,feature,key):
    names=[] if name=="logits" else ["token","bindings"] if name=="token_bindings" else [name]
    arrays=[p["compact"], *(p[n] for n in ("unix",) if key=="joint_low"), *(feature[n] for n in names)]
    return logit(np.column_stack(arrays).clip(1e-6,1-1e-6))


def adjusted(p,active,q,strength,boost=0.):
    result=p.copy().astype(np.float64)
    shifted=expit(logit(q.clip(1e-6,1-1e-6))+boost)
    result[active]=np.maximum(p[active],(1-strength)*p[active]+strength*shifted)
    return result


def exact_budget(scores,y,budget):
    """Freeze a rank budget for evaluation. Equal scores use stable input order."""
    index=np.argsort(-scores,kind="stable")
    negatives=np.flatnonzero(y[index]==0)
    finish=negatives[budget] if budget<len(negatives) else len(index)
    bits=np.zeros(len(scores),dtype=bool)
    bits[index[:finish]]=True
    cutoff=float(scores[index[finish]]) if finish<len(index) else -1.
    if int((bits&(y==0)).sum())!=budget:
        raise ValueError("False-positive budget calculation failed")
    return bits,cutoff


def counts(y,p):
    return dict(tp=int(((y==1)&p).sum()),fp=int(((y==0)&p).sum()),
                fn=int(((y==1)&~p).sum()),tn=int(((y==0)&~p).sum()))


def grouped_delta(y,old,new,files):
    pad=np.zeros(len(y),dtype=np.int8)
    return bootstrap_delta(np.column_stack((y,pad)),np.column_stack((old,pad)),
                           np.column_stack((new,pad)),files)["vulnerability"]


def floor_f1(y,p):
    tp=np.count_nonzero((y==1)&p)
    return 2*tp/max(int(np.count_nonzero(y))+int(np.count_nonzero(p)),1)


def summarize_row(y,orig,new,target):
    first,second=counts(y,orig),counts(y,new)
    return dict(baseline=first,candidate=second,baseline_f1=floor_f1(y,orig),candidate_f1=floor_f1(y,new),
                delta_tp=second["tp"]-first["tp"],delta_fp=second["fp"]-first["fp"],
                repaired_in_target=int((target&(y==1)&~orig&new).sum()),
                repaired_all=int(((y==1)&~orig&new).sum()),lost_all=int(((y==1)&orig&~new).sum()),
                new_false_positives_in_target=int((target&(y==0)&~orig&new).sum()))


def tune():
    _,y,projects,_,roles=inputs()
    probe,dev=roles["probe"],roles["dev"]
    q0,q1=(raw(role) for role in ("probe","dev"))
    short0,features0=bindings.posterior("probe",probe,y)
    short1,features1=bindings.posterior("dev",dev,y)
    vote=calibration(y[dev,0],q1["compact"],projects[dev])
    base=q1["compact"]>vote["cutoff"]
    budget=counts(y[dev,0],base)["fp"]
    base_rank,_=exact_budget(q1["compact"],y[dev,0],budget)
    models={}
    rules={}
    for key in COHORTS:
        a=cohort(key,short0,q0)
        b=cohort(key,short1,q1)
        rows={}
        for method in NAMES:
            x0=matrix(method,q0,features0,key)
            x1=matrix(method,q1,features1,key)
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.25,max_iter=250))
            clf.fit(x0[a],y[probe[a],0])
            models[f"{key}/{method}"]=clf
            p=clf.predict_proba(x1[b])[:,1]
            configurations=[]
            for s in (.25,.5,.75,1.):
                for boost in (0.,.7,1.4,2.1,2.8,3.5):
                    score=adjusted(q1["compact"],b,p,s,boost)
                    evaluate,cutoff=exact_budget(score,y[dev,0],budget)
                    row=summarize_row(y[dev,0],base_rank,evaluate,b)
                    configurations.append(dict(strength=s,boost=boost,cutoff=cutoff,dev_equal_fp=row))
            chosen=max(configurations,key=lambda r:(r["dev_equal_fp"]["delta_tp"],
                                              r["dev_equal_fp"]["repaired_in_target"],-r["boost"],-r["strength"]))
            rows[method]=dict(selected=chosen,grid=configurations)
            print(key,method,chosen,flush=True)
        best=max(rows,key=lambda m:(rows[m]["selected"]["dev_equal_fp"]["delta_tp"],
                            rows[m]["selected"]["dev_equal_fp"]["repaired_in_target"]))
        rules[key]=dict(frozen=best,candidates=rows,probe_rows=int(a.sum()),dev_rows=int(b.sum()))
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/"model.joblib"
    joblib.dump(models,dest)
    write_json(OUT/"policy.json",dict(compact_threshold=vote["cutoff"],development_fp_limit=budget,
                      models_sha256=checksum(dest),variants=rules,baseline_dev=counts(y[dev,0],base)))


def test():
    _,labels,projects,paths,roles=inputs()
    ids=roles["test"]
    y=labels[ids,0]
    config=json.loads((OUT/"policy.json").read_text())
    path=OUT/"model.joblib"
    if checksum(path)!=config["models_sha256"]:
        raise ValueError("Frozen readout has changed")
    models=joblib.load(path)
    posterior=raw("test")
    short,shape=bindings.posterior("test",ids,labels)
    old=posterior["compact"]>config["compact_threshold"]
    budget=counts(y,old)["fp"]
    old_rank,_=exact_budget(posterior["compact"],y,budget)
    cluster=group_codes(zip(projects[ids],[paths[i] for i in ids],strict=True))
    target=cohort("joint_low",short,posterior)
    group=projects[ids]
    report={}
    for name,settings in config["variants"].items():
        active=cohort(name,short,posterior)
        for feature,recipe in settings["candidates"].items():
            rule=recipe["selected"]
            x=matrix(feature,posterior,shape,name)
            q=models[f"{name}/{feature}"].predict_proba(x[active])[:,1]
            score=adjusted(posterior["compact"],active,q,rule["strength"],rule["boost"])
            equal,_=exact_budget(score,y,budget)
            locked=score>rule["cutoff"]
            partition={}
            for project in np.unique(group):
                there=group==project
                baseline=old[there]
                b=counts(y[there],baseline)["fp"]
                matched,_=exact_budget(score[there],y[there],b)
                old_local,_=exact_budget(posterior["compact"][there],y[there],b)
                partition[project]=dict(pooled_budget=summarize_row(y[there],old_rank[there],equal[there],target[there]),
                    project_budget=summarize_row(y[there],old_local,matched,target[there]))
            short_budget=counts(y[short],old[short])["fp"]
            short_base,_=exact_budget(posterior["compact"][short],y[short],short_budget)
            short_matched,_=exact_budget(score[short],y[short],short_budget)
            strict,_=exact_budget(score[target],y[target],0)
            record=dict(test_fp_budget=summarize_row(y,old_rank,equal,target), frozen_dev=summarize_row(y,old,locked,target),
                         by_project=partition,scope_rows=int(active.sum()),rule=rule,
                         equal_budget_short=summarize_row(y[short],short_base,short_matched,target[short]),
                         cohort_zero_fp_count=counts(y[target],strict)["tp"],
                         frozen_f1_interval=grouped_delta(y,old,locked,cluster) if feature==settings["frozen"] else None)
            report[f"{name}/{feature}"]=record
            with (OUT/f"{name}_{feature}.csv").open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(("row","project","label","short","joint_low","base_score","candidate_score","base_pred","base_exact_fp","pred_dev_cut","pred_exact_fp"))
                w.writerows(zip(ids,group,y,short.astype(int),target.astype(int),posterior["compact"],score,
                                   old.astype(int),old_rank.astype(int),locked.astype(int),equal.astype(int),strict=True))
    summary=dict(results=report,baseline_counts=counts(y,old),baseline_rank_counts=counts(y,old_rank),rows=len(ids),short_count=int(short.sum()),
                   target_rows=int(target.sum()),target_misses=int((target&(y==1)&~old).sum()),
                   selections={name:cfg["frozen"] for name,cfg in config["variants"].items()})
    write_json(OUT/"analysis.json",summary)
    print("test",summary["rows"],summary["short_count"],summary["target_misses"],summary["baseline_counts"],flush=True)
    for name in sorted(report):
        absolute=report[name]["test_fp_budget"]
        locked=report[name]["frozen_dev"]
        print(name,absolute["repaired_in_target"],absolute["delta_tp"],absolute["delta_fp"],locked["delta_tp"],locked["delta_fp"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","test"))
    args=ap.parse_args()
    tune() if args.stage=="tune" else test()
