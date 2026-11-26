"""Distill complementary full-function structural evidence into the compact encoder."""

import argparse

import numpy as np
from scipy.special import expit,logit

from structural_teacher import HERE,ROOT,OUT as TEACHER,checksum,inputs,json_at,positions,write_json
from view_train import fit,open_view,select_indices,small,tokenize
from view_eval import project,benchmark,pinned
from local_public import heldout,run
from experiment import comparison,strict_budget
from evaluate import counts,exact_budget
from pipeline import metric
from score_runtime import competition_formula


MODES=("full","structure","blend","complement")
SEEDS=(42,101)
PROJECT=HERE/"projects"
OFFICIAL=HERE/"official"


def local_logit(v):
    return logit(v.clip(1e-5,1-1e-5))


def targets(scope,labels,teacher):
    manifest=json_at(TEACHER/"fit.json")
    structural=TEACHER/"oof.npz"
    neural=ROOT/"round15/mining/scores.npz"
    if checksum(structural)!=manifest["oof_sha256"] or checksum(neural)!=json_at(neural.parent/"manifest.json")["sha256"]:
        raise ValueError("Out-of-fold teacher probabilities changed")
    with np.load(structural) as s,np.load(neural) as n:
        if not np.array_equal(s["ids"],n["ids"]) or not np.array_equal(s["y"],n["y"]) or not np.array_equal(s["ids"],inputs()[4]["fit"]):
            raise ValueError("Teachers disagree about the training rows")
        all_local=s["ids"]
        if scope=="projects":
            loc=np.arange(len(labels))
            cached=loc
        else:
            original,_=select_indices(False)
            _,loc,cached=np.intersect1d(original,all_local,return_indices=True)
        if not np.array_equal(labels[loc],s["y"][cached]):
            raise ValueError("Out-of-fold teacher labels do not match the current fit")
        q=s["combined"][cached]
        u=n["unix"][cached,0]
    gold=labels[loc,0]
    advantage=np.where(gold==1,q-u,u-q)
    amount=.75*np.clip((advantage-.12)/.6,0,1)
    base=teacher[loc,0]
    a,b=local_logit(base),local_logit(q)
    plans={"structure":q,"blend":expit(.55*a+.45*b),"complement":expit((1-amount)*a+amount*b)}
    override={}
    for mode,p in plans.items():
        adjusted=teacher.copy()
        adjusted[loc,0]=p
        override[mode]=adjusted
    details=dict(structural_sha256=manifest["oof_sha256"],neural_oof_sha256=checksum(neural),teacher_rows=len(loc),
        positive_advice=int(((amount>0)&(gold==1)).sum()),negative_advice=int(((amount>0)&(gold==0)).sum()),
        fraction_with_advice=float((amount>0).mean()),mean_advice_weight=float(amount.mean()))
    print("teacher targets",scope,details,flush=True)
    return override,details


def train_project(seed):
    fit("projects",seed,MODES,output_root=PROJECT,override=lambda y,t:targets("projects",y,t))


def evaluate_project(seed):
    project(seed,root=PROJECT,variants=MODES)


def fit_official():
    OFFICIAL.mkdir(exist_ok=True)
    records=[json_at(PROJECT/str(s)/"policy.json")["modes"] for s in SEEDS]
    gains={n:[r[n]["calibration"]["delta_tp"] for r in records] for n in MODES}
    chosen=max(MODES[1:],key=lambda n:sum(gains[n]))
    write_json(OFFICIAL/"transfer.json",dict(candidate=chosen,two_seed_dev_tp=gains,seeds=SEEDS))
    fit("official",42,("full",chosen),output_root=OFFICIAL,override=lambda y,t:targets("official",y,t))


def tune():
    texts,y,ids,cal,review=heldout(include_test=False)
    route,threshold=pinned()
    names=("reference","threshold","full",json_at(OFFICIAL/"transfer.json")["candidate"])
    tok=small.tokenizer()
    p,c=run(open_view("official",42,"reference",tok),texts,route,tokenize)
    stable=p[:,0]>threshold
    fp=counts(y[cal,0],stable[cal])["fp"]
    ref,_=exact_budget(p[cal,0],y[cal,0],fp)
    result={}
    for name in names:
        q,cost=(p,c) if name in names[:2] else run(open_view("official",42,name,tok,OFFICIAL),texts,route,tokenize)
        if not np.array_equal(q[:,1],p[:,1]):
            raise ValueError("Structural distillation altered the comment rule")
        _,cut=strict_budget(q[cal,0],y[cal,0],fp)
        cut=threshold if name=="reference" else cut
        before=q[cal]>(cut,.5)
        grade=competition_formula(metric(y[cal],before)["mean"]["f1"],0.,cost["analytic_gflops"]*30655/len(y))
        result[name]=dict(threshold=cut,projected_score=grade,calibration=comparison(y[cal,0],ref,before[:,0]),
                          review=comparison(y[review,0],stable[review],q[review,0]>cut),cost=cost)
        print(name,result[name],flush=True)
    selected=max(names,key=lambda n:(result[n]["projected_score"],n=="reference"))
    write_json(OFFICIAL/"policy.json",dict(modes=result,selected=selected,cal_rows=ids[cal].tolist(),
                 review_rows=ids[review].tolist(),fp=fp,route=route))


def public():
    rule=json_at(OFFICIAL/"policy.json")
    names=list(dict.fromkeys(("reference",rule["selected"])))
    tok=small.tokenizer()
    pipes={n:open_view("official",42,"reference" if n=="threshold" else n,tok,OFFICIAL) for n in names}
    benchmark(OFFICIAL,names,rule,lambda name,texts:run(pipes[name],texts,rule["route"],tokenize))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fit_project","eval_project","fit_official","tune","public"))
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.stage in {"fit_project","eval_project"}:
        {"fit_project":train_project,"eval_project":evaluate_project}[args.stage](args.seed)
    else:
        {"fit_official":fit_official,"tune":tune,"public":public}[args.stage]()
