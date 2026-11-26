"""Freeze added-feature decisions on development projects and measure common errors."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from project_split import HERE, ROOT, checksum, family, read_partition
sys.path[:0]=[str(ROOT/"round11"),str(ROOT/"round10")]
from comment_head import collect as comment_features, matrix as comment_matrix, text_features
from mat_replay import predict as tag_rule
from students import group_codes, metric
from round2_report import bootstrap_delta
from pipeline import TRAIN, load, write_json
import structure

OUT=HERE/"heads"
ENCODERS=("compact","unix")
FEATURES=("logit","layout","ast_counts","token_motifs","ast_motifs","all")


@lru_cache(maxsize=1)
def corpus():
    texts,y,meta=load(TRAIN,audit=True)
    project=np.asarray([family(p) for p in meta["Projectname"]])
    return texts,y,project,meta["Filepath"],read_partition()


def primary(part,names=ENCODERS):
    _, all_y, _, _, split=corpus()
    ids=split[part]
    logits={}
    for name in names:
        path=HERE/"backbones"/name/f"{part}.npz"
        receipt=json.loads(path.with_suffix(".json").read_text())
        if checksum(path)!=receipt["sha256"]:
            raise ValueError("Prediction file changed")
        with np.load(path) as data:
            if not np.array_equal(data["ids"],ids) or not np.array_equal(data["y"],all_y[ids]):
                raise ValueError("Project-disjoint outputs are out of order")
            logits[name]=data["probabilities"]
    return logits


def structural(part):
    _, y, _, _, split=corpus()
    ids=split[part]
    with np.load(structure.OUT/f"{part}.npz") as data:
        if not np.array_equal(data["ids"],ids) or not np.array_equal(data["y"],y[ids]):
            raise ValueError("Structural features align with a different partition")
        return data["dense"], structure.probabilities(part)


def channels(p,dense,motifs,name):
    raw=logit(p.clip(1e-6,1-1e-6)).reshape(len(p),-1)
    cols=len(structure.LAYOUT_NAMES)
    if name=="logit":
        return raw
    if name=="layout":
        extra=dense[:,:cols]
    elif name=="ast_counts":
        extra=dense[:,cols:]
    elif name in ("token_motifs","ast_motifs"):
        k=0 if name=="token_motifs" else 1
        extra=logit(motifs[:,k:k+1].clip(1e-6,1-1e-6))
    else:
        extra=np.column_stack((dense,logit(motifs.clip(1e-6,1-1e-6))))
    return np.column_stack((raw,extra))


def binary_f1(y,bits):
    tp=np.count_nonzero(bits & (y==1))
    return float(2*tp/max(1,np.count_nonzero(bits)+np.count_nonzero(y)))


def calibration(y,p,projects):
    masks=[projects==name for name in np.unique(projects)]
    selected=None
    for t in np.r_[np.linspace(.035,.95,90),.5]:
        bits=p>t
        pooled=binary_f1(y,bits)
        by_group=[binary_f1(y[z],bits[z]) for z in masks]
        value=(pooled+sum(by_group)/len(by_group))/2
        candidate=(value,pooled,-abs(t-.5),float(t))
        if selected is None or candidate>selected:
            selected=candidate
    return dict(objective=selected[0],pooled_f1=selected[1],cutoff=selected[3])


def model_for(family):
    if family=="logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(C=.075,max_iter=500))
    return HistGradientBoostingClassifier(max_iter=125,max_leaf_nodes=13,min_samples_leaf=110,
                            learning_rate=.07,l2_regularization=1.,random_state=42)


def learn_mat(texts,y,dev_text,dev_y,projects):
    source,words=comment_features(texts)
    target,target_words=comment_features(dev_text)
    prepared=[]
    for name in ("comment","ngrams"):
        a=comment_matrix(source,text_features(words) if name=="ngrams" else None,name)
        b=comment_matrix(target,text_features(target_words) if name=="ngrams" else None,name)
        model=LogisticRegression(C=.3,solver="liblinear",max_iter=500,random_state=42)
        model.fit(a,y,sample_weight=np.where(y==1,3.,1.))
        p=model.predict_proba(b)[:,1]
        dense=calibration(dev_y,p,projects)
        prepared.append((dict(dense,name=name,calibration_ap=average_precision_score(dev_y,p)),model))
    plain=tag_rule(dev_text)
    fit_tags=tag_rule(texts)
    fit_untagged=(~fit_tags) & (source[:,-4]>0)
    dev_untagged=(~plain) & (target[:,-4]>0)
    if np.unique(y[fit_untagged]).size==2:
        a=comment_matrix(source,text_features(words),"ngrams")
        b=comment_matrix(target,text_features(target_words),"ngrams")
        tail=LogisticRegression(C=.3,solver="liblinear",max_iter=500,random_state=42)
        tail.fit(a[fit_untagged],y[fit_untagged],sample_weight=np.where(y[fit_untagged]==1,20.,1.))
        q=plain.astype(np.float32)
        q[dev_untagged]=tail.predict_proba(b[dev_untagged])[:,1]
        prepared.append((dict(calibration(dev_y,q,projects),name="untagged",calibration_ap=average_precision_score(dev_y,q)),tail))
    pooled=binary_f1(dev_y,plain)
    groups=[projects==name for name in np.unique(projects)]
    base=dict(name="tags",objective=(pooled+np.mean([binary_f1(dev_y[i],plain[i]) for i in groups]))/2,
              cutoff=.5,pooled_f1=pooled,calibration_ap=average_precision_score(dev_y,plain))
    prepared.append((base,None))
    return prepared


def tune():
    texts,y,groups,_,split=corpus()
    probe,dev=split["probe"],split["dev"]
    a,b=primary("probe"),primary("dev")
    dense_a,sparse_a=structural("probe")
    dense_b,sparse_b=structural("dev")
    xp,yp=y[probe,0],y[dev,0]
    models={}
    evidence={}
    for name in (*a,"joint"):
        pa=np.column_stack([a[k][:,0] for k in a]) if name=="joint" else a[name][:,0]
        pb=np.column_stack([b[k][:,0] for k in b]) if name=="joint" else b[name][:,0]
        candidates={}
        raw=pb if pb.ndim==1 else pb.mean(axis=1)
        candidates["raw"]=dict(calibration(yp,raw,groups[dev]),feature="raw",recipe="fixed",project_dev=True)
        for add in FEATURES:
            xa=channels(pa,dense_a,sparse_a,add)
            xb=channels(pb,dense_b,sparse_b,add)
            recipes=("logistic",) if add in ("logit","token_motifs","ast_motifs") else ("logistic","hgb")
            for recipe in recipes:
                model=model_for(recipe).fit(xa,xp)
                p=model.predict_proba(xb)[:,1]
                key=f"{add}_{recipe}"
                models[f"{name}/{key}"]=model
                candidates[key]=dict(calibration(yp,p,groups[dev]),feature=add,recipe=recipe,project_dev=True)
                print(name,key,candidates[key]["pooled_f1"],flush=True)
        winner=max(candidates,key=lambda k:candidates[k]["objective"])
        per_feature={key:max((k for k,v in candidates.items() if v["feature"]==key),
                        key=lambda k:candidates[k]["objective"]) for key in FEATURES}
        evidence[name]=dict(frozen=winner,candidates=candidates,representatives=per_feature)
    idx=split["fit"]
    dev_functions=[texts[i] for i in dev]
    mat=learn_mat([texts[i] for i in idx],y[idx,1],dev_functions,y[dev,1],groups[dev])
    selected=max(mat,key=lambda item:(item[0]["objective"],item[0]["calibration_ap"]))[0]
    for row,model in mat:
        if model is not None:
            models["mat/"+row["name"]]=model
    OUT.mkdir(parents=True,exist_ok=True)
    joblib.dump(models,OUT/"heads.joblib",compress=3)
    write_json(OUT/"policy.json",dict(choices=evidence,mat=selected,mat_controls=[r[0] for r in mat],
        checkpoint_sha256=checksum(OUT/"heads.joblib"),selection="mean of pooled vulnerability F1 and unweighted project F1 on disjoint development projects"))
    print("frozen",{k:v["frozen"] for k,v in evidence.items()},"MAT",selected,flush=True)


def mat_scores(texts,rule,models):
    if rule["name"]=="tags":
        return tag_rule(texts).astype(np.float32)
    x,words=comment_features(texts)
    name=rule["name"]
    if name=="untagged":
        tags=tag_rule(texts)
        chosen=(~tags)&(x[:,-4]>0)
        p=tags.astype(np.float32)
        if chosen.any():
            matrix=comment_matrix(x,text_features(words),"ngrams")
            p[chosen]=models["mat/untagged"].predict_proba(matrix[chosen])[:,1]
        return p
    matrix=comment_matrix(x,text_features(words) if name=="ngrams" else None,name)
    return models["mat/"+name].predict_proba(matrix)[:,1]


def error_cells(y,pred):
    return dict(false_negative=int(((y==1)&~pred).sum()),false_positive=int(((y==0)&pred).sum()))


def test():
    policy=json.loads((OUT/"policy.json").read_text())
    if checksum(OUT/"heads.joblib")!=policy["checkpoint_sha256"]:
        raise ValueError("Frozen project readout has changed")
    models=joblib.load(OUT/"heads.joblib")
    texts,labels,project,paths,split=corpus()
    ids=split["test"]
    y=labels[ids]
    group=project[ids]
    blocks=group_codes(zip(group,[paths[i] for i in ids],strict=True))
    thresholds=policy["mat"]
    start=time.perf_counter()
    mat=mat_scores([texts[i] for i in ids],thresholds,models)
    mat_s=time.perf_counter()-start
    mat_bits=mat>thresholds["cutoff"]
    dense,sparse=structural("test")
    encoders=primary("test")
    all_outputs={}
    for family in (*ENCODERS,"joint"):
        raw=encoders[family][:,0] if family!="joint" else np.column_stack([encoders[k][:,0] for k in ENCODERS])
        plan=policy["choices"][family]
        alternatives=dict.fromkeys(("raw",*plan["representatives"].values(),plan["frozen"]))
        for key in alternatives:
            spec=plan["candidates"][key]
            if key=="raw":
                p=raw if raw.ndim==1 else raw.mean(axis=1)
            else:
                p=models[f"{family}/{key}"].predict_proba(channels(raw,dense,sparse,spec["feature"]))[:,1]
            bits=np.column_stack((p>spec["cutoff"],mat_bits))
            all_outputs[(family,key)]=(p,bits)
    old={k: all_outputs[(k,"raw")][1] for k in ENCODERS}
    fn=(y[:,0]==1)&~old["unix"][:,0]&~old["compact"][:,0]
    fp=(y[:,0]==0)&old["unix"][:,0]&old["compact"][:,0]
    common=fn|fp
    results={}
    for (family,key),(p,bits) in all_outputs.items():
        base=all_outputs[(family,"raw")][1]
        details={name:metric(y[group==name],bits[group==name]) for name in np.unique(group)}
        rescued=np.count_nonzero(common&(bits[:,0]==y[:,0]))
        original_correct=base[:,0]==y[:,0]
        row=dict(family=family,features=key,scores=metric(y,bits),by_project=details,
                 rescued_common=int(rescued),harm_from_baseline=int((original_correct&(bits[:,0]!=y[:,0])).sum()),
                 changed=int((base[:,0]!=bits[:,0]).sum()))
        if key==policy["choices"][family]["frozen"] and key!="raw":
            row["interval_vs_raw"]=bootstrap_delta(y,base,bits,blocks)
        results[f"{family}/{key}"]=row
        filename=OUT/f"{family}_{key}_test.csv"
        with filename.open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(("row","project","p_vulnerability","p_MAT","pred_vulnerability","pred_MAT"))
            writer.writerows(zip(ids,group,p,mat,bits[:,0].astype(int),bits[:,1].astype(int),strict=True))
    audit={}
    layout_cols=len(structure.LAYOUT_NAMES)
    x=layout_cols+len(structure.AST_TYPES)
    cohorts={"first_quartile_length":dense[:,0]<=np.quantile(dense[:,0],.25),
             "last_quartile_length":dense[:,0]>=np.quantile(dense[:,0],.75),
             "parser_reports_error":dense[:,x+7]>0,"tree_without_error":dense[:,x+7]==0,
             **{p:group==p for p in np.unique(group)}}
    for key,mask in cohorts.items():
        audit[key]=dict(rows=int(mask.sum()),positives=int(y[mask,0].sum()),
                        common_false_negative=int((fn&mask).sum()),common_false_positive=int((fp&mask).sum()))
    test_text=[texts[i] for i in ids]
    old_mat=tag_rule(test_text)
    empty=np.asarray([not any(m["comment"] is not None for m in structure.TOKEN.finditer(t)) for t in test_text])
    mat_results={}
    for spec in policy["mat_controls"]:
        p=mat if spec["name"]==thresholds["name"] else mat_scores(test_text,spec,models)
        bits=p>spec["cutoff"]
        mat_results[spec["name"]]=dict(f1=binary_f1(y[:,1],bits),errors=error_cells(y[:,1],bits),
            tagged_baseline_differences=int((bits!=old_mat).sum()),
            rescued_uncaught=int((bits&~old_mat&(y[:,1]==1)).sum()),
            added_false_positive=int((bits&~old_mat&(y[:,1]==0)).sum()),
            by_project={name:binary_f1(y[group==name,1],bits[group==name]) for name in np.unique(group)})
    write_json(OUT/"analysis.json",dict(tests=results,common_errors=dict(false_negative=int(fn.sum()),false_positive=int(fp.sum())),
        strata=audit,project_holdout=dict(rows=len(ids),projects=sorted(set(group))),mat_seconds=mat_s,
        mat=dict(selected=thresholds["name"],controls=mat_results,common_errors=error_cells(y[:,1],mat_bits),
        positive_without_parsed_comment=int(((y[:,1]==1)&empty).sum()))))
    print("common",int(fn.sum()),int(fp.sum()),flush=True)
    for family in (*ENCODERS,"joint"):
        for key in ("raw",policy["choices"][family]["frozen"]):
            row=results[f"{family}/{key}"]
            print(family,key,row["scores"]["vulnerability"]["f1"],row["scores"]["mean"]["f1"],row["rescued_common"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","test"))
    args=ap.parse_args()
    tune() if args.stage=="tune" else test()
