"""Extract call-name fragments, with an ablation for their local syntactic roles."""
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
from scipy.sparse import hstack,load_npz,save_npz,vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from branches import BRANCH,CALL,EFFECT,EXIT,IDS,HASHER,LANGUAGE,Parser,branch_fields
from scope_eval import (HERE,ROOT,R22,REFERENCE,PARTS,f,channel,calibrate,baseline,blend,cohorts,
                        checksum,json_at,write_json)

OUTPUT=HERE/"call_roles"
PARTS_OF_NAME=re.compile(r"[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|[^a-zA-Z]|$)|[a-z]+")
WRAPPERS=frozenset(("field_expression","qualified_identifier","template_function","parenthesized_expression",
                    "pointer_expression"))
ROLE_TYPES=EFFECT|EXIT|frozenset(("argument_list","binary_expression","unary_expression"))


def leaf_name(node,raw):
    while node.type in WRAPPERS:
        inside=node.child_by_field_name("field") or node.child_by_field_name("name") or (
            node.named_children[-1] if node.named_child_count else None)
        if inside is None:
            return ()
        node=inside
    if node.type not in IDS:
        return ()
    name=raw[node.start_byte:node.end_byte].decode("ascii",errors="ignore")
    words=[w.lower() for w in PARTS_OF_NAME.findall(name) if len(w)>=2]
    return tuple(dict.fromkeys(words+[" ".join((a,b)) for a,b in zip(words,words[1:])]))


def extract(text,parser):
    raw=text.encode()
    root=parser.parse(raw).root_node
    bag,roles=Counter(),Counter()
    previous={}
    stack=[(root,0,"function","top")]
    while stack:
        node,block,path,context=stack.pop()
        kind=node.type
        if kind=="comment":
            continue
        if kind=="compound_statement":
            block=node.id
            path="statement"
        elif kind in ROLE_TYPES:
            path=kind
        if kind in CALL:
            fn=node.child_by_field_name("function")
            if fn is not None:
                names=leaf_name(fn,raw)
                for word in names:
                    bag["api:"+word]+=1
                    roles[f"{context}/{path}:{word}"]+=1
                prior=previous.get(block,())
                for a in prior[:2]:
                    for b in names[:2]:
                        roles[f"sequence:{a}/{b}"]+=1
                previous[block]=names
        entries={c.id:f"{kind}/{role}" for role,c in branch_fields(node)} if kind in BRANCH else {}
        for child in reversed(node.named_children):
            stack.append((child,block,path,entries.get(child.id,context)))
    return bag,roles


def block(texts):
    parser=Parser(LANGUAGE)
    bags,roles=zip(*(extract(s,parser) for s in texts),strict=True)
    return HASHER.transform(bags),HASHER.transform(roles)


def featurize(texts,workers):
    chunks=[texts[i:i+768] for i in range(0,len(texts),768)]
    if workers==1:
        pieces=list(map(block,chunks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pieces=list(pool.map(block,chunks,chunksize=2))
    arrays=[]
    for i in range(2):
        x=vstack([p[i] for p in pieces],format="csr")
        x.data[:]=np.sign(x.data)*np.log1p(np.abs(x.data))
        arrays.append(normalize(x,copy=False))
    return arrays


def stage():
    OUTPUT.mkdir(exist_ok=True)
    text,y,_,_,_=f.inputs()
    start=time.perf_counter()
    paths=[OUTPUT/f"{s}.npz" for s in ("name","context")]
    arrays=featurize(text,5)
    for p,x in zip(paths,arrays,strict=True):
        save_npz(p,x)
    write_json(OUTPUT/"manifest.json",dict(rows=len(text),seconds=time.perf_counter()-start,
        train_sha256=checksum(f.ROOT/"data/raw/train-00000-of-00001.parquet"),
        code_sha256=checksum(Path(__file__)),files={p.name:checksum(p) for p in paths}))
    print("prepared",len(text),round(time.perf_counter()-start,2),flush=True)


def run(rotation,arrays):
    dest=OUTPUT/rotation
    dest.mkdir(exist_ok=True)
    weak,y,cohort=cohorts()
    s=json_at(f.R19/"partition.json")["rotations"][rotation]
    k=f.R19/s["partition_file"]
    if checksum(k)!=s["partition_sha256"]:
        raise ValueError("Project partition changed")
    with np.load(k) as z:
        rows=z["fit"]
    training=rows[weak[rows]==0]
    data={}
    for part in ("dev","test"):
        with np.load(R22/rotation/f"{part}.npz") as z:
            data[part]=dict(z)
    dev,test=data["dev"],data["test"]
    axis=f.valid(dev["raw"])&~f.weak(dev["raw"])
    old=json_at(f.WEAK/rotation/"policy.json")
    fit=np.isin(dev["ids"],old["fit_rows"])
    cal=np.isin(dev["ids"],old["calibration_rows"])
    budget=json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    b_model,g,cut,control,_=calibrate(data,REFERENCE,fit,cal,budget)
    a=baseline(dev,b_model,g)
    b=baseline(test,b_model,g)
    ref=b["p"]>cut
    negative=f.valid(test["raw"])&~f.weak(test["raw"])
    evidence={}
    choices={}
    packages={}
    for name,x in arrays.items():
        model=LogisticRegression(C=2.,solver="liblinear",max_iter=100,random_state=42).fit(x[training],y[training])
        first,second=(model.predict_proba(x[dev["ids"]])[:,1],model.predict_proba(x[test["ids"]])[:,1])
        features=channel(dev[REFERENCE],first)
        c=LogisticRegression(C=.3,max_iter=350,random_state=42).fit(features[fit&axis],dev["y"][fit&axis])
        d=c.predict_proba(features)[:,1]
        t=c.predict_proba(channel(test[REFERENCE],second))[:,1]
        packages[name]=model,c
        ap=dict(local=float(average_precision_score(test["y"][negative],second[negative])),
                combined=float(average_precision_score(test["y"][negative],t[negative])))
        for where,mix in product(("observed","prior"),(.5,1.)):
            rr=blend(a,d,where,mix)
            hit,th=f.strict_budget(rr[cal],dev["y"][cal],budget)
            result=f.comparison(dev["y"][cal],control,hit)
            key=f"{name}_{where}_{mix:g}"
            choices[key]=dict(dev=result,threshold=th)
            pp=blend(b,t,where,mix)
            m,per=f.per_project(test["y"],test["project"],b["p"],pp,ref)
            frozen=f.comparison(test["y"],ref,pp>th)
            evidence[key]=dict(matched=m,per_project=per,frozen=frozen,ap=ap)
            with (dest/f"{key}_test.csv").open("w",newline="") as stream:
                w=csv.writer(stream)
                w.writerow(("row","project","gold","weak","control","p","fixed"))
                w.writerows(zip(test["ids"],test["project"],test["y"],f.weak(test["raw"]).astype(int),
                              b["p"],pp,(pp>th).astype(int),strict=True))
            print(rotation,key,m,result["delta_tp"],frozen["delta_tp"],frozen["delta_fp"],flush=True)
    selected=max(choices,key=lambda name: choices[name]["dev"]["delta_tp"])
    if choices[selected]["dev"]["delta_tp"]<=0:
        selected=REFERENCE
    joblib.dump(packages,dest/"models.joblib",compress=3)
    write_json(dest/"results.json",dict(selected=selected,cal=choices,test=evidence,fit_rows=len(training),
              models_sha256=checksum(dest/"models.joblib"),cohort_sha256=cohort,
              feature_sha256=checksum(OUTPUT/"manifest.json")))


if __name__=="__main__":
    stage()
    spec=json_at(OUTPUT/"manifest.json")
    for filename, sha in spec["files"].items():
        if checksum(OUTPUT/filename)!=sha:
            raise ValueError("Hashed call cache changed")
    name=load_npz(OUTPUT/"name.npz")
    context=load_npz(OUTPUT/"context.npz")
    for r in PARTS:
        run(r,dict(scoped_names=context,flat_names=name,
                   combined=hstack((name,context),format="csr")))
