"""Fit a text-only veto on high-confidence out-of-fold vulnerability decisions."""

import argparse
from collections import Counter
import csv
import json
import re
import time

import joblib
import numpy as np
from scipy.sparse import hstack
from scipy.special import expit, logit
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from crossfit import HERE, ROOT, checksum, write_json
from evaluate import inputs, raw, counts, exact_budget, grouped_delta
from mine import OUT as MINING
from normalize_code import IDENTIFIER, KEYWORDS, TOKEN
from precision import summary
from students import group_codes


OUT = HERE/"veto"
TOKEN_WORD = HashingVectorizer(n_features=2**16,token_pattern=r"\S+",lowercase=False,ngram_range=(1,3),dtype=np.float32)
SHAPES = FeatureHasher(n_features=2**14,input_type="dict",alternate_sign=False,dtype=np.float32)
SUBWORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+")
SCOPES = ("shape", "words", "both")


def describe(code):
    shapes=Counter()
    words,layout=[],[]
    for match in TOKEN.finditer(code):
        token=match[0]
        if match["comment"] is not None:
            shape="line_comment" if token.startswith("//") else "block_comment"
            words.append(shape)
        elif IDENTIFIER.fullmatch(token) and token not in KEYWORDS:
            shape="upper" if token.isupper() else "name"
            if token.startswith("__"):
                shapes["double_underscore"]+=1
            words.extend(piece.lower() for piece in SUBWORDS.findall(token))
        elif token.startswith(('"',"'",'R"')):
            shape="literal"
            words.append(shape)
        elif token[0].isdigit():
            shape="number"
            words.append(shape)
        else:
            shape=token
            words.append(token)
        layout.append(shape)
    for width in (2,4,7):
        for i in range(len(layout)-width+1):
            shapes["syntax/"+"/".join(layout[i:i+width])]+=1
    for line in code.splitlines():
        if line.strip():
            n=len(line)-len(line.lstrip())
            shapes[f"indent/{min(n,12)}"]+=1
    values={key:float(np.log1p(value)) for key,value in shapes.items()}
    return " ".join(words),values


def batch_features(texts):
    words, shapes = zip(*(describe(s) for s in texts),strict=True)
    return SHAPES.transform(shapes),TOKEN_WORD.transform(words)


def view(features,name):
    if name=="shape":
        return features[0]
    if name=="words":
        return features[1]
    return hstack(features,format="csr")


def veto(base,evidence,start,slope,anchor):
    p=logit(base.clip(1e-7,1-1e-7))
    correction=np.minimum(logit(evidence.clip(1e-7,1-1e-7))-logit(anchor),0.)
    return expit(p+np.where(base>start,slope*correction,0.))


def training(labels):
    data=MINING/"scores.npz"
    policy=json.loads((MINING/"manifest.json").read_text())
    if checksum(data)!=policy["sha256"]:
        raise ValueError("Out-of-fold evidence changed")
    with np.load(data) as f:
        rows=f["ids"]
        if not np.array_equal(labels[rows],f["y"]):
            raise ValueError("Wrong training evidence labels")
        pool=f["compact"][:,0]>.65
        hard=f["fp_90"]
    return rows[pool],hard[pool],policy["sha256"]


def tune():
    t=time.perf_counter()
    texts,labels,_,_,parts=inputs()
    ids,hard,digest=training(labels)
    fit=labels[ids,0]
    dev=parts["dev"]
    y=labels[dev,0]
    baselines=raw("dev")
    base=baselines["compact"]
    dev_hard=(y==0)&(base>.9)&(baselines["unix"]>.9)
    fit_x=batch_features([texts[i] for i in ids])
    dev_x=batch_features([texts[i] for i in dev])
    threshold=json.loads((ROOT/"round13/study/policy.json").read_text())["compact_threshold"]
    budget=counts(y,base>threshold)["fp"]
    reference,_=exact_budget(base,y,budget)
    models,recipes={},{}
    for name in SCOPES:
        for weight in (1.,4.):
            for reg in (.04,.20):
                mode=f"{name}/{weight:g}/{reg:g}"
                model=LogisticRegression(C=reg,max_iter=250,solver="liblinear",random_state=42)
                scale=np.where(hard,weight,1.)
                model.fit(view(fit_x,name),fit,sample_weight=scale)
                p=model.predict_proba(view(dev_x,name))[:,1]
                models[mode]=model
                candidates=[]
                for start in (.55,.75,.9):
                    for slope in (.25,.5,1.,1.5):
                        for anchor in (.65,.8,.9,.95):
                            s=veto(base,p,start,slope,anchor)
                            ix,cut=exact_budget(s,y,budget)
                            values=summary(y,reference,ix,dev_hard)
                            candidates.append(dict(mode=mode,view=name,start=start,slope=slope,anchor=anchor,threshold=cut,dev=values))
                recipes[mode]=max(candidates,key=lambda z:(z["dev"]["delta_tp"],z["dev"]["corrected_confident_fp"],-z["slope"]))
                print(mode,recipes[mode]["dev"]["delta_tp"],recipes[mode]["dev"]["corrected_confident_fp"],flush=True)
    compare=max((z for z in recipes.values() if z["mode"].split("/")[1]=="1"),key=lambda r:r["dev"]["delta_tp"])
    targeted=max((z for z in recipes.values() if z["mode"].split("/")[1]=="4"),key=lambda r:r["dev"]["delta_tp"])
    OUT.mkdir(exist_ok=True)
    dest=OUT/"weights.joblib"
    joblib.dump({name:models[r["mode"]] for name,r in (("control",compare),("weighted",targeted))},dest,compress=3)
    write_json(OUT/"policy.json",dict(control=compare,weighted=targeted,grid_best=recipes,
             base_threshold=threshold,fit_rows=len(fit),fit_fp90=int(hard.sum()),weights_sha256=checksum(dest),
             out_of_fold_sha256=digest,dev_budget=budget,seconds=time.perf_counter()-t))


def test():
    catalog=json.loads((OUT/"policy.json").read_text())
    weights=OUT/"weights.joblib"
    if checksum(weights)!=catalog["weights_sha256"]:
        raise ValueError("Veto model differs from the frozen calibration")
    models=joblib.load(weights)
    texts,labels,projects,paths,parts=inputs()
    ids=parts["test"]
    raw_scores=raw("test")
    p=raw_scores["compact"]
    gold=labels[ids,0]
    target=(gold==0)&(p>.9)&(raw_scores["unix"]>.9)
    original=p>catalog["base_threshold"]
    budget=counts(gold,original)["fp"]
    reference,_=exact_budget(p,gold,budget)
    groups=group_codes(zip(projects[ids],[paths[i] for i in ids],strict=True))
    data=batch_features([texts[i] for i in ids])
    out={}
    for name,model in models.items():
        r=catalog[name]
        vector=model.predict_proba(view(data,r["view"]))[:,1]
        score=veto(p,vector,r["start"],r["slope"],r["anchor"])
        equal,_=exact_budget(score,gold,budget)
        frozen=score>r["threshold"]
        out[name]=dict(equal=summary(gold,reference,equal,target),frozen=summary(gold,original,frozen,target),
                       fixed_interval=grouped_delta(gold,original,frozen,groups))
        with (OUT/f"{name}.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(("row","project","label","base_score","veto_score","new_score","base_equal","new_equal","base_frozen","new_frozen","shared_high_fp"))
            writer.writerows(zip(ids,projects[ids],gold,p,vector,score,reference.astype(int),equal.astype(int),
                                 original.astype(int),frozen.astype(int),target.astype(int),strict=True))
    write_json(OUT/"results.json",dict(test_rows=len(gold),common_confident_fp=int(target.sum()),modes=out))
    print(out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("tune","test"))
    tune() if parser.parse_args().stage=="tune" else test()
