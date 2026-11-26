"""Syntactic identifier-role links for short-function classification."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import re
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.sparse import hstack, load_npz, save_npz, vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from rotate import HERE, ROOT, PLAN, checksum, read_partition
sys.path.insert(0,str(ROOT/"round12/vendor"))
from tree_sitter import Language, Parser
import tree_sitter_cpp
from structure import lexical
from pipeline import TRAIN, PUBLIC, load, write_json

OUT=HERE/"features"
SOURCES=("token","grammar","bindings","interface")
MODELS=("token","grammar","bindings","grammar_bindings","interface","interface_bindings")
N_FEATURES=2**16
HASHER=FeatureHasher(n_features=N_FEATURES,input_type="dict",dtype=np.float32,alternate_sign=True)
LANGUAGE=Language(tree_sitter_cpp.language())
SUBWORDS=re.compile(rb"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+")
DECLARATIONS={"parameter_declaration","optional_parameter_declaration","declaration","init_declarator"}


def roles(text,parser):
    source=text.encode("utf-8")
    root=parser.parse(source).root_node
    names=defaultdict(list)
    syntax=Counter()
    interface=Counter()
    stack=[(root,(),False)]
    while stack:
        node,parent,decl=stack.pop()
        kind=node.type
        if kind=="comment":
            continue
        trace=(*parent[-2:],kind)
        syntax["/".join(trace)]+=1
        if kind=="identifier":
            value=source[node.start_byte:node.end_byte]
            names[value].append((node.start_byte,"/".join(parent[-2:]),decl))
        candidate=kind in {"type_identifier","field_identifier"} or (kind=="identifier" and
                   any(edge.startswith(("function_declarator.declarator","call_expression.function")) for edge in parent))
        if candidate:
            spelling=source[node.start_byte:node.end_byte]
            fragments=[piece.lower().decode("ascii") for piece in SUBWORDS.findall(spelling)]
            for piece in fragments:
                interface[f"{kind}/{piece}"]+=1
            if fragments:
                interface[f"{kind}/suffix/{fragments[-1]}"]+=1
        for index,child in enumerate(node.children):
            if not child.is_named:
                continue
            field=node.field_name_for_child(index) or "_"
            declares=(decl and field!="value") or (kind in DECLARATIONS and field=="declarator")
            edge=f"{kind}.{field}"
            stack.append((child,(*parent[-1:],edge),declares))
    traces=Counter()
    for items in names.values():
        use=sorted(items)
        introductions=[p for p,_,declaration in use if declaration]
        binding_type="local" if len(introductions)==1 else "external" if not introductions else "ambiguous"
        if len(use)==1:
            traces[f"single/{binding_type}/{use[0][1]}"]+=1
            continue
        traces[f"reuse/{binding_type}/{min(len(use),5)}"]+=1
        for left,right in zip(use,use[1:]):
            near="adjacent" if right[0]-left[0]<32 else "separated"
            traces[f"transition/{binding_type}/{near}/{left[1]}/{right[1]}"]+=1
        for position,context,_ in use:
            if introductions:
                before=position<introductions[0]
                traces[f"origin/{binding_type}/{int(before)}/{context}"]+=1
    syntax[f"parse_recovered/{int(root.has_error)}"]+=1
    return syntax,traces,interface


def transform_block(texts):
    parser=Parser(LANGUAGE)
    out=[[] for _ in SOURCES]
    for text in texts:
        _,token=lexical(text)
        syntax,binding,interface=roles(text,parser)
        for dest,entry in zip(out,(token,syntax,binding,interface),strict=True):
            dest.append(entry)
    return [HASHER.transform(x) for x in out]


def transform(texts,workers=1):
    chunks=[texts[i:i+512] for i in range(0,len(texts),512)]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            prepared=list(pool.map(transform_block,chunks))
    else:
        prepared=list(map(transform_block,chunks))
    result={}
    for index,name in enumerate(SOURCES):
        rows=vstack([x[index] for x in prepared],format="csr")
        rows.data[:]=np.sign(rows.data)*np.log1p(np.abs(rows.data))
        result[name]=normalize(rows,copy=False)
    return result


def chosen_inputs(features,name):
    if name.endswith("_bindings"):
        return hstack([features[name.removesuffix("_bindings")],features["bindings"]],format="csr")
    return features[name]


def cache(role,workers):
    dataset=PUBLIC if role=="public" else TRAIN
    texts,y,_=load(dataset)
    split=read_partition()
    ids=split[role] if role!="public" else np.arange(len(texts),dtype=np.int32)
    limit=json.loads(PLAN.read_text())["short_chars_max"]
    selected=np.asarray([i for i in ids if len(texts[i])<=limit],dtype=np.int32)
    if not len(selected):
        raise ValueError(f"No short functions in {role}")
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    feature=transform([texts[i] for i in selected],workers)
    for name,x in feature.items():
        save_npz(OUT/f"{role}_{name}.npz",x)
    np.savez_compressed(OUT/f"{role}.npz",ids=selected,y=y[selected])
    write_json(OUT/f"{role}.json",dict(rows=len(selected),positives=y[selected].sum(0).tolist(),
        wall_seconds=time.perf_counter()-start,length=limit,workers=workers,partition=checksum(HERE/"partition.npz")))
    print(role,len(selected),y[selected].sum(0),flush=True)


def fit_bundle(inputs,y):
    output={}
    for name in MODELS:
        x=chosen_inputs(inputs,name)
        clf=LogisticRegression(C=.7,solver="liblinear",max_iter=250,random_state=42)
        clf.fit(x,y)
        output[name]=clf
    return output


def fit():
    with np.load(OUT/"fit.npz") as x:
        y=x["y"][:,0]
    inputs={k:load_npz(OUT/f"fit_{k}.npz") for k in SOURCES}
    models=fit_bundle(inputs,y)
    dest=OUT/"models.joblib"
    joblib.dump(models,dest,compress=3)
    write_json(OUT/"models.json",dict(sha256=checksum(dest),rows=len(y),features=list(models),hash_bits=16))


def read_models():
    path=OUT/"models.joblib"
    if checksum(path)!=json.loads((OUT/"models.json").read_text())["sha256"]:
        raise ValueError("Feature heads were changed")
    return joblib.load(path)


def posterior(role,ids,y,models=None):
    if models is None:
        models=read_models()
    with np.load(OUT/f"{role}.npz") as x:
        short_ids=x["ids"]
        if not np.all(np.isin(short_ids,ids)) or not np.array_equal(x["y"],y[short_ids]):
            raise ValueError("Feature rows align with a different partition")
    matrices={k:load_npz(OUT/f"{role}_{k}.npz") for k in SOURCES}
    index=np.searchsorted(ids,short_ids)
    flags=np.zeros(len(ids),dtype=bool)
    flags[index]=True
    out={}
    for n,model in models.items():
        full=np.full(len(ids),.5,dtype=np.float64)
        full[index]=model.predict_proba(chosen_inputs(matrices,n))[:,1]
        out[n]=full
    return flags,out


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("cache","fit"))
    ap.add_argument("--role",choices=("fit","probe","dev","test","public"))
    ap.add_argument("--workers",type=int,default=3)
    args=ap.parse_args()
    if args.stage=="fit":
        fit()
    else:
        if args.role is None:
            ap.error("cache requires --role")
        cache(args.role,args.workers)
