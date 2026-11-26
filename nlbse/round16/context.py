"""Measure the compact encoder's sensitivity to the tokens dropped from the middle."""

import argparse
import csv
from collections import Counter

import numpy as np
from scipy.special import expit, logit

from experiment import HERE,ROOT,PROJECT,comparison,infer,inputs,json_at,open_model,small,strict_budget
from evaluate import raw, counts, exact_budget
from pipeline import PUBLIC,load,metric,write_json
from score_runtime import competition_formula


OUT=HERE/"context"
LIMIT=512
NAMES=("center","distributed","rare_tiles","full")


def tokenize(texts,tok):
    all_ids=[]
    for i in range(0,len(texts),1024):
        all_ids.extend(tok(texts[i:i+1024],truncation=False,verbose=False)["input_ids"])
    return all_ids


def layout(tokens,mode,idf):
    if len(tokens)<=LIMIT or mode=="full":
        return tokens
    if mode=="standard":
        return tokens[:384]+tokens[-128:]
    head,tail=tokens[:128],tokens[-96:]
    middle=tokens[128:-96]
    if mode=="center":
        at=(len(middle)-288)//2
        return head+middle[at:at+288]+tail
    # Preserve local order inside and between contiguous 16-token tiles.
    blocks=[middle[i:i+16] for i in range(0,len(middle),16)]
    k=288//16
    if mode=="rare_tiles":
        # Document-frequency weights learned on training inputs only.
        order=sorted(range(len(blocks)),key=lambda i:float(np.mean(idf[blocks[i]])),reverse=True)
        indices=sorted(order[:k])
    elif mode=="distributed":
        indices=np.linspace(0,len(blocks)-1,k,dtype=int)
    else:
        raise ValueError(f"Unknown token layout: {mode}")
    return head+[x for i in indices for x in blocks[i]]+tail


def fit_idf(tok):
    sentences,_,_,_,roles=inputs()
    df=Counter()
    for i in range(0,len(roles["fit"]),2048):
        examples=[sentences[row] for row in roles["fit"][i:i+2048]]
        for vec in tokenize(examples,tok):
            df.update(set(vec))
    idf=np.zeros(len(tok),dtype=np.float32)
    for token,freq in df.items():
        idf[token]=min(np.log((1+len(roles["fit"]))/(1+freq)),7.)
    return idf


def audit():
    texts,y,_=load(PUBLIC)
    lengths=np.asarray([len(v) for v in tokenize(texts,small.tokenizer())])
    path=ROOT/"round15/official_distill/original_teacher_public.csv"
    with path.open(newline="") as stream:
        saved=list(csv.DictReader(stream))
    if [int(r["row"]) for r in saved]!=list(range(len(y))) or not np.array_equal(y,np.asarray([[int(r[k]) for k in ("truth_vulnerability","truth_mat")] for r in saved])):
        raise ValueError("Round 15 labels differ in the context census")
    bits=np.asarray([[int(r["vulnerability"]),int(r["mat"])] for r in saved],dtype=bool)
    full=bits.copy()
    affected=lengths>LIMIT
    full[affected,0]=y[affected,0]
    cost=json_at(ROOT/"round15/official_distill/results.json")["original_teacher"]["cost"]
    result=dict(rows=len(y),affected=int(affected.sum()),longest=int(lengths.max()),quantiles=np.quantile(lengths,[.5,.9,.95,.99]).tolist(),
                cohort=counts(y[affected,0],bits[affected,0]),
                oracle_score=competition_formula(metric(y,full)["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]))
    OUT.mkdir(exist_ok=True)
    write_json(OUT/"audit.json",result)
    print(result,flush=True)


def get_scores(role,idf):
    texts,y,projects,_,parts=inputs()
    ids=parts[role]
    old=raw(role)["compact"]
    tok=small.tokenizer()
    full=tokenize([texts[i] for i in ids],tok)
    long=np.asarray([len(v)>LIMIT for v in full])
    subset=[vec for vec,active in zip(full,long,strict=True) if active]
    examples=[texts[idx] for idx in ids[long]]
    model=open_model("projects",PROJECT/"42","reference",tok)
    q={}
    costs={}
    for name in ("standard",*NAMES):
        probability,cost=infer(model,examples,[layout(v,name,idf) for v in subset])
        costs[name]=cost
        q[name]=probability[:,0]
        print(role,name,len(examples),cost,flush=True)
    np.savez_compressed(OUT/f"{role}.npz",ids=ids,length=np.asarray([len(x) for x in full]),y=y[ids,0],original=old,**q)
    return ids,y[ids,0],projects[ids],old,long,q,costs


def mix(base,changed,active,alpha):
    a=base.copy()
    g=lambda p:logit(p.clip(1e-6,1-1e-6))
    a[active]=expit((1-alpha)*g(a[active])+alpha*g(changed))
    return a


def project():
    OUT.mkdir(exist_ok=True)
    idf=fit_idf(small.tokenizer())
    np.save(OUT/"idf.npy",idf)
    threshold=json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    _,y,_,base,long,q,d=get_scores("dev",idf)
    fp=counts(y,base>threshold)["fp"]
    baseline,_=exact_budget(base,y,fp)
    recipes={}
    for name in ("standard",*NAMES):
        choices=[]
        for alpha in (.35,.65,1.):
            p=mix(base,q[name],long,alpha)
            bits,_=exact_budget(p,y,fp)
            _,cut=strict_budget(p,y,fp)
            choices.append(dict(alpha=alpha,threshold=cut,dev=comparison(y,baseline,bits)))
        recipes[name]=max(choices,key=lambda x:(x["dev"]["delta_tp"],x["alpha"]))
    strongest=max(NAMES,key=lambda n:(recipes[n]["dev"]["delta_tp"],-d[n]["analytic_gflops"]))
    write_json(OUT/"policy.json",dict(base_threshold=threshold,selected=strongest,recipes=recipes,
              selection="maximum additional true positives at matched development FP",dev_cost=d,dev_fp=fp))

    ids,y,projects,base,long,q,costs=get_scores("test",idf)
    raw_bits=base>threshold
    fp=counts(y,raw_bits)["fp"]
    equal,_=exact_budget(base,y,fp)
    records={}
    for name,r in recipes.items():
        p=mix(base,q[name],long,r["alpha"])
        bits,_=exact_budget(p,y,fp)
        frozen=p>r["threshold"]
        records[name]=dict(matched=comparison(y,equal,bits),frozen=comparison(y,raw_bits,frozen),cost=costs[name],
                           long_only=comparison(y[long],equal[long],bits[long]))
        with (OUT/f"{name}_test.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","label","base","score","over_length_cap","equal_before","equal_after","frozen_before","frozen_after"))
            w.writerows(zip(ids,projects,y,base,p,long.astype(int),equal.astype(int),bits.astype(int),
                            raw_bits.astype(int),frozen.astype(int),strict=True))
        print(name,records[name],flush=True)
    write_json(OUT/"results.json",dict(rows=len(y),affected=int(long.sum()),fp=fp,selected=strongest,modes=records))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("audit","project"))
    {"audit":audit,"project":project}[p.parse_args().stage]()
