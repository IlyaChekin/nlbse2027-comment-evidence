"""Test training-only neighbors in the compact encoder's own feature space."""

import csv
import time

import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from view_train import HERE,ROOT,batch,batch_plan,inputs,json_at,lexical_features,open_view,small,tokenize,write_json
from experiment import comparison,infer,strict_budget
from evaluate import counts,exact_budget,raw
from normalize_code import batch as canonicalize
from view_eval import pinned,selected_views


OUT=HERE/"latent_probe"


@torch.inference_mode()
def features(pipe,texts,produce_probability):
    model,tok=pipe.encoder,pipe.tok
    h=np.empty((len(texts),model.width*2),dtype=np.float16)
    p=np.empty(len(texts),dtype=np.float32) if produce_probability else None
    for start in range(0,len(texts),2048):
        chunk=texts[start:start+2048]
        seq=tokenize(chunk,tok)
        side=torch.as_tensor(lexical_features(chunk),device="cuda") if produce_probability else None
        with torch.autocast("cuda",dtype=torch.float16):
            for slots in batch_plan(list(range(len(chunk))),seq,small.BATCH):
                x,mask=batch(slots,seq,tok.pad_token_id)
                states=model.token_states(x,mask)
                avg=(states.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                mx=states.masked_fill(~mask.bool().unsqueeze(-1),-torch.inf).amax(1)
                target=start+np.asarray(slots)
                h[target]=torch.cat((avg,mx),dim=1).half().cpu().numpy()
                if p is not None:
                    p[target]=model.readout(states,mask,side[slots])[:,0].float().sigmoid().cpu().numpy()
    if not np.isfinite(h).all():
        raise ValueError("Nonfinite compact latent states")
    return h,p


def matrix(pca,h):
    q=np.vstack([pca.transform(h[i:i+4096].astype(np.float32)) for i in range(0,len(h),4096)])
    q/=np.linalg.norm(q,axis=1,keepdims=True).clip(1e-6)
    return torch.as_tensor(q,device="cuda",dtype=torch.float16)


@torch.inference_mode()
def nearest(query,bank,label):
    records=[]
    for at in range(0,len(query),256):
        similarity=query[at:at+256]@bank.T
        values,ranks=similarity.topk(24)
        close_label=label[ranks].float()
        pos=similarity.masked_fill(~label,-torch.inf).amax(1).float()
        neg=similarity.masked_fill(label,-torch.inf).amax(1).float()
        stats=torch.stack((pos,neg,pos-neg,close_label[:,:3].mean(1),close_label[:,:8].mean(1),
                           close_label.mean(1),values[:,0].float(),values[:,-1].float()),dim=1)
        records.append(stats.cpu().numpy())
    return np.vstack(records)


def role(name,pipe,texts,labels,parts,pca,bank,classes,route):
    ids=parts[name]
    local=[texts[i] for i in ids]
    h,p=features(pipe,local,True)
    q,_=infer(pipe,canonicalize(local),encoding=tokenize)
    base,active=selected_views(p,q[:,0],route)
    started=time.perf_counter()
    stats=nearest(matrix(pca,h),bank,classes)
    print(name,len(ids),"projection and exact search",round(time.perf_counter()-started,2),flush=True)
    np.savez_compressed(OUT/f"{name}.npz",ids=ids,y=labels[ids,0],base=base,neighbors=stats,routed=active)
    return ids,labels[ids,0],base,stats


def blend(base,evidence,features,policy):
    p=logit(base.clip(1e-6,1-1e-6))
    q=logit(evidence.clip(1e-6,1-1e-6))
    mask=(base>policy["lower"])&(base<policy["upper"])&(features[:,6]>=policy["similarity"])
    return np.where(mask,expit(p+policy["weight"]*(q-p)),base),mask


def main():
    OUT.mkdir(exist_ok=True)
    texts,y,projects,_,parts=inputs()
    tok=small.tokenizer()
    pipe=open_view("projects",42,"reference",tok)
    cfg, _=pinned()
    cutoff=json_at(ROOT/"round13/study/policy.json")["compact_threshold"]
    fit=parts["fit"]
    h,_=features(pipe,[texts[i] for i in fit],False)
    sample=np.random.default_rng(42).choice(len(fit),min(len(fit),30000),replace=False)
    pca=PCA(n_components=48,whiten=True,svd_solver="randomized",iterated_power=3,random_state=42)
    pca.fit(h[sample].astype(np.float32))
    bank=matrix(pca,h)
    classes=torch.as_tensor(y[fit,0]==1,device="cuda")
    del h
    _,g,p,x=role("probe",pipe,texts,y,parts,pca,bank,classes,cfg)
    vector=lambda p,x:np.column_stack((logit(p.clip(1e-6,1-1e-6)),x))
    head=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=250,random_state=42)).fit(vector(p,x),g)
    _,g,p,x=role("dev",pipe,texts,y,parts,pca,bank,classes,cfg)
    q=head.predict_proba(vector(p,x))[:,1]
    budget=counts(g,raw("dev")["compact"]>cutoff)["fp"]
    ref,_=exact_budget(p,g,budget)
    grid=[]
    for lo,hi in ((.0,1.),(.15,.9),(.35,.85),(.5,.85)):
        for sim in (.0,.5,.65,.8):
            for weight in (0.,.25,.5,.75,1.):
                spec=dict(lower=lo,upper=hi,similarity=sim,weight=weight)
                s,active=blend(p,q,x,spec)
                bits,t=strict_budget(s,g,budget)
                grid.append(dict(**spec,threshold=t,dev=comparison(g,ref,bits),routed=int(active.sum()) if weight else 0))
    choice=max(grid,key=lambda r:(r["dev"]["delta_tp"],-r["routed"]))
    _,original_cut=strict_budget(p,g,budget)
    joblib.dump(dict(pca=pca,head=head),OUT/"head.joblib",compress=3)
    write_json(OUT/"policy.json",dict(selected=choice,base_cut=original_cut,dev_fp=budget,fit_rows=len(fit),pca_fit=len(sample),grid=grid))

    ids,g,p,x=role("test",pipe,texts,y,parts,pca,bank,classes,cfg)
    q=head.predict_proba(vector(p,x))[:,1]
    s,active=blend(p,q,x,choice)
    budget=counts(g,raw("test")["compact"]>cutoff)["fp"]
    old,_=exact_budget(p,g,budget)
    after,_=exact_budget(s,g,budget)
    fixed=s>choice["threshold"]
    record=dict(rows=len(g),fp=budget,mode=choice,matched=comparison(g,old,after),
                frozen=comparison(g,p>original_cut,fixed),routed=int(active.sum()) if choice["weight"] else 0)
    with (OUT/"predictions.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("row","project","label","base","neighbor_score","score","before_equal","after_equal","before_frozen","after_frozen"))
        w.writerows(zip(ids,projects[ids],g,p,q,s,old.astype(int),after.astype(int),(p>original_cut).astype(int),fixed.astype(int),strict=True))
    write_json(OUT/"results.json",record)
    print(record,flush=True)


if __name__=="__main__":
    main()
