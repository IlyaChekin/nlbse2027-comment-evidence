"""Test a readout of existing compact activations on the short confident-negative slice."""
import argparse
import csv
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rotate import HERE, checksum, load_backbones
import bindings
from evaluate import adjusted, cohort, counts, exact_budget, inputs, raw, summarize_row, OUT as STUDY
from pipeline import write_json

OUT=HERE/"activation"


@torch.inference_mode()
def vectors(functions):
    source=load_backbones()
    model,tok=source.frozen("compact")
    model.eval()
    buffer=[]
    handle=model.head.register_forward_pre_hook(lambda module,args: buffer.append(args[0].float().cpu().numpy()))
    try:
        seq=source.small.encode(functions,tok)
        side=torch.as_tensor(source.students.lexical_features(functions),device="cuda")
        result=np.empty((len(seq),2*model.width+5),dtype=np.float32)
        with torch.autocast("cuda",dtype=torch.float16):
            for rows in source.students.batch_plan(list(range(len(seq))),seq,source.small.BATCH):
                x,mask=source.students.batch(rows,seq,tok.pad_token_id)
                model(x,mask,side[rows])
                result[rows]=buffer.pop()
        return result
    finally:
        handle.remove()


def get(role):
    texts,y,_,_,roles=inputs()
    ids=roles[role]
    raw_p=raw(role)
    short,shape=bindings.posterior(role,ids,y)
    active=cohort("compact_low",short,raw_p)
    path=OUT/f"{role}.npz"
    if path.is_file():
        with np.load(path) as x:
            if not np.array_equal(x["ids"],ids[active]):
                raise ValueError("Activation cache IDs changed")
            z=x["embedding"]
    else:
        z=vectors([texts[i] for i in ids[active]])
        np.savez_compressed(path,ids=ids[active],embedding=z)
    return ids,y[ids,0],raw_p["compact"],active,z,shape


def train():
    OUT.mkdir(parents=True,exist_ok=True)
    _,y0,_,m0,x0,_=get("probe")
    _,y1,p1,m1,x1,_=get("dev")
    threshold=json.loads((STUDY/"policy.json").read_text())["compact_threshold"]
    base=p1>threshold
    fp=counts(y1,base)["fp"]
    rank,_=exact_budget(p1,y1,fp)
    records=[]
    models={}
    for n in (12,32):
        enc=PCA(n_components=n,whiten=True,random_state=42)
        model=make_pipeline(StandardScaler(),enc,LogisticRegression(C=.055,max_iter=400))
        model.fit(x0,y0[m0])
        score=model.predict_proba(x1)[:,1]
        models[n]=model
        for boost in (0.,.7,1.4,2.1,2.8,3.5):
            for strength in (.5,.75,1.):
                p=adjusted(p1,m1,score,strength,boost)
                matched,cutoff=exact_budget(p,y1,fp)
                records.append(dict(dimension=n,boost=boost,strength=strength,cutoff=cutoff,
                                    development=summarize_row(y1,rank,matched,m1)))
    selected=max(records,key=lambda r:(r["development"]["delta_tp"],r["development"]["repaired_in_target"],-r["boost"]))
    import joblib
    artifact=OUT/"head.joblib"
    joblib.dump(models[selected["dimension"]],artifact,compress=3)
    write_json(OUT/"policy.json",dict(selected=selected,recipe=records,base_threshold=threshold,sha256=checksum(artifact)))
    print(selected,flush=True)


def test():
    import joblib
    config=json.loads((OUT/"policy.json").read_text())
    path=OUT/"head.joblib"
    if checksum(path)!=config["sha256"]:
        raise ValueError("Activation readout changed after fit")
    model=joblib.load(path)
    ids,y,raw_p,active,x,_=get("test")
    project=inputs()[2][ids]
    posterior=raw("test")
    target=active&(posterior["unix"]<.05)
    recipe=config["selected"]
    p=adjusted(raw_p,active,model.predict_proba(x)[:,1],recipe["strength"],recipe["boost"])
    baseline=raw_p>config["base_threshold"]
    fp=counts(y,baseline)["fp"]
    rank,_=exact_budget(raw_p,y,fp)
    matched,_=exact_budget(p,y,fp)
    locked=p>recipe["cutoff"]
    report=dict(equal=summarize_row(y,rank,matched,target),frozen=summarize_row(y,baseline,locked,target),
                scope_rows=int(active.sum()))
    with (OUT/"predictions.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("row","project","y","base_score","candidate_score","base_pred","base_exact_fp","exact_fp","frozen"))
        w.writerows(zip(ids,project,y,raw_p,p,baseline.astype(int),rank.astype(int),matched.astype(int),
                        locked.astype(int),strict=True))
    write_json(OUT/"analysis.json",report)
    print(report,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","test"))
    a=ap.parse_args()
    train() if a.stage=="tune" else test()
