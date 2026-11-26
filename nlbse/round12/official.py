"""Confirm heldout structural features on the official split and measure their real cost."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np
from scipy.sparse import load_npz, vstack
from sklearn.linear_model import SGDClassifier

from project_split import HERE, ROOT, checksum, read_partition
sys.path[:0]=[str(ROOT/"round11"),str(ROOT/"round10")]
import cascade
import comment_head as mat
import small
import structure
from project_heads import channels, model_for
from students import group_codes, metric, predict_bits, save
from pipeline import TRAIN, PUBLIC, load, select_indices, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT=HERE/"official"
BASE="compact"


def read(path):
    return json.loads(path.read_text())


def eligible():
    transfer=read(HERE/"heads/analysis.json")["tests"]
    policy=read(HERE/"heads/policy.json")["choices"][BASE]
    primary=transfer[f"{BASE}/raw"]["scores"]["vulnerability"]["f1"]
    result=[]
    for name in ("token_motifs","ast_motifs","ast_counts","layout","all"):
        key=policy["representatives"][name]
        row=transfer[f"{BASE}/{key}"]["scores"]["vulnerability"]["f1"]
        # Positive held-out changes qualify for an independent public confirmation.
        if row>primary+.001:
            result.append((name,policy["candidates"][key]["recipe"]))
    return result


def existing_inputs():
    entries=read_partition()
    piles={}
    ids=[]
    features=[]
    matrices={"token":[],"ast":[]}
    for key in entries:
        with np.load(structure.OUT/f"{key}.npz") as z:
            if not np.array_equal(z["ids"],entries[key]):
                raise ValueError("Structure cache alignment differs")
            ids.extend(z["ids"].tolist())
            features.append(z["dense"])
        for name in matrices:
            matrices[name].append(load_npz(structure.OUT/f"{key}_{name}.npz"))
    texts, y, _=load(TRAIN)
    missing=sorted(set(range(len(texts)))-set(ids))
    if missing:
        dense,token,ast=structure.assemble([texts[i] for i in missing])
        ids.extend(missing)
        features.append(dense)
        matrices["token"].append(token)
        matrices["ast"].append(ast)
    ordered=np.argsort(ids)
    if not np.array_equal(np.asarray(ids)[ordered],np.arange(len(y))):
        raise ValueError("Structural coverage does not match official train")
    for name, rows in matrices.items():
        piles[name]=vstack(rows,format="csr")[ordered]
    return y,np.concatenate(features)[ordered],piles


def base_pipe():
    return cascade.Pipeline("small_lexical",256,"gold_disagreement",shifted=True)


def tune():
    variants=eligible()
    OUT.mkdir(parents=True,exist_ok=True)
    if not variants:
        write_json(OUT/"policy.json",dict(selected=None,eligible=variants,
                  reason="No registered structural feature improved heldout pooled F1 by 0.001"))
        return
    y,dense,matrices=existing_inputs()
    fit,internal=select_indices(False)
    with np.load(ROOT/"round8/frozen/holdout.npz") as f:
        if internal!=f["rows"].tolist():
            raise ValueError("Frozen official validation changed")
        fold=f["folds"]
    learning,cal,review=fold<=2,fold==3,fold==4
    base=ROOT/"round11/small256_gold_disagreement"
    initial=read(base/"selected.json")
    with np.load(base/f"{Path(initial['checkpoint']).stem}_holdout.npz") as a:
        if a["ids"].tolist()!=internal or not np.array_equal(a["y"],y[internal]):
            raise ValueError("Compact validation order changed")
        raw=a["p"][:,0]
    with np.load(ROOT/"round11/mat/holdout.npz") as a:
        if a["ids"].tolist()!=internal:
            raise ValueError("MAT validation order changed")
        mat_bits=a["p"]>read(ROOT/"round11/mat/selected.json")["chosen"]["threshold"]
    candidates=[]
    learned={}
    for r,matrix in matrices.items():
        learned[r]=SGDClassifier(loss="log_loss",alpha=3e-6,max_iter=40,random_state=42).fit(matrix[fit],y[fit,0])
    motif=np.column_stack([learned[k].predict_proba(matrices[k][internal])[:,1] for k in ("token","ast")])
    y=y[internal]
    original=float(initial["thresholds"][0])
    scores_old=metric(y[cal],np.column_stack((raw[cal]>original,mat_bits[cal])))
    for name,recipe in variants:
        x=channels(raw,dense[internal],motif,name)
        model=model_for(recipe).fit(x[learning],y[learning,0])
        learned[name]=model
        q=model.predict_proba(x)[:,1]
        for margin in (.05,.10,.2,.35,1.0):
            active=np.abs(raw-original)<margin
            overhead=10*active[cal].mean()
            for cutoff in np.linspace(.1,.9,41):
                pred=np.column_stack((np.where(active,q>cutoff,raw>original),mat_bits))
                result=metric(y[cal],pred[cal])["mean"]["f1"]
                # Cost proxy calibrates the route using placeholders only. Runtime is benchmarked later.
                candidates.append(dict(feature=name,recipe=recipe,margin=margin,cutoff=float(cutoff),
                    calibration_f1=result,projected_cost_objective=.6*result-.0002*overhead,
                    fraction=float(active[cal].mean()),review=metric(y[review],pred[review])))
    selected=max(candidates,key=lambda r:r["projected_cost_objective"])
    model_file=OUT/"weights.joblib"
    joblib.dump(learned,model_file,compress=3)
    write_json(OUT/"policy.json",dict(selected=selected,eligible=variants,all_candidates=candidates,
        sha256=checksum(model_file),base_cutoff=original,baseline_calibration=scores_old))
    print(selected,flush=True)


def cheap_features(texts,name,weights):
    from sklearn.preprocessing import normalize

    layout_columns=len(structure.LAYOUT_NAMES)
    dense=np.zeros((len(texts),layout_columns+len(structure.AST_NAMES)),dtype=np.float32)
    motif=np.full((len(texts),2),.5,dtype=np.float64)
    if name=="layout":
        vals=[structure.lexical(text,with_motifs=False)[0] for text in texts]
        dense[:,:layout_columns]=np.log1p(vals)
    elif name=="ast_counts":
        parser=structure.Parser(structure.Language(structure.tree_sitter_cpp.language()))
        vals=[structure.syntax(text,parser,with_motifs=False)[0] for text in texts]
        dense[:,layout_columns:]=np.log1p(vals)
    elif name in {"token_motifs","ast_motifs"}:
        if name=="token_motifs":
            scores=(structure.lexical(text)[1] for text in texts)
            column,role=0,"token"
        else:
            parser=structure.Parser(structure.Language(structure.tree_sitter_cpp.language()))
            scores=(structure.syntax(text,parser)[1] for text in texts)
            column,role=1,"ast"
        x=structure.HASHER.transform(scores)
        x.data[:]=np.sign(x.data)*np.log1p(np.abs(x.data))
        motif[:,column]=weights[role].predict_proba(normalize(x,copy=False))[:,1]
    else:
        dense,lex,ast=structure.assemble(texts,workers=1)
        motif=np.column_stack((weights["token"].predict_proba(lex)[:,1],weights["ast"].predict_proba(ast)[:,1]))
    return dense,motif


def run(pipe,texts,config,weights):
    start=time.perf_counter()
    raw,_,cost=small.scores(pipe.encoder,pipe.tok,texts)
    raw=raw[:,0]
    base=pipe.student["thresholds"][0]
    v=raw>base
    active=np.zeros(len(texts),dtype=bool)
    feature_seconds=0.
    if config is not None:
        active=np.abs(raw-base)<config["margin"]
        selected=[text for text,chosen in zip(texts,active,strict=True) if chosen]
        if selected:
            began=time.perf_counter()
            dense,motif=cheap_features(selected,config["feature"],weights)
            x=channels(raw[active],dense,motif,config["feature"])
            p=weights[config["feature"]].predict_proba(x)[:,1]
            v[active]=p>config["cutoff"]
            feature_seconds=time.perf_counter()-began
    t=time.perf_counter()
    mat_p,_,nnz=mat.predict(texts,pipe.mat["chosen"],pipe.mat_weights,pipe.mat_bias)
    cost["forward_seconds"]+=feature_seconds+(time.perf_counter()-t)
    cost["analytic_gflops"]+=2*nnz/1e9
    cost.update(wall_seconds=time.perf_counter()-start,structural_seconds=feature_seconds,processed=int(active.sum()))
    return np.column_stack((v,mat_p>pipe.mat["chosen"]["threshold"])),cost


def public():
    settings=read(OUT/"policy.json")
    selected=settings["selected"]
    if selected is None:
        print(settings["reason"],flush=True)
        return
    filename=OUT/"weights.joblib"
    if checksum(filename)!=settings["sha256"]:
        raise ValueError("Official readout was changed after calibration")
    weights=joblib.load(filename)
    pipe=base_pipe()
    texts,y,meta=load(PUBLIC,audit=True)
    ids=list(range(len(y)))
    groups=group_codes(zip(meta["Projectname"],meta["Filepath"],strict=True))
    base=predict_bits(ROOT/"round11/cold_shift/predictions.csv",ids,{"thresholds":(.5,.5)})
    p={}; timings={"comparator":[],"structure":[]}
    for name,cfg in (("comparator",None),("structure",selected)):
        run(pipe,texts[:2048],cfg,weights)
    for i in range(3):
        for name in timings if i%2==0 else reversed(timings):
            cfg=None if name=="comparator" else selected
            bits,receipt=run(pipe,texts,cfg,weights)
            if name=="comparator" and not np.array_equal(bits,base):
                raise ValueError("Round 11 reference changed")
            if name in p and not np.array_equal(p[name],bits):
                raise ValueError("An official prediction changed on repeat")
            p[name]=bits
            timings[name].append(receipt)
            print(name,receipt,flush=True)
    output={}
    for name,bits in p.items():
        c={k:median(r[k] for r in timings[name]) for k in timings[name][0]}
        scores=metric(y,bits)
        output[name]=dict(metrics=scores,cost=c,trials=timings[name],interval_vs_existing=bootstrap_delta(y,base,bits,groups),
                          local_formula=competition_formula(scores["mean"]["f1"],c["forward_seconds"],c["analytic_gflops"]),
                          wall_charged=competition_formula(scores["mean"]["f1"],c["wall_seconds"],c["analytic_gflops"]))
        save(OUT/name,y,ids,np.zeros(len(y),dtype=bool),bits,c,(.5,.5))
    write_json(OUT/"analysis.json",output)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","public"))
    args=ap.parse_args()
    tune() if args.stage=="tune" else public()
