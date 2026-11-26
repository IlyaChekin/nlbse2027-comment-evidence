"""Send uncertain predictions from the 128-wide encoder to a stronger encoder."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

import small
import comment_head as mat
from students import (ROOT, PUBLIC, dataset, group_codes, load, load_selected, metric, predict_bits,
                      save, scores as teacher_scores, sha256, write_json)
from round2_report import bootstrap_delta
from score_runtime import competition_formula

def run_directory(width, continuation=None):
    name = "cascade" if width == 128 else f"cascade{width}"
    return small.HERE / (f"{name}_{continuation}" if continuation else name)
TEACHERS = ("t5_small", "unixcoder9")


def read(path):
    return json.loads(path.read_text())


def arrays(path, ids):
    with np.load(path) as d:
        if d["ids"].tolist()!=ids:
            raise ValueError(f"Unexpected rows in {path}")
        return {k:d[k] for k in d.files}


def gate_features(texts, probability):
    shape = np.asarray([(len(text), text.count("\n"), text.count("("), text.count(";"), text.count("{"))
                        for text in texts], dtype=np.float32)
    return np.column_stack((probability, probability*(1-probability), np.log1p(shape)))


def band(p, policy):
    return (p > policy["lower"]) & (p < policy["upper"])


def gate_rows(model, texts, p, policy):
    estimate = model.predict(gate_features(texts, p))
    unit = np.asarray([max(1,min(len(x),1910)) for x in texts], dtype=np.float32) / 512
    return estimate > policy["gain"] + policy["price"] * unit


def tune(width, continuation=None):
    out = run_directory(width, continuation)
    student_out = small.run_directory(width, continuation)
    _,_,texts,y,ids,_=dataset(full=True)
    selected=read(student_out/"selected.json")
    a=arrays(student_out/f"{Path(selected['checkpoint']).stem}_holdout.npz",ids)
    gold_mat=arrays(mat.OUT/"holdout.npz",ids)["p"]>read(mat.OUT/"selected.json")["chosen"]["threshold"]
    fit,cal,review=a["folds"]<=2,a["folds"]==3,a["folds"]==4
    p=a["p"][:,0]
    threshold=selected["thresholds"][0]
    width_proxy=np.asarray([max(1,min(len(x),1910)) for x in texts])
    student_cost = read(student_out/"public/analysis.json")["cost"]["analytic_gflops"]
    fits=[]
    for teacher in TEACHERS:
        folder=ROOT/"round10/runs"/teacher
        recipe=read(folder/"selected.json")
        stem=Path(recipe["checkpoint"]).stem
        ref=arrays(folder/f"{stem}_holdout.npz",ids)["p"][:,0]
        true_threshold=recipe["settings"]["thresholds"][0]
        reference=read(folder/"public/analysis.json")["cost"]
        def record(config, route):
            result = np.column_stack((np.where(route, ref>true_threshold, p>threshold), gold_mat))
            rho = width_proxy[cal & route].sum()/width_proxy[cal].sum()
            cost = student_cost + rho*reference["analytic_gflops"]
            evaluation = metric(y[cal],result[cal])
            objective = competition_formula(evaluation["mean"]["f1"],5+rho*reference["forward_seconds"],cost)
            fits.append(dict(config, fallback=teacher, proxy=objective,calibration=evaluation,
                             review=metric(y[review],result[review]), review_route_rate=float(route[review].mean()),
                             cost_proxy=float(cost)))

        for lower in (.0,.005,.015,.025,.04,.06,.10,.15,.22,.3,.38):
            for upper in (.60,.69,.76,.82,.88,.93,.97,.99,1.):
                rule = dict(kind="band", lower=lower, upper=upper)
                record(rule,band(p,rule))
        x = gate_features(texts,p)
        utility = ((ref>true_threshold)==y[:,0]).astype(np.float32) - ((p>threshold)==y[:,0])
        regressor = HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=11,learning_rate=.07,
                         min_samples_leaf=90,l2_regularization=.5,random_state=42).fit(x[fit],utility[fit])
        predicted = regressor.predict(x)
        out.mkdir(parents=True,exist_ok=True)
        gate = out/f"gate_{teacher}.joblib"
        joblib.dump(regressor,gate)
        for price in (0.,.015,.03,.05,.08):
            for gain in (-.02,0.,.025,.04,.06,.09,.12):
                rule = dict(kind="learned",price=price,gain=gain,gate=gate.name,sha256=sha256(gate))
                record(rule,predicted > gain + price * width_proxy/512)
    best=max(fits,key=lambda r:(r["proxy"],r["calibration"]["mean"]["f1"]))
    best_band=max((r for r in fits if r["kind"]=="band"),key=lambda r:r["proxy"])
    write_json(out/"policy.json",dict(selected=best,band=best_band,candidates=fits,student_sha256=selected["sha256"],
        mat_sha256=read(mat.OUT/"selected.json")["weights_sha256"]))
    print(best,flush=True)


class Pipeline:
    def __init__(self,mode,width,continuation=None,*,shifted=False):
        student_out = small.run_directory(width,continuation)
        self.mode=mode
        self.student=read(student_out/"selected.json")
        self.mat=read(mat.OUT/"selected.json")
        self.config=read(run_directory(width,continuation)/"policy.json")
        self.tok=small.tokenizer()
        self.encoder=small.Encoder(len(self.tok),width).cuda()
        if shifted:
            for layer in self.encoder.local:
                layer.use_shifted = True
        weight=student_out/self.student["checkpoint"]
        mat_weight=mat.OUT/"weights.npz"
        if sha256(weight)!=self.config["student_sha256"] or sha256(mat_weight)!=self.config["mat_sha256"]:
            raise ValueError("Calibration refers to different weights")
        self.encoder.load_state_dict(torch.load(weight,weights_only=True,map_location="cpu"))
        with np.load(mat_weight) as w:
            self.mat_weights,self.mat_bias=w["weight"],w["bias"]
        self.policy = self.config["band" if mode == "band" else "selected"]
        self.gate = None
        if mode != "small_lexical":
            self.fallback = self.policy["fallback"]
            self.teacher,self.teacher_tokenizer,self.teacher_thresholds=load_selected(self.fallback)
            if self.policy["kind"] == "learned":
                artifact = run_directory(width,continuation)/self.policy["gate"]
                if sha256(artifact) != self.policy["sha256"]:
                    raise ValueError("Routing model changed after calibration")
                self.gate=joblib.load(artifact)

    def run(self,texts):
        start=time.perf_counter()
        logits,_,cost=small.scores(self.encoder,self.tok,texts)
        cost["small_encoder_forward_seconds"] = cost["forward_seconds"]
        cost["mat_readout_seconds"] = 0.
        cutoff=self.student["thresholds"][0]
        pred=logits[:,0]>cutoff
        selected=np.zeros(len(texts),dtype=bool)
        start_gate=time.perf_counter()
        if self.mode != "small_lexical":
            selected=band(logits[:,0],self.policy) if self.gate is None else gate_rows(self.gate,texts,logits[:,0],self.policy)
        cost["forward_seconds"]+=time.perf_counter()-start_gate
        if selected.any():
            p,extra=teacher_scores(self.teacher,self.teacher_tokenizer,[t for t,s in zip(texts,selected,strict=True) if s],self.fallback)
            pred[selected]=p[:,0]>self.teacher_thresholds[0]
            for name in ("forward_seconds","analytic_gflops"):
                cost[name]+=extra[name]
        begin_mat=time.perf_counter()
        p,_,nnz=mat.predict(texts,self.mat["chosen"],self.mat_weights,self.mat_bias)
        cost["mat_readout_seconds"] = time.perf_counter()-begin_mat
        cost["forward_seconds"]+=cost["mat_readout_seconds"]
        cost["analytic_gflops"]+=2*nnz/1.e9
        cost.update(wall_seconds=time.perf_counter()-start,routed=int(np.count_nonzero(selected)))
        return np.column_stack((pred,p>self.mat["chosen"]["threshold"])).astype(np.int8),cost


def evaluate(width, continuation=None):
    out = run_directory(width,continuation)
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    base=predict_bits(ROOT/"round10/compact_exit/lexical_deferred/predictions.csv",ids,{"thresholds":(.5,.5)})
    results={}
    config=read(out/"policy.json")
    modes=("small_lexical","adaptive","band") if config["selected"]["kind"]=="learned" else ("small_lexical","adaptive")
    if continuation:
        modes = ("small_lexical",)
    for mode in modes:
        pipe=Pipeline(mode,width,continuation)
        pipe.run(texts[:2048])
        timings=[]
        expected=None
        for _ in range(3):
            bits,receipt=pipe.run(texts)
            if expected is not None and not np.array_equal(expected,bits):
                raise ValueError("The same inputs produced different labels")
            expected=bits
            timings.append(receipt)
            print(mode,receipt,flush=True)
        cost={k:median(r[k] for r in timings) for k in timings[0]}
        scores=metric(y,bits)
        results[mode]=dict(scores=scores,cost=cost,interval=bootstrap_delta(y,base,bits,groups),timings=timings,
                         score=competition_formula(scores["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
                         wall_charged_score=competition_formula(scores["mean"]["f1"],cost["wall_seconds"],cost["analytic_gflops"]))
        save(out/mode,y,ids,cal,bits,cost,(.5,.5))
    write_json(out/"analysis.json",results)


def predict(mode,source,target,width,continuation=None,*,shifted=False):
    with source.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if "Function" not in (reader.fieldnames or ()):
            raise ValueError("Input CSV requires a Function column")
        texts=[row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        raise ValueError("Every row must supply a function")
    values,_=Pipeline(mode,width,continuation,shifted=shifted).run(texts)
    with target.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("vulnerability","MAT"))
        writer.writerows(values.tolist())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("tune","public","predict"))
    parser.add_argument("--mode",default="small_lexical",choices=("small_lexical","adaptive","band"))
    parser.add_argument("--width",type=int,choices=(128,256),default=128)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--continuation",choices=("matched","gold_disagreement"))
    args=parser.parse_args()
    if args.stage=="tune":
        tune(args.width,args.continuation)
    elif args.stage=="public":
        evaluate(args.width,args.continuation)
    else:
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        predict(args.mode,args.input,args.output,args.width,args.continuation)
