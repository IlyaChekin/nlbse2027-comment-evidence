"""Distinguish which exact rules fired once the existing gate requested a scan."""
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from project_eval import HERE, f, OP, PARTS, REFERENCE, calibrate, json_at, write_json

MODES = ("indicators", "counts", "interactions")


def features(values, raw, mode):
    logits = f.features(values,"positive")
    found = (raw>0).astype(np.float32)
    if mode=="indicators":
        return np.column_stack((logits,found))
    count = np.log1p(raw.clip(0,64))
    if mode=="counts":
        return np.column_stack((logits,found,count))
    clipped = np.clip(logits[:,1],-4,4)
    return np.column_stack((logits,found,count,found*clipped[:,None]))


def correct(old, chosen, model, values, raw, mode):
    got = chosen & f.weak(raw)
    result = old.copy()
    if got.any():
        result[got] = model.predict_proba(features(values[got],raw[got],mode))[:,1]
    return result


def test(rotation):
    folder = HERE/rotation
    known = json_at(f.WEAK/rotation/"policy.json")
    frames = {}
    for name in ("dev","test"):
        with np.load(folder/f"{name}.npz") as z:
            frames[name] = dict(z)
    dev,test = frames["dev"],frames["test"]
    fit = np.isin(dev["ids"],known["fit_rows"])
    cal = np.isin(dev["ids"],known["calibration_rows"])
    budget = json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    heads,gate,cut,dbits,_ = calibrate(frames,REFERENCE,fit,cal,budget)
    results = {}
    for phase in (dev,test):
        values = phase[REFERENCE]
        p,asked = f.apply(values,f.estimate(heads,values),OP,gate,phase["raw"])
        results[id(phase)] = (p,asked)
    d,q = results[id(dev)]
    x,scan = results[id(test)]
    base = x>cut
    selected = fit & f.weak(dev["raw"])
    weights, summary = {}, {}
    for mode in MODES:
        model = LogisticRegression(C=.3,max_iter=350,random_state=42)
        model.fit(features(dev[REFERENCE][selected],dev["raw"][selected],mode),dev["y"][selected])
        weights[mode] = model
        score = correct(d,q,model,dev[REFERENCE],dev["raw"],mode)
        bits,threshold = f.strict_budget(score[cal],dev["y"][cal],budget)
        y = test["y"]
        pp = correct(x,scan,model,test[REFERENCE],test["raw"],mode)
        delta, by_project = f.per_project(y,test["project"],x,pp,base)
        frozen = f.comparison(y,base,pp>threshold)
        summary[mode] = dict(dev=f.comparison(dev["y"][cal],dbits,bits),
                             project_tp=delta,project_results=by_project,frozen=frozen,threshold=threshold)
        with (folder/f"rule_{mode}_test.csv").open("w",newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("row","project","gold","base","score","fixed","query"))
            writer.writerows(zip(test["ids"],test["project"],y,base.astype(int),pp,
                                 (pp>threshold).astype(int),scan.astype(int),strict=True))
        print("rule",rotation,mode,delta,frozen["delta_tp"],frozen["delta_fp"],
              summary[mode]["dev"]["delta_tp"],flush=True)
    joblib.dump(weights,folder/"rule_heads.joblib",compress=3)
    write_json(folder/"rule_evaluation.json",dict(scan=int(scan.sum()),positive_fit=int(selected.sum()),
                                                  comparisons=summary))


if __name__=="__main__":
    for rotation in PARTS:
        test(rotation)
