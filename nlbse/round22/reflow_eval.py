"""Evaluate additional static evidence from line joining at fixed project FP budgets."""
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from project_eval import HERE, f, OP, PARTS, REFERENCE, calibrate, json_at, write_json

MODES = ("any_joined","rule_joined","selective_rules")


def features(values, added, mode):
    basic = f.features(values,"negative")
    indicator = added>0
    evidence = indicator.any(1)[:,None] if mode=="any_joined" else indicator
    return np.column_stack((basic,evidence))


def frame(path, changed_data):
    with np.load(path) as contents:
        x = dict(contents)
    i = np.searchsorted(changed_data["rows"],x["ids"])
    if not np.array_equal(changed_data["rows"][i],x["ids"]):
        raise ValueError("Layout scan does not cover these rows")
    mask = np.zeros(len(changed_data["rows"]),dtype=bool)
    mask[changed_data["changed"]] = True
    return x | dict(joined=changed_data["raw"][i],layout_diff=mask[i])


def reference(fr, heads, gate):
    p, queried = f.apply(fr[REFERENCE],f.estimate(heads,fr[REFERENCE]),OP,gate,fr["raw"])
    return p, queried


def revised(fr, p, asked, model, mode, cutoff):
    has_rule = f.weak(fr["raw"])
    candidates = asked & f.valid(fr["raw"]) & ~has_rule & fr["layout_diff"]
    if mode=="selective_rules":
        candidates &= (p>max(.025,cutoff-.10))&(p<cutoff+.08)
    fired = candidates & f.weak(fr["joined"])
    q = p.copy()
    if fired.any():
        q[fired] = model.predict_proba(features(fr[REFERENCE][fired],fr["joined"][fired],mode))[:,1]
    return q,candidates,fired


def run(rotation, cached):
    folder = HERE/rotation
    frames = {name:frame(folder/f"{name}.npz",cached) for name in ("dev","test")}
    dev,test = frames["dev"],frames["test"]
    old = json_at(f.WEAK/rotation/"policy.json")
    fit = np.isin(dev["ids"],old["fit_rows"])
    cal = np.isin(dev["ids"],old["calibration_rows"])
    budget = json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    head,gate,cut,dev_bits,_ = calibrate(frames,REFERENCE,fit,cal,budget)
    d,dq = reference(dev,head,gate)
    t,tq = reference(test,head,gate)
    ref = t>cut
    fit &= f.valid(dev["raw"]) & ~f.weak(dev["raw"]) & dev["layout_diff"]
    result,models = {},{}
    for mode in MODES:
        model = LogisticRegression(C=.25,max_iter=350,random_state=42)
        model.fit(features(dev[REFERENCE][fit],dev["joined"][fit],mode),dev["y"][fit])
        p,asks,_ = revised(dev,d,dq,model,mode,cut)
        bits,threshold = f.strict_budget(p[cal],dev["y"][cal],budget)
        x,active,new = revised(test,t,tq,model,mode,cut)
        fixed = f.comparison(test["y"],ref,x>threshold)
        n,by_project = f.per_project(test["y"],test["project"],t,x,ref)
        result[mode] = dict(project_tp=n,project_results=by_project,frozen=fixed,threshold=threshold,
                            extra_scans=int(active.sum()),extra_signals=int(new.sum()),
                            dev=f.comparison(dev["y"][cal],dev_bits,bits),dev_queries=int(asks[cal].sum()))
        models[mode] = model
        with (folder/f"layout_{mode}_test.csv").open("w",newline="") as stream:
            w=csv.writer(stream)
            w.writerow(("row","project","gold","base","score","changed_view","extra_scan","signal","fixed"))
            w.writerows(zip(test["ids"],test["project"],test["y"],ref.astype(int),x,test["layout_diff"].astype(int),
                            active.astype(int),new.astype(int),(x>threshold).astype(int),strict=True))
        print(rotation,mode,n,fixed["delta_tp"],fixed["delta_fp"],result[mode]["dev"]["delta_tp"],
              result[mode]["extra_scans"],flush=True)
    joblib.dump(models,folder/"layout_heads.joblib",compress=3)
    write_json(folder/"layout_evaluation.json",result)


if __name__=="__main__":
    with np.load(HERE/"reflow/full/join_parentheses.npz") as z:
        cache = dict(z)
    for rotation in PARTS:
        run(rotation,cache)
