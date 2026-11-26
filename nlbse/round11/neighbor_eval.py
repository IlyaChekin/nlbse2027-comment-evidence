"""Test whether train-only neighbors improve vulnerability decisions at fixed MAT."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"round10"))
from students import (PUBLIC, ROOT, calibrated, dataset, group_codes, load, metric, save, sha256, write_json)
from round2_report import bootstrap_delta
from score_runtime import competition_formula

OUT = HERE/"neighbors"
FAMILIES = ("t5_small", "unixcoder9")
MAT_REFERENCE = ROOT/"round10/student_context/student_lexical/predictions.csv"


def neighbors(path, rows):
    values = np.fromfile(path, dtype="<f4").reshape(-1,16)
    if len(values)!=rows or not np.isfinite(values).all():
        raise ValueError("Unexpected neighbor score matrix")
    values[:, 12:] = np.log1p(values[:, 12:])
    return values


def features(sim, neural=None):
    margins = sim[:, 1:12:2] - sim[:, :12:2]
    base = np.column_stack((sim, margins))
    if neural is None:
        return base
    raw = logit(neural.clip(1e-6, 1-1e-6))
    return np.column_stack((raw, base, raw*(sim[:, 4:6].max(axis=1)), np.column_stack([(1-np.abs(neural-.5)*2)*margins[:,i] for i in range(3)])))


def table(path):
    import csv
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    return np.asarray([[int(row["row"]),float(row["p_vulnerability"]),int(row["pred_vulnerability"]),int(row["pred_MAT"])] for row in rows])


def control(y, raw, threshold):
    return metric(y, np.column_stack((raw > threshold, np.zeros(len(y), dtype=bool))))["vulnerability"]["f1"]


def cutoff(y, p):
    dummy = np.column_stack((p,np.zeros(len(p))))
    return calibrated(y,dummy)[0]


def receipt_for(family):
    folder = ROOT/"round10/runs"/family
    chosen = json.loads((folder/"selected.json").read_text())
    return folder,chosen


def tune():
    _, _, _, y, ids, _ = dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as reserve:
        if ids != reserve["rows"].tolist():
            raise ValueError("Unexpected holdout row order")
        folds = reserve["folds"]
    fit, cal, review = folds <= 2, folds == 3, folds == 4
    sim = neighbors(OUT/"holdout.f32",len(y))
    selections = {}
    for family in ("neighbor_only", *FAMILIES):
        if family == "neighbor_only":
            p = None
            baseline = None
        else:
            folder, chosen = receipt_for(family)
            name = Path(chosen["checkpoint"]).stem
            with np.load(folder/f"{name}_holdout.npz") as data:
                if ids != data["ids"].tolist():
                    raise ValueError("Encoder probabilities are misaligned")
                p = data["p"][:,0]
            baseline = control(y[review], p[review], chosen["settings"]["thresholds"][0])
        x = features(sim,p)
        candidates = []
        recipes = {"logistic": LogisticRegression(C=.2,max_iter=400),
                   "trees":HistGradientBoostingClassifier(learning_rate=.075, max_iter=155,max_leaf_nodes=12,
                         min_samples_leaf=90,l2_regularization=1., random_state=42)}
        for name, model in recipes.items():
            model.fit(x[fit],y[fit,0])
            q = model.predict_proba(x)[:,1]
            threshold = cutoff(y[cal],q[cal])
            candidates.append((dict(name=name,threshold=threshold,calibration=control(y[cal],q[cal],threshold),
                                    review=control(y[review],q[review],threshold)),model,q))
        best, model, q = max(candidates,key=lambda record:record[0]["calibration"])
        artifact = OUT/f"{family}.joblib"
        joblib.dump(model,artifact)
        np.savez_compressed(OUT/f"{family}_holdout.npz", ids=ids, y=y, p=q, similarity=sim, folds=folds)
        selections[family] = dict(chosen=best, baseline_review=baseline, variants=[r[0] for r in candidates],
                                  sha256=sha256(artifact))
        print(family,selections[family],flush=True)
    write_json(OUT/"policy.json",selections)


def public(query_seconds):
    _, _, _, y, ids, cal = dataset(public=True)
    _, _, fields = load(PUBLIC,audit=True)
    group_ids = group_codes(zip(fields["Projectname"], fields["Filepath"],strict=True))
    sim = neighbors(OUT/"public.f32",len(y))
    selections = json.loads((OUT/"policy.json").read_text())
    mat = table(MAT_REFERENCE)
    if ids != mat[:,0].tolist():
        raise ValueError("Unexpected MAT reference rows")
    result = {}
    import time
    for family, recipe in selections.items():
        if family == "neighbor_only":
            raw = None
            forward, gflops = 0.,0.
            reference = mat[:,2:4].astype(bool)
        else:
            folder, _ = receipt_for(family)
            base = table(folder/"public/predictions.csv")
            if ids != base[:,0].tolist():
                raise ValueError("Unexpected encoder reference rows")
            raw = base[:,1]
            reference = np.column_stack((base[:,2],mat[:,3])).astype(bool)
            cost = json.loads((folder/"public/analysis.json").read_text())["cost"]
            forward,gflops = cost["forward_seconds"],cost["analytic_gflops"]
        artifact = OUT/f"{family}.joblib"
        if sha256(artifact) != recipe["sha256"]:
            raise ValueError("Mismatch in fitted neighbor head weights")
        model = joblib.load(artifact)
        began = time.perf_counter()
        q = model.predict_proba(features(sim,raw))[:,1]
        readout = time.perf_counter()-began
        t = (recipe["chosen"]["threshold"],.5)
        probs = np.column_stack((q,mat[:,3]))
        bits = probs > t
        m = metric(y,bits)
        cost = dict(forward_seconds=forward+query_seconds+readout, analytic_gflops=gflops,
                    cpu_index_query_seconds=query_seconds, readout_seconds=readout)
        save(OUT/family,y,ids,cal,probs,cost,t)
        result[family] = dict(scores=m, interval=bootstrap_delta(y,reference,bits,group_ids),cost=cost,
               combined_cost_formula=competition_formula(m["mean"]["f1"],cost["forward_seconds"],gflops),
               similarity_coverage={str(c):float((sim[:,4:6].max(1)>=c).mean()) for c in (.5,.75,.9)})
    write_json(OUT/"analysis.json",result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("tune","public"))
    ap.add_argument("--query-seconds",type=float)
    args=ap.parse_args()
    if args.stage=="tune":
        tune()
    else:
        if args.query_seconds is None:
            ap.error("Public evaluation requires measured query time")
        public(args.query_seconds)
