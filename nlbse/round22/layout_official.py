"""Select the additional layout check using the official internal file folds."""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from project_eval import HERE, ROOT, REFERENCE, f, raw_values, checksum, json_at, write_json
from reflow import reflow, PROGRAM

sys.path.insert(0,str(ROOT/"round21/mat"))
from marker import recover
from pipeline import PUBLIC, load

OUT = HERE/"layout_official"
ARCHIVE = ROOT/"round21/mat"
FACTORS = ROOT/"round20/factor_numeric/official"
RULE = "783"
POLICIES = (("all",1.),("all",.5),("near",1.),("near",.5))


def features(values, positive):
    return np.column_stack((f.features(values,"negative"),positive))


def observation(text, eligible):
    ticks = time.perf_counter()
    valid, scan_text = [], []
    for i in np.flatnonzero(eligible):
        reformatted = reflow(text[i],"join_parentheses")
        if reformatted!=text[i]:
            valid.append(i)
            scan_text.append(reformatted)
    found = np.zeros(len(text),dtype=bool)
    if scan_text:
        cols, raw = f.scanner(scan_text,only=(RULE,),program=PROGRAM)
        if cols!=[f"rule_{RULE}"]:
            raise ValueError("Unexpected layout scanner output")
        found[valid] = f.weak(raw)
    return found,len(valid),time.perf_counter()-ticks


def eligible(mode, p, cut, actual_negative):
    if mode=="near":
        return actual_negative & (p>max(.025,cut-.10)) & (p<cut+.08)
    return actual_negative


def adjust(p, values, model, hit, weight):
    q = p.copy()
    if hit.any():
        q[hit] = (1-weight)*p[hit]+weight*model.predict_proba(features(values[hit],hit[hit]))[:,1]
    return q


def recipe():
    record = json_at(FACTORS/"policy.json")
    if not record["numeric"] or checksum(PROGRAM)!=record["program_sha256"]:
        raise ValueError("The archived static setup changed")
    ref = record["modes"][record["selected"]]
    return ref,f.Policy(**ref["policy"]),joblib.load(FACTORS/"heads.joblib")


def tune():
    OUT.mkdir(exist_ok=True)
    _,_,text,y,ids,_ = f.dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as z:
        if not np.array_equal(ids,z["rows"]):
            raise ValueError("Official tuning folds changed")
        groups = z["folds"]
    core,cal,review = groups<3,groups==3,groups==4
    ref,policy,heads = recipe()
    values,mat,_ = f.factor_data("official",text,f.small.tokenizer())
    mat = recover(text,mat)
    raw = raw_values(ids)
    scores,asked = f.apply(values,f.estimate(heads,values),policy,ref["gate"],raw)
    phase = f.valid(raw) & ~f.weak(raw)
    # The fit head learns the relationship on rule-negative functions independently of the gate.
    core_hit,n,dt = observation(text,core & phase)
    model = LogisticRegression(C=.25,max_iter=350,random_state=42)
    mask = core & phase
    model.fit(features(values[mask],core_hit[mask]),y[mask,0])
    print("fit",int(mask.sum()),n,int(core_hit.sum()),dt,flush=True)
    joblib.dump(model,OUT/"head.joblib",compress=3)
    old = scores>ref["threshold"]
    budget = f.counts(y[cal,0],old[cal])["fp"]
    selected = REFERENCE
    best = 0.
    entries = {}
    cached = {}
    for mode,weight in POLICIES:
        if mode not in cached:
            e = eligible(mode,scores,ref["threshold"],phase & asked)
            c = [observation([s for s,k in zip(text,cal,strict=True) if k],e[cal]) for _ in range(3)]
            hit,queries,elapsed = observation(text,e & (cal|review))
            if not np.array_equal(hit[cal],c[0][0]):
                raise ValueError("Duplicate scans differed")
            cached[mode] = (hit,queries,median(r[2] for r in c),c[0][1])
        hit,queries,elapsed,cal_queries = cached[mode]
        modified = adjust(scores,values,model,hit,weight)
        bits,threshold = f.strict_budget(modified[cal],y[cal,0],budget)
        cal_metrics = f.comparison(y[cal,0],old[cal],bits)
        rev_metrics = f.comparison(y[review,0],old[review],modified[review]>threshold)
        quality_gain = .3*(cal_metrics["f1_after"]-cal_metrics["f1_before"])
        proxy = quality_gain-.0002*elapsed*f.PUBLIC_N/cal.sum()
        key = f"{mode}_{weight:g}"
        entries[key] = dict(mode=mode,weight=weight,threshold=threshold,cal=cal_metrics,review=rev_metrics,
                            proxy=float(proxy),cal_quality_gain=quality_gain,cal_seconds=elapsed,
                            cal_queries=cal_queries,cal_review_queries=queries)
        if proxy > best:
            selected,best = key,proxy
        print(key,entries[key],flush=True)
    write_json(OUT/"policy.json",dict(selected=selected,recipes=entries,rule=RULE,source_threshold=ref["threshold"],
              source_policy_sha256=checksum(FACTORS/"policy.json"),head_sha256=checksum(OUT/"head.joblib"),
              holdout_sha256=checksum(ROOT/"round8/frozen/holdout.npz"),script_sha256=checksum(Path(__file__))))


def public():
    stored = json_at(OUT/"policy.json")
    selected = stored["selected"]
    if selected==REFERENCE:
        raise ValueError("The internal calibration preserved the reference")
    config = stored["recipes"][selected]
    if checksum(OUT/"head.joblib")!=stored["head_sha256"] or checksum(FACTORS/"policy.json")!=stored["source_policy_sha256"]:
        raise ValueError("Frozen internal policy dependencies changed")
    ref,policy,heads = recipe()
    layout_model = joblib.load(OUT/"head.joblib")
    texts,y,_ = load(PUBLIC)
    with (ARCHIVE/"profile_marker.csv").open(newline="") as stream:
        old = np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")] for r in csv.DictReader(stream)],dtype=bool)
    expected_cols = json_at(f.SOURCE/"policy.json")["rule_columns"] + ["rule_195"]
    tok = f.small.tokenizer()
    model = f.pipe("official",tok)
    route,_ = f.pinned()
    def runner(variant,text):
        p,aux,cost = f.infer_student(model,text,route,columns=22,return_all=True)
        t = time.perf_counter()
        values = np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
        score,no,yes,active = f.choose(values,f.estimate(heads,values),policy,ref["gate"])
        exact_negative = np.zeros(len(text),dtype=bool)
        if active.any():
            cols,raw = f.scanner([s for s,on in zip(text,active,strict=True) if on],program=PROGRAM)
            if cols != expected_cols:
                raise ValueError("Baseline scan changed")
            exact_negative[active] = f.valid(raw)&~f.weak(raw)
            score[active] = np.where(f.valid(raw),np.where(f.weak(raw),yes[active],no[active]),score[active])
        calls, layout_seconds = 0, 0.
        threshold = ref["threshold"]
        if variant != REFERENCE:
            allow = eligible(config["mode"],score,threshold,exact_negative)
            hit,calls,layout_seconds = observation(text,allow)
            score = adjust(score,values,layout_model,hit,config["weight"])
            threshold = config["threshold"]
        mat = recover(text,p[:,1]>.5)
        interval = time.perf_counter()-t
        cost["forward_seconds"]+=interval
        cost["wall_seconds"]+=interval
        cost.update(extra_scans=calls,scanned=int(active.sum()),layout_seconds=layout_seconds)
        return np.column_stack((score>threshold,mat)),score,cost
    modes = (REFERENCE,selected)
    trials = {k:[] for k in modes}
    results = {}
    for k in modes:
        runner(k,texts[:2048])
    for turn in range(5):
        for k in modes if turn%2==0 else reversed(modes):
            bits,p,cost = runner(k,texts)
            if (k==REFERENCE and not np.array_equal(bits,old)) or not np.array_equal(bits[:,1],old[:,1]):
                raise ValueError("Reference decisions or fixed MAT changed")
            if k in results and not np.array_equal(bits,results[k][0]):
                raise ValueError("Repeated decisions changed")
            results[k] = bits,p
            trials[k].append(cost)
            print(k,cost,flush=True)
    fp = int((old[:,0]&(y[:,0]==0)).sum())
    report = {}
    for k in modes:
        bits,p = results[k]
        quality = f.metric(y,bits)
        cost = {key:median(t[key] for t in trials[k]) for key in trials[k][0]}
        matched,_ = f.exact_budget(p,y[:,0],fp)
        out = OUT/f"{k}_public.csv"
        with out.open("w",newline="") as file:
            w = csv.writer(file)
            w.writerow(("row","label_vulnerability","label_mat","pred_vulnerability","pred_MAT","probability","matched"))
            w.writerows((i,*yy,*b,s,m) for i,(yy,b,s,m) in enumerate(zip(y,bits.astype(int),p,matched.astype(int),strict=True)))
        report[k] = dict(score=f.competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"]),
            quality=quality,cost=cost,trials=trials[k],frozen=f.comparison(y[:,0],old[:,0],bits[:,0]),
            matched=f.comparison(y[:,0],old[:,0],matched),predictions_sha256=checksum(out))
    write_json(OUT/"results.json",dict(selected=selected,results=report,public_sha256=checksum(PUBLIC),
                                     policy_sha256=checksum(OUT/"policy.json")))
    print({k:v["score"] for k,v in report.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("tune","public"))
    {"tune":tune,"public":public}[p.parse_args().stage]()
