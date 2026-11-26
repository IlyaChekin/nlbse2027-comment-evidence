"""Measure the unchanged classifier with the early-exit scanner alternated against the full scan."""
import csv
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np

from layout_official import ARCHIVE, PROGRAM as FULL, f, load, PUBLIC, recipe, recover
from union_scanner import HERE, ROOT, PROGRAM as FAST, CONFIG, checksum, json_at, scan, write_json

MODES = ("reference","short_circuit")
OUT = HERE/"union_public"


def benchmark():
    cfg = json_at(CONFIG)
    auth = json_at(HERE/"union_verification.json")
    if auth["differences"] or auth["rows"]!=214586 or not auth["program_sha256"]==cfg["program_sha256"]==checksum(FAST):
        raise ValueError("Shortcut failed exact row-wise verification")
    ref,policy,heads = recipe()
    texts,y,_ = load(PUBLIC)
    with (ARCHIVE/"profile_marker.csv").open(newline="") as stream:
        old = np.asarray([[int(r[k]) for k in ("pred_vulnerability","pred_MAT")]
                          for r in csv.DictReader(stream)],dtype=bool)
    expected_cols = json_at(f.SOURCE/"policy.json")["rule_columns"] + ["rule_195"]
    model = f.pipe("official",f.small.tokenizer())
    route,_ = f.pinned()
    selected = {}

    def runner(name, text):
        p,aux,cost = f.infer_student(model,text,route,columns=22,return_all=True)
        t = time.perf_counter()
        values = np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
        score,no,yes,active = f.choose(values,f.estimate(heads,values),policy,ref["gate"])
        requests = [s for s,on in zip(text,active,strict=True) if on]
        scan_started = time.perf_counter()
        if requests:
            if name=="short_circuit":
                outcome = scan(requests)
            else:
                cols,raw = f.scanner(requests,program=FULL)
                if cols!=expected_cols:
                    raise ValueError("Baseline scan columns changed")
                outcome = np.where(f.valid(raw),f.weak(raw),-1)
            score[active] = np.where(outcome>=0,np.where(outcome==1,yes[active],no[active]),score[active])
        else:
            outcome = np.empty(0,dtype=np.int8)
        elapsed = time.perf_counter()-scan_started
        tags = recover(text,p[:,1]>.5)
        total = time.perf_counter()-t
        cost["forward_seconds"]+=total
        cost["wall_seconds"]+=total
        cost.update(scanned=len(requests),scan_seconds=elapsed)
        p[:,0] = score
        return np.column_stack((p[:,0]>ref["threshold"],tags)),active,outcome,cost

    times = {m:[] for m in MODES}
    for m in MODES:
        runner(m,texts[:2048])
    for pair in range(5):
        for m in MODES if pair%2==0 else reversed(MODES):
            result,active,outcome,cost = runner(m,texts)
            if not np.array_equal(result,old):
                raise ValueError("Archived binary decisions changed")
            if "gate" not in selected:
                selected.update(gate=active,flag=outcome)
            elif not np.array_equal(active,selected["gate"]) or not np.array_equal(outcome,selected["flag"]):
                raise ValueError("A gate request returned a different exact outcome")
            times[m].append(cost)
            print(m,cost,flush=True)
    OUT.mkdir(exist_ok=True)
    quality = f.metric(y,old)
    stats = {}
    for m in MODES:
        cost = {k:median(trial[k] for trial in times[m]) for k in times[m][0]}
        score = f.competition_formula(quality["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])
        stats[m] = dict(score=score,cost=cost,trials=times[m])
    np.savez_compressed(OUT/"scanner_predictions.npz",**selected)
    write_json(OUT/"results.json",dict(stats=stats,quality=quality,probe_count=int(selected["gate"].sum()),
                         public_sha256=checksum(PUBLIC),program_sha256=checksum(FAST),
                         baseline_predictions_sha256=checksum(ARCHIVE/"profile_marker.csv"),
                         scanner_predictions_sha256=checksum(OUT/"scanner_predictions.npz")))
    print({m:stats[m]["score"] for m in MODES},flush=True)


if __name__=="__main__":
    benchmark()
