"""Compare fresh conditional scores on frozen project rotations."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from train import HERE, ROOT, MODES, SEED, checksum, json_at, pipeline, small, write_json
import factor as f

REFERENCE = "reference"
OP = f.Policy(1., .24, "margin")
PARTS = f.base.ROTATIONS
ALL = (REFERENCE, *MODES)


def raw_values(ids):
    return np.column_stack((f.raw_rule_counts()[ids],f.numeric_counts()[ids]))


def cache(rotation, modes):
    texts, y, project, _, _ = f.inputs()
    indices = json_at(f.R19/"partition.json")["rotations"][rotation]
    path = f.R19/indices["partition_file"]
    if checksum(path)!=indices["partition_sha256"]:
        raise ValueError("Unexpected project rotation")
    with np.load(path) as data:
        ids_by_part = {phase:data[phase] for phase in ("dev","test")}
    saved = {}
    for phase, ids in ids_by_part.items():
        with np.load(f.HERE/"factor"/rotation/f"{phase}.npz") as old:
            if not np.array_equal(old["ids"],ids):
                raise ValueError("The reference differs from frozen project rotation")
            values = {REFERENCE:old["values"],REFERENCE+"_cost":old["model_cost"]}
        saved[phase] = dict(ids=ids,y=y[ids,0],project=project[ids],raw=raw_values(ids),**values)
    tok = small.tokenizer()
    route = f.pinned()[0]
    for mode in modes:
        print(rotation,"collect",mode,flush=True)
        runner = pipeline(rotation,mode,tok)
        for frame in saved.values():
            sample = [texts[i] for i in frame["ids"]]
            p,aux,cost = f.infer_student(runner,sample,route,columns=22,return_all=True)
            frame[mode] = np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
            frame[mode+"_cost"] = np.asarray([cost["forward_seconds"],cost["analytic_gflops"]])
        del runner
    folder = HERE/rotation
    folder.mkdir(exist_ok=True)
    for part, frame in saved.items():
        np.savez_compressed(folder/f"{part}.npz",**frame)


def calibrate(frames, variant, fit, dev, budget):
    train = frames["dev"]
    values = train[variant]
    heads = f.fit(values,train["raw"],train["y"],fit)
    learned = f.estimate(heads,values[dev])
    gate = f.calibrate(values[dev],learned,OP,train["y"][dev],budget)
    p, scan = f.apply(values[dev],learned,OP,gate,train["raw"][dev])
    bits, cutoff = f.strict_budget(p,train["y"][dev],budget)
    return heads, gate, cutoff, bits, scan


def record_fit(reference, candidate, y, flags):
    out = {}
    for name, selected in (("without_rules",flags==0),("with_rules",flags==1)):
        yi = y[selected]
        out[name] = f.comparison(yi,reference[selected],candidate[selected])
    return out


def evaluate(rotation, modes, folder=None):
    folder = HERE/rotation if folder is None else folder
    known = json_at(f.WEAK/rotation/"policy.json")
    loaded = {}
    for part in ("dev","test"):
        with np.load(folder/f"{part}.npz") as z:
            loaded[part] = dict(z)
    a,b = loaded["dev"],loaded["test"]
    fit = np.isin(a["ids"],known["fit_rows"])
    cal = np.isin(a["ids"],known["calibration_rows"])
    if (fit & cal).any() or fit.sum()!=len(known["fit_rows"]) or cal.sum()!=len(known["calibration_rows"]):
        raise ValueError("The separate development files do not align")
    budget = json_at(f.HERE/"factor_numeric"/rotation/"study.json")["dev_fp"]
    trained = {mode:calibrate(loaded,mode,fit,cal,budget) for mode in (REFERENCE,*modes)}
    reference_config = json_at(f.HERE/"factor_numeric"/rotation/"study.json")["options"][OP.name]
    if (abs(trained[REFERENCE][2]-reference_config["threshold"])>1e-7 or
            abs(trained[REFERENCE][1]["base"]-reference_config["gate"]["base"])>1e-7):
        raise ValueError("Reference does not reproduce round 20 on project development")
    data = {}
    for name,(heads,gate,cut,at_dev,asked_dev) in trained.items():
        values = b[name]
        estimates = f.estimate(heads,values)
        p,scan = f.apply(values,estimates,OP,gate,b["raw"])
        data[name] = dict(p=p, bits=p>cut, scan=scan)
    old = data[REFERENCE]
    a_ref = trained[REFERENCE][3]
    flag = np.where(f.valid(b["raw"]),f.weak(b["raw"]).astype(np.int8),-1)
    report = {}
    for name,(head,gate,cut,at_dev,query) in trained.items():
        item = data[name]
        net, by_project = f.per_project(b["y"],b["project"],old["p"],item["p"],old["bits"])
        bits_matched = np.zeros(len(b["y"]),dtype=bool)
        for project in by_project:
            there = b["project"]==project
            fp = int((((b["y"]==0)&old["bits"])[there]).sum())
            bits_matched[there],_ = f.exact_budget(item["p"][there],b["y"][there],fp)
        raw_negative = flag==0
        head_negative = f.estimate(head,b[name])["negative"]
        report[name] = dict(
            gate=gate,threshold=cut,project_tp=net,project_results=by_project,scans=int(item["scan"].sum()),
            dev=f.comparison(a["y"][cal],a_ref,at_dev),dev_scans=int(query.sum()),
            frozen=f.comparison(b["y"],old["bits"],item["bits"]),
            matched=f.comparison(b["y"],old["bits"],bits_matched),
            frozen_cohort=record_fit(old["bits"],item["bits"],b["y"],flag),
            matched_cohort=record_fit(old["bits"],bits_matched,b["y"],flag),
            weak_negative_ap=float(average_precision_score(b["y"][raw_negative],head_negative[raw_negative])),
            analytic_gflops=float(b[name+"_cost"][1]),
        )
        with (folder/f"{name}_test.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(("row","project","gold","weak","p","fixed","match_fp","query","negative_head"))
            writer.writerows(zip(b["ids"],b["project"],b["y"],flag,item["p"],item["bits"].astype(int),
                                 bits_matched.astype(int),item["scan"].astype(int),head_negative,strict=True))
        print(rotation,name,net,report[name]["frozen"]["delta_tp"],report[name]["frozen"]["delta_fp"],
              report[name]["dev"]["delta_tp"],report[name]["weak_negative_ap"],flush=True)
    write_json(folder/"evaluation.json",dict(
        policy=asdict(OP),fit_rows=int(fit.sum()),calibration_rows=int(cal.sum()),dev_fp=budget,
        source_train_sha256=checksum(f.ROOT/"data/raw/train-00000-of-00001.parquet"),
        dev_sha256=checksum(folder/"dev.npz"),test_sha256=checksum(folder/"test.npz"),results=report,
    ))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=PARTS)
    p.add_argument("--modes",nargs="+",choices=MODES,default=MODES)
    p.add_argument("--cache-ready",action="store_true")
    a=p.parse_args()
    if not a.cache_ready:
        cache(a.rotation,a.modes)
    evaluate(a.rotation,a.modes)
