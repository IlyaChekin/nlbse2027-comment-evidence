"""Bootstrap per-project matched false-positive gains by source file."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from transfer import HERE, ROTATIONS, inputs, write_json


def read(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return (np.asarray([int(row["row"]) for row in rows]),
            np.asarray([float(row["probability"]) for row in rows], dtype=np.float32),
            np.asarray([row["baseline_by_project"]=="1" for row in rows]))


def hits(p_order, gold, weight, budget):
    negatives = ~gold[p_order]
    cum = np.cumsum(negatives*weight[p_order])
    stop = np.searchsorted(cum,budget,side="right")
    selected = p_order[:stop]
    return int(weight[selected[gold[selected]]].sum())


def bootstrap(folder, draws, labels, project, files):
    record = json.loads((folder/"results.json").read_text())["test"]
    row, p0, base = read(folder/"test_reference.csv")
    vectors = {}
    for name in record:
        if name=="reference":
            continue
        ids, p, old = read(folder/f"test_{name}.csv")
        if not np.array_equal(ids,row) or not np.array_equal(old,base):
            raise ValueError("Project predictions differ in row order or fixed comparison")
        vectors[name] = p
    rng = np.random.default_rng(42)
    total = {m:np.zeros(draws,dtype=np.int32) for m in vectors}
    group_results = {m:{} for m in vectors}
    for group in sorted(set(project[row])):
        mask = project[row]==group
        rows = row[mask]
        y = labels[rows,0].astype(bool)
        b = base[mask]
        _, file_codes = np.unique([files[i] for i in rows],return_inverse=True)
        groups = file_codes.max()+1
        base_order = np.argsort(-p0[mask],kind="stable")
        orders = {k:np.argsort(-p[mask],kind="stable") for k,p in vectors.items()}
        expectation = counts_at_reference(base_order,y,b,orders)
        for arm,tp in expectation.items():
            if tp != record[arm]["by_project"][group]["delta_tp"]:
                raise ValueError("Audit ranking did not reproduce the project results")
        local = {m:np.empty(draws,dtype=np.int32) for m in vectors}
        for i in range(draws):
            counts = np.bincount(rng.integers(groups,size=groups),minlength=groups)
            weights = counts[file_codes]
            budget = int(weights[~y & b].sum())
            before = hits(base_order,y,weights,budget)
            for name, order in orders.items():
                local[name][i] = hits(order,y,weights,budget)-before
        for name, values in local.items():
            total[name] += values
            group_results[name][group] = np.quantile(values,[.025,.975]).tolist()
    return {name:dict(draws=draws, project_tp_delta_by_file=group_results[name],
                      delta_tp_by_file=np.quantile(values,[.025,.975]).tolist())
            for name,values in total.items()}


def counts_at_reference(base_order, y, b, orders):
    all_once = np.ones(len(y),dtype=np.int8)
    budget = int((~y & b).sum())
    baseline = hits(base_order,y,all_once,budget)
    return {k:hits(order,y,all_once,budget)-baseline for k,order in orders.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=HERE)
    p.add_argument("--draws",type=int,default=1000)
    args = p.parse_args()
    _, y, project, files, _ = inputs()
    for rotation in ROTATIONS:
        for result in sorted((args.source/rotation).glob("*/results.json")):
            path = result.parent
            intervals = bootstrap(path,args.draws,y,project,files)
            write_json(path/"intervals.json",intervals)
            print(rotation,path.name,{n:r["delta_tp_by_file"] for n,r in intervals.items()},flush=True)
