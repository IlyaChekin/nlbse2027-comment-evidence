"""Run and combine the versioned rule scan without needing any training data."""
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit, logit


HERE = Path(__file__).resolve().parent
GATES = {"all": (0., 1.), "medium": (.08, .96), "narrow": (.25, .88)}


def node_path():
    return os.environ.get("NLBSE_NODE", "node")


def scan_chunk(texts, only=None, program=None):
    source = "\n".join(json.dumps((i,t),separators=(",",":")) for i,t in enumerate(texts))+"\n"
    command = [node_path(), "--no-warnings", "--experimental-vm-modules", str(program or HERE/"scan.mjs")]
    if only:
        command.append("--only="+",".join(only))
    result = subprocess.run(command,input=source,text=True,capture_output=True,timeout=150,check=True)
    reader = csv.reader(io.StringIO(result.stdout))
    names = next(reader)[1:]
    rows = np.asarray([tuple(map(int,row)) for row in reader],dtype=np.int32)
    if not np.array_equal(rows[:,0],np.arange(len(texts))):
        raise ValueError("Static scans changed the function order")
    return names,rows[:,1:]


def scanner(texts, only=None, program=None):
    cpus = len(os.sched_getaffinity(0))
    workers = max(1,min(int(os.environ.get("NLBSE_SCANNER_WORKERS","6")),cpus,len(texts)//512))
    if workers == 1:
        return scan_chunk(texts,only,program)
    size = min(1600,(len(texts)+workers-1)//workers)
    batches = [texts[i:i+size] for i in range(0,len(texts),size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda batch:scan_chunk(batch,only,program),batches))
    names = results[0][0]
    if any(keys != names for keys,_ in results):
        raise ValueError("Parallel scanner workers disagree on feature names")
    return names,np.concatenate([values for _,values in results])


def get_features(p, raw):
    l = logit(p.clip(1e-5,1-1e-5))
    rule = raw > 0
    seen = rule.any(1)
    count = np.log1p(raw.clip(0,255))
    features = np.column_stack((l, rule, count, seen, rule.sum(1)))
    return l, seen, features


def selected_rows(p, mode):
    gate = "all" if mode.startswith("rule_") else mode.rsplit("_",1)[-1]
    lo, hi = GATES[gate]
    return np.ones(len(p),dtype=bool) if gate=="all" else (p>lo)&(p<hi)


def combine(probability, raw, mode, heads):
    logp, hits, features = get_features(probability,raw)
    if mode == "rule_priority":
        return .4*probability + .6*hits
    if mode.startswith("rule_boost_"):
        return expit(logp+float(mode.rsplit("_",1)[1])*hits)
    name, _ = mode.rsplit("_",1)
    score = heads[name].predict_proba(features)[:,1]
    chosen = selected_rows(probability,mode)
    t = logit(score.clip(1e-5,1-1e-5))
    return expit(logp+chosen*.75*(t-logp))


def refine(texts, probability, config, heads, rule_subset=None):
    mode = config["selected"]
    active = selected_rows(probability,mode)
    if not active.any():
        return probability
    columns, raw = scanner([text for text,enabled in zip(texts,active,strict=True) if enabled],rule_subset)
    if columns != config["rule_columns"]:
        raise ValueError("Rule feature order changed")
    features = np.zeros((len(probability),len(columns)),dtype=np.int32)
    features[active] = raw
    return combine(probability,features,mode,heads)
