"""Measure residual MAT information in the compact encoder's pooled representation."""

import argparse
import csv
import json
from time import perf_counter

import joblib
import numpy as np
from scipy.special import expit
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit import HERE, ROOT, dataset, split, write_json, checksum
from mat import cut_at_budget, evaluate
from rotate import load_backbones


OUT = HERE / "readout"


@torch.inference_mode()
def extract(model, tokenizer, algorithms, texts):
    width = 2*model.width+5
    vectors = np.empty((len(texts), width+1), dtype=np.float32)
    buffer = []
    handle = model.head[0].register_forward_hook(lambda module, args, result: buffer.append(result))
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            for offset in range(0,len(texts),2048):
                block = texts[offset:offset+2048]
                side = torch.as_tensor(algorithms.students.lexical_features(block), device="cuda")
                seq = algorithms.small.encode(block, tokenizer)
                for rows in algorithms.students.batch_plan(list(range(len(seq))),seq,algorithms.small.BATCH):
                    x, mask = algorithms.students.batch(rows,seq,tokenizer.pad_token_id)
                    output = model(x,mask,side[rows])
                    combined = torch.cat((buffer.pop(), output[:,1:2]), dim=1).float().cpu().numpy()
                    vectors[offset+np.asarray(rows)] = combined
    finally:
        handle.remove()
    return vectors


def get(role, cached):
    texts, labels, _, _ = dataset()
    ids = split()[role]
    with np.load(HERE / f"evidence_{role}.npz") as flags:
        if not np.array_equal(ids, flags["ids"]):
            raise ValueError(f"Evidence cache for {role} changed")
        tagged = flags["flags"][:,0]
    selected = ids[~tagged]
    folder = OUT / f"{role}.npz"
    if folder.is_file():
        with np.load(folder) as f:
            if not np.array_equal(f["rows"], selected):
                raise ValueError(f"Activation cache for {role} changed")
            data = f["x"]
    else:
        start = perf_counter()
        model, tok, code = cached
        data = extract(model, tok, code, [texts[i] for i in selected])
        np.savez(folder, rows=selected, x=data)
        print(role, len(selected), perf_counter()-start, flush=True)
    return ids, labels[ids,1].astype(bool), tagged, data


def setup():
    algorithms = load_backbones()
    model, tokenizer = algorithms.frozen("compact")
    model.eval()
    return model, tokenizer, algorithms


def score_vectors(tagged, data, model):
    p = np.full(len(tagged), 2., dtype=np.float32)
    p[~tagged] = model.predict_proba(data)[:,1] if model is not None else expit(data[:,-1])
    return p


def tune():
    OUT.mkdir(exist_ok=True)
    net = setup()
    _, fit_gold, fit_tagged, x = get("fit", net)
    _, dev_gold, dev_tagged, dev_x = get("dev", net)
    records = {}
    models = {"native": None}
    for strength in (.003, .02, .12):
        key = f"ridge/{strength:g}"
        model = make_pipeline(StandardScaler(), LogisticRegression(C=strength, solver="liblinear", max_iter=240, random_state=42))
        gold = fit_gold[~fit_tagged]
        model.fit(x, gold, logisticregression__sample_weight=np.where(gold, 3., 1.))
        models[key] = model
    budget = int((dev_tagged & ~dev_gold).sum())
    for key, model in models.items():
        p = score_vectors(dev_tagged,dev_x,model)
        threshold = cut_at_budget(p,dev_gold,budget)
        records[key] = dict(threshold=threshold, metrics=evaluate(dev_gold,dev_tagged,p>threshold),
                            residual_ap=average_precision_score(dev_gold[~dev_tagged],p[~dev_tagged]))
        print(key, records[key], flush=True)
    def quality(key):
        return records[key]["metrics"]["recovered"], records[key]["residual_ap"]

    winner = max(records, key=quality)
    learned = max((key for key in records if key != "native"), key=quality)
    path = OUT/"frozen.joblib"
    joblib.dump({k:v for k,v in models.items() if k in {learned,"native"}}, path, compress=3)
    source = ROOT / "round13/backbones/compact/selected.json"
    write_json(OUT/"policy.json",dict(chosen=winner, learned=learned, candidates=records, artifact_sha256=checksum(path),
                                      source=str(source.relative_to(ROOT)), source_sha256=checksum(source)))


def test():
    config=json.loads((OUT/"policy.json").read_text())
    path=OUT/"frozen.joblib"
    if checksum(path)!=config["artifact_sha256"] or checksum(ROOT/config["source"])!=config["source_sha256"]:
        raise ValueError("Frozen activation heads changed")
    models=joblib.load(path)
    net=setup()
    result={}
    for role in ("test_a","test_b"):
        rows,y,tagged,d=get(role,net)
        budget=int((~y&tagged).sum())
        result[role]={}
        _,_,project,_=dataset()
        for key,m in models.items():
            score=score_vectors(tagged,d,m)
            exact=score>cut_at_budget(score,y,budget)
            frozen=score>config["candidates"][key]["threshold"]
            result[role][key]=dict(exact=evaluate(y,tagged,exact),frozen=evaluate(y,tagged,frozen))
            csv_name=f"{role}_{key.replace('/','_')}.csv"
            with (OUT/csv_name).open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(("row","project","label","base","score","fixed","matched"))
                w.writerows(zip(rows,project[rows],y.astype(int),tagged.astype(int),score,frozen.astype(int),exact.astype(int),strict=True))
            print(role,key,result[role][key],flush=True)
    write_json(OUT/"results.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("tune","test"))
    options=parser.parse_args()
    tune() if options.stage=="tune" else test()
