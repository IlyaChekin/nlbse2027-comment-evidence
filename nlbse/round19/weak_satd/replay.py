"""Evaluate passive WeakSATD rule evidence as a residual to the compact encoder."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from transfer import ROTATIONS, checksum, comparison, counts, exact_budget, inputs, strict_budget, write_json
from project_split import fingerprint
from runtime import combine, get_features


def read_scores(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.asarray([int(r["row"]) for r in rows]),
            np.asarray([float(r["probability"]) for r in rows], dtype=np.float32),
            np.asarray([r["baseline_equal"]=="1" for r in rows]))


def rule_data(path):
    with path.open(newline="") as f:
        reader = csv.reader(f)
        keys = next(reader)[1:]
        matrix = np.asarray([tuple(map(int,r)) for r in reader],dtype=np.int32)
    if not np.all(np.diff(matrix[:,0])>0) or not keys or matrix.shape[1]!=len(keys)+1:
        raise ValueError("Rule rows must be unique and sorted")
    return keys, matrix[:,0], matrix[:,1:]


def suite(p, raw, models):
    modes = ["rule_priority",*(f"rule_boost_{shift:g}" for shift in (.5,1.,2.,4.))]
    modes.extend(f"{name}_{gate}" for name in models for gate in ("all","medium","narrow"))
    return {"reference":p, **{mode:combine(p,raw,mode,models) for mode in modes}}


def fit_combiner(x, y):
    models = dict(linear=LogisticRegression(C=.3,max_iter=500,random_state=42),
                  trees=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=12,min_samples_leaf=70,
                                                       l2_regularization=3,random_state=42,
                                                       monotonic_cst=[1]+[0]*(x.shape[1]-1)))
    for model in models.values():
        model.fit(x,y)
    return models


def experiment(rotation, keys, all_ids, all_rules):
    texts, label, project, files, _ = inputs()
    reference = HERE.parent/rotation/"42"
    out = HERE/rotation
    out.mkdir(exist_ok=True)
    rows = {}
    for stage in ("dev","test"):
        ids, p, base = read_scores(reference/f"{stage}_reference.csv")
        pos = np.searchsorted(all_ids,ids)
        if (pos>=len(all_ids)).any() or not np.array_equal(all_ids[pos],ids):
            raise ValueError("Passive rule cache does not cover the selected rows")
        raw = all_rules[pos]
        y = label[ids,0]
        budget = counts(y,base)["fp"]
        rows[stage] = ids,p,base,raw,y,budget
    ids, p, base, raw, y, _ = rows["dev"]
    model_fit = np.asarray([sha256(f"WS19\0{project[i]}\0{files[i]}".encode()).digest()[0] < 128 for i in ids])
    model_dev = ~model_fit
    calibration_texts = {fingerprint(texts[i]) for i in ids[model_dev]}
    model_fit &= np.asarray([fingerprint(texts[i]) not in calibration_texts for i in ids])
    _, _, x = get_features(p,raw)
    models = fit_combiner(x[model_fit],y[model_fit])
    scores = suite(p,raw,models)
    dev_y = y[model_dev]
    budget = counts(dev_y,base[model_dev])["fp"]
    ref,_ = exact_budget(p[model_dev],dev_y,budget)
    candidates = {}
    for name, score in scores.items():
        equal, cutoff = strict_budget(score[model_dev],dev_y,budget)
        candidates[name] = dict(cutoff=cutoff, matched=comparison(dev_y,ref,equal))
    chosen = [n for n in candidates if n != "rule_priority"]
    selected = max(chosen,key=lambda n:(candidates[n]["matched"]["delta_tp"],n=="reference"))
    model_path = out/"models.joblib"
    joblib.dump(models,model_path,compress=3)
    write_json(out/"policy.json",dict(selected=selected, candidates=candidates, rule_columns=keys,
                   calibration_rows=ids[model_dev].tolist(), fit_rows=ids[model_fit].tolist(),
                   reference_sha256=checksum(reference/"dev_reference.csv"), model_sha256=checksum(model_path)))
    ids, p, base, raw, y, budget = rows["test"]
    scores = suite(p,raw,models)
    ref,_ = exact_budget(p,y,budget)
    frozen = p>candidates["reference"]["cutoff"]
    audit = dict(any_rule=counts(y,(raw>0).any(1)), per_rule={k:counts(y,raw[:,i]>0) for i,k in enumerate(keys)},
                 pass_errors={k:int((raw[:,i]<0).sum()) for i,k in enumerate(keys)})
    result = {}
    for name, score in scores.items():
        strict = score>candidates[name]["cutoff"]
        equal,_ = exact_budget(score,y,budget)
        separate = np.zeros(len(y),dtype=bool)
        base_each_project = np.zeros(len(y),dtype=bool)
        report = {}
        for prj in sorted(set(project[ids])):
            m = project[ids]==prj
            fp = counts(y[m],ref[m])["fp"]
            hit,_ = exact_budget(score[m],y[m],fp)
            local_reference,_ = exact_budget(p[m],y[m],fp)
            report[prj]=comparison(y[m],local_reference,hit)
            separate[m]=hit
            base_each_project[m]=local_reference
        result[name] = dict(matched=comparison(y,ref,equal), frozen=comparison(y,frozen,strict),
                            each_project=comparison(y,base_each_project,separate), by_project=report)
        with (out/f"{name}_test.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(("row","project","label","p_base","p","ref_equal","matched","ref_frozen",
                             "frozen","ref_each_project","matched_each_project"))
            writer.writerows(zip(ids,project[ids],y,p,score,ref.astype(int),equal.astype(int),
                                 frozen.astype(int),strict.astype(int),base_each_project.astype(int),
                                 separate.astype(int),strict=True))
    write_json(out/"results.json",dict(selected=selected, reference_budget=budget, rule_audit=audit, modes=result))
    print(rotation, "rule audit", audit["any_rule"], "selected", selected, flush=True)
    print(rotation, {k:(v["matched"]["delta_tp"],v["each_project"]["delta_tp"]) for k,v in result.items()},
          flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=HERE/"heldout_rules.csv")
    args=parser.parse_args()
    keys, ids, rules = rule_data(args.rules)
    for rotation in ROTATIONS:
        experiment(rotation,keys,ids,rules)
