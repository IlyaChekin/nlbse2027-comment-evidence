"""Assemble a verified out-of-fold score for every training function."""

import json

import numpy as np

from crossfit import HERE, N_FOLDS, FOLDS, PLAN, TRAIN, checksum, load, write_json


OUT = HERE / "mining"


def main():
    manifest = json.loads(PLAN.read_text())
    if checksum(FOLDS) != manifest["partition_sha256"] or checksum(TRAIN) != manifest["input_sha256"]:
        raise ValueError("Fold source inputs changed")
    _, labels, _ = load(TRAIN)
    scores, tested, fold_column = {name: [] for name in manifest["families"]}, [], []
    source_digests = {}
    with np.load(FOLDS) as f:
        for fold in range(N_FOLDS):
            ids = f[f"held_{fold}"]
            if np.intersect1d(ids, f[f"fit_{fold}"]).size:
                raise ValueError(f"Overlapping rows in fold {fold}")
            tested.append(ids)
            fold_column.append(np.full(len(ids),fold,dtype=np.int8))
            for name in scores:
                prefix = HERE/f"fold_{fold}"/name
                path = prefix/"test.npz"
                expected = json.loads((prefix/"test.json").read_text())["sha256"]
                split = json.loads((prefix/"config.json").read_text())
                if checksum(path) != expected or split["partition_sha256"] != manifest["partition_sha256"]:
                    raise ValueError(f"Unverified model output {fold}/{name}")
                with np.load(path) as r:
                    if not np.array_equal(r["ids"],ids) or not np.array_equal(r["y"],labels[ids]):
                        raise ValueError(f"Wrong labels or rows in {fold}/{name}")
                    p = r["probabilities"]
                    if p.shape != labels[ids].shape or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
                        raise ValueError(f"Invalid probabilities in {fold}/{name}")
                    scores[name].append(p)
                source_digests[f"{fold}/{name}"] = expected
    ids = np.concatenate(tested)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Multiple predictions for the same training row")
    idx = np.argsort(ids)
    ids = ids[idx]
    predictions = {name:np.concatenate(parts)[idx] for name,parts in scores.items()}
    y = labels[ids]
    both = np.stack([p[:,0] for p in predictions.values()])
    masks = {"fp_80":(both.min(0)>.8)&(y[:,0]==0),"fp_90":(both.min(0)>.9)&(y[:,0]==0),
             "fp_95":(both.min(0)>.95)&(y[:,0]==0),"fn_05":(both.max(0)<.05)&(y[:,0]==1)}
    OUT.mkdir(exist_ok=True)
    pending = OUT/"scores.part.npz"
    artifact = OUT/"scores.npz"
    folds = np.concatenate(fold_column)[idx]
    np.savez_compressed(pending,ids=ids,y=y,fold=folds,**predictions,**masks)
    pending.replace(artifact)
    write_json(OUT/"manifest.json",dict(rows=len(ids),sha256=checksum(artifact),input_sha256=manifest["input_sha256"],
                  partition_sha256=manifest["partition_sha256"],fold_score_sha256=source_digests,
                  counts={k:int(v.sum()) for k,v in masks.items()},
                  counts_by_fold={str(f):{k:int((v&(folds==f)).sum()) for k,v in masks.items()} for f in range(N_FOLDS)}))
    print((OUT/"manifest.json").read_text(),flush=True)


if __name__=="__main__":
    main()
