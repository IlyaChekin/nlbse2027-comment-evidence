"""Train a new classifier with entire repository families held out."""
from collections import Counter
import hashlib
import time

import numpy as np
import torch

from common import HERE, SEED, TRAIN, export, load, stratified, write_json
from neural import Config, encode, model_for, train

OUT = HERE / "project"
HELDOUT = {"chromium", "qemu", "ffmpeg"}
ALIASES = {"chrome": "chromium", "chromium": "chromium", "linux kernel": "linux",
           "linux": "linux", "php-src": "php", "php": "php"}


def family(name):
    name = name.strip().casefold()
    return ALIASES.get(name, name)


def split_project(texts, gold, fields):
    groups = [family(name) for name in fields["Projectname"]]
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    test = np.asarray([i for i, k in enumerate(groups) if k in HELDOUT], dtype=np.int32)
    blocked = {hashes[i] for i in test}
    eligible = [i for i,k in enumerate(groups) if k not in HELDOUT and hashes[i] not in blocked]
    cal_seed = stratified(eligible, gold, 3500, SEED+1)
    calibration_hashes = {hashes[i] for i in cal_seed}
    cal = np.asarray([i for i in eligible if hashes[i] in calibration_hashes], dtype=np.int32)
    fit = stratified([i for i in eligible if hashes[i] not in calibration_hashes], gold, 40000)
    return fit, cal, test, groups


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    texts, y, fields = load(TRAIN, audit=True)
    fit, cal, test, groups = split_project(texts, y, fields)
    test_hashes = {hashlib.sha256(texts[i].encode()).hexdigest() for i in test}
    fit_hashes = {hashlib.sha256(texts[i].encode()).hexdigest() for i in fit}
    cal_hashes = {hashlib.sha256(texts[i].encode()).hexdigest() for i in cal}
    fit_groups = {groups[i] for i in fit}
    cal_groups = {groups[i] for i in cal}
    if any((fit_groups & HELDOUT, cal_groups & HELDOUT, test_hashes & fit_hashes,
            test_hashes & cal_hashes, fit_hashes & cal_hashes)):
        raise ValueError("The project or exact-function holdout overlaps with training")
    np.savez_compressed(OUT/"split.npz", fit=fit, calibration=cal, heldout=test)
    by_part = {k: dict(rows=len(v), label_positives=y[v].sum(axis=0).tolist(),
                       families=dict(Counter(groups[i] for i in v)))
               for k,v in (("training", fit), ("calibration", cal), ("heldout", test))}
    exclusions = sum(groups[i] not in HELDOUT and hashlib.sha256(texts[i].encode()).hexdigest() in test_hashes
                     for i in range(len(texts)))
    write_json(OUT/"split.json", dict(splits=by_part, heldout_families=sorted(HELDOUT),
                                     exact_heldout_duplicates_excluded=exclusions))
    print("Project split", by_part, flush=True)
    model, tokenizer, pad, layers = model_for("codet5p")
    config = Config("codet5p", 512, 32, 2, 3e-5)
    start = time.perf_counter()
    inputs = encode([texts[i] for i in fit], tokenizer, config.cap)
    print("Project fit tokenized", len(inputs), "rows in", round(time.perf_counter()-start, 2), flush=True)
    model = model.cuda()
    cost = train(model, config, inputs, pad, y[fit])
    checkpoint = export(model, tokenizer, OUT/"checkpoint")
    write_json(OUT/"training.json", dict(cost=cost, checkpoint=checkpoint, retained_layers=layers,
                                        initialized_from="original Salesforce/codet5p-220m",
                                        config=vars(config)))


if __name__ == "__main__":
    main()
