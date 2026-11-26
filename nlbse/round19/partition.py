"""Lock two multi-project training rotations using the de-duplicated round 13 rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "round14"))
from audit import dataset, split
from project_split import checksum, fingerprint
from pipeline import TRAIN, write_json


def file_is_review(family, file):
    packed = f"NLBSE-R19-42\0{family}\0{file}".encode()
    return int.from_bytes(hashlib.sha256(packed).digest()[:8], "big") % 100 < 15


def prepare():
    texts, label, project, files = dataset()
    original = split()
    source = ROOT / "round13/partition.npz"
    raw = np.load(source)
    shared_fit, shared_dev = original["fit"], original["dev"]
    shared_fit_keys = {fingerprint(texts[i]) for i in shared_fit}
    shared_dev_keys = {fingerprint(texts[i]) for i in shared_dev}
    records = {}
    for held, other in (("test_a", "test_b"), ("test_b", "test_a")):
        test = original[held]
        excluded = set(project[test])
        pool = original[other]
        # The original Linux-only checkpoint saw neither test family nor R13 probe files.
        review = np.fromiter((file_is_review(project[i], files[i]) for i in pool), dtype=bool)
        keepout = {fingerprint(texts[i]) for i in pool[review]}
        fit_extra = np.asarray([i for i in pool[~review] if fingerprint(texts[i]) not in keepout], dtype=np.int32)
        fit = np.sort(np.r_[shared_fit, fit_extra])
        dev = np.sort(np.r_[shared_dev, pool[review]])
        if set(project[fit]) & excluded or set(project[dev]) & excluded:
            raise ValueError("A held-out project leaked into fit or calibration")
        train_files = {(project[i], files[i]) for i in fit}
        dev_files = {(project[i], files[i]) for i in dev}
        if train_files & dev_files or shared_fit_keys & (shared_dev_keys | keepout):
            raise ValueError("Training and calibration overlap")
        fingerprints_held = {fingerprint(texts[i]) for i in test}
        fingerprints_extra = {fingerprint(texts[i]) for i in fit_extra}
        if fingerprints_held & (shared_fit_keys | shared_dev_keys | keepout | fingerprints_extra):
            raise ValueError("The held-out project has training or calibration duplicates")
        if fingerprints_extra & (shared_dev_keys | keepout):
            raise ValueError("A same-project training copy matches calibration")
        if not np.all(np.isin(shared_fit, raw["fit"])):
            raise ValueError("Warm checkpoint may have a different fit set")
        path = HERE/f"{held}.npz"
        np.savez_compressed(path, fit=fit, dev=dev, test=test)
        records[held] = dict(heldout_projects=sorted(map(str, excluded)),
            partition_file=path.name, partition_sha256=checksum(path),
            removed_training_copies=int((~review).sum())-len(fit_extra),
            roles={name: dict(rows=len(ids), labels=label[ids].sum(0).tolist(),
                              projects=dict(Counter(map(str, project[ids]))))
                   for name, ids in (("fit", fit), ("dev", dev), ("test", test))})
    raw.close()
    result = dict(parent_sha256=checksum(source), train_sha256=checksum(TRAIN), rotations=records)
    write_json(HERE/"partition.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
