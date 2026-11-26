"""Run the domain-adversarial arm and merge checkpoints on both project rotations."""
import argparse

import merge
import transfer
from transfer import ADVERSARY, ROTATIONS, SEEDS


def project(rotation):
    transfer.RESULTS = merge.EXTRA
    for seed in SEEDS:
        folder = transfer.RESULTS / rotation / str(seed)
        if not (folder / "fit.json").exists():
            transfer.fit(rotation, seed, (ADVERSARY,))
        if not (folder / "results.json").exists():
            transfer.project_scores(rotation, seed, (ADVERSARY,))
    transfer.prepared_training.cache_clear()
    transfer.RESULTS = merge.OUT
    folder = transfer.RESULTS / rotation / "42"
    if not (folder / "fit.json").exists():
        merge.prepare(rotation,42)
    if not (folder / "results.json").exists():
        transfer.project_scores(rotation,42,merge.MERGES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rotation",choices=ROTATIONS)
    args = parser.parse_args()
    project(args.rotation)
