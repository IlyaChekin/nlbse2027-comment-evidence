"""Combine matched continuations in weight space and score a single encoder."""
import argparse
import time

import torch

import transfer
from transfer import HERE, ROTATIONS, SEEDS, checkpoint, checksum, json_at, write_json


EXTRA = HERE/"adversary"
OUT = HERE/"merged"
MERGES = ("seed_mean", "project_mean", "invariant_mean", "anchored")


def trained(root, rotation, seed, arm):
    folder = root/rotation/str(seed)
    entry = json_at(folder/"fit.json")["modes"][arm]
    return folder/entry["checkpoint"], entry["sha256"]


def load(source):
    path, digest = source
    if checksum(path) != digest:
        raise ValueError(f"Source checkpoint changed: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def mean_state(sources):
    state = None
    for source, weight in sources:
        current = load(source)
        if state is None:
            state = {name:(tensor.float()*weight if tensor.is_floating_point() else tensor.clone())
                     for name,tensor in current.items()}
            continue
        if state.keys() != current.keys():
            raise ValueError("Checkpoint keys differ")
        for name, tensor in current.items():
            if tensor.is_floating_point():
                state[name].add_(tensor.float(), alpha=weight)
            elif not torch.equal(state[name],tensor):
                raise ValueError("Non-floating state differs between continuations")
    return state


def uniform(seq, total=1.):
    return [(item,total/len(seq)) for item in seq]


def prepare(rotation, seed):
    folder = OUT/rotation/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    normal = [trained(HERE,rotation,s,"pooled") for s in SEEDS]
    group = normal+[trained(HERE,rotation,s,a) for s in SEEDS for a in ("balanced","robust")]
    inv = group+[trained(EXTRA,rotation,s,"invariant") for s in SEEDS]
    original = checkpoint("projects")
    plans = dict(seed_mean=uniform(normal), project_mean=uniform(group), invariant_mean=uniform(inv),
                 anchored=[(original,.40),*uniform(inv,.60)])
    record = {}
    for name, sources in plans.items():
        started = time.perf_counter()
        if abs(sum(w for _,w in sources)-1.)>1e-8:
            raise ValueError("Checkpoint mixture has incorrect total weight")
        model = mean_state(sources)
        target = folder / f"{name}.pt"
        temporary = target.with_suffix(".part")
        torch.save(model,temporary)
        temporary.replace(target)
        record[name] = dict(checkpoint=target.name, sha256=checksum(target),
                            seconds=time.perf_counter()-started,
                            sources=[dict(file=str(path.relative_to(HERE.parent)), sha256=digest,weight=w)
                                     for (path,digest),w in sources])
        print(rotation,name,record[name]["seconds"],flush=True)
    write_json(folder/"fit.json",dict(config=dict(seed=seed),modes=record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rotation",choices=ROTATIONS)
    parser.add_argument("--score",action="store_true")
    args = parser.parse_args()
    if args.score:
        transfer.RESULTS = OUT
        transfer.project_scores(args.rotation,42,MERGES)
    else:
        prepare(args.rotation,42)
