"""Store just the teacher targets used for distillation, with row identities."""

import numpy as np
from students import ACTIONS, HERE, ROOT, sha256, write_json


def main():
    source = ROOT / "round9/cache/fit.npz"
    destination = HERE / "teacher.npz"
    action = ACTIONS[1]
    with np.load(source) as cache:
        values = {name: cache[name] for name in ("ids", "y", action)}
    pending = destination.with_suffix(".part.npz")
    np.savez_compressed(pending, **values)
    with np.load(pending) as saved:
        if not all(np.array_equal(values[k], saved[k]) for k in values):
            raise ValueError("Export changed the teacher targets")
    pending.replace(destination)
    write_json(HERE / "teacher.json", dict(sha256=sha256(destination), source_sha256=sha256(source),
               rows=len(values["ids"]), policy=action))


if __name__ == "__main__":
    main()
