"""Export length-delimited functions for the C++ scorer and verify fit-only labels."""
import argparse
from pathlib import Path
from struct import Struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pipeline import PUBLIC, TRAIN, load, select_indices

DIRECTORY = HERE / "neighbors"
LENGTH = Struct("<I")


def export(path, texts, indices, labels=None):
    pending = path.with_suffix(".part")
    with pending.open("wb") as stream:
        for i in indices:
            code = texts[i].encode("utf-8")
            stream.write(LENGTH.pack(len(code)))
            stream.write(code)
            stream.write(bytes([0 if labels is None else labels[i, 0]]))
    pending.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("holdout", "public"))
    args = ap.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.stage == "holdout":
        train, labels, _ = load(TRAIN)
        fit, holdout = select_indices(False)
        export(DIRECTORY/"fit.bin", train, fit, labels)
        export(DIRECTORY/"holdout.bin", train, holdout)
    else:
        public, _, _ = load(PUBLIC)
        export(DIRECTORY/"public.bin", public, range(len(public)))


if __name__ == "__main__":
    main()
