"""Run the single compact checkpoint pinned in entry.json."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    entry=json.loads((HERE/"entry.json").read_text())
    if entry["backend"]=="reuse_tags":
        from fast_mat import predict
    else:
        from efficient_aug import predict
    predict(args.input,args.output)


if __name__=="__main__":
    main()
