"""Run the single-encoder policy pinned in the release manifest."""
import argparse
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    entry=json.loads((HERE/"entry.json").read_text())
    if entry["mode"]=="deep_exit_selected":
        from deep_exit import predict
        predict(args.input,args.output)
    else:
        from cascade import predict
        predict("small_lexical",args.input,args.output,entry["width"],entry.get("continuation"),
                shifted=entry.get("kernel")=="shift")


if __name__=="__main__":
    main()
