"""Predict the ordered vulnerability and SATD labels from a Function CSV."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from pipeline import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(256, 512), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bucket-window", type=int, default=0,
                        help="Group functions of similar token lengths within each window.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bucket_window < 0 or args.bucket_window % args.batch_size:
        parser.error("Use a positive batch size and a bucket window that is a multiple of it.")
    with args.input.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "Function" not in (reader.fieldnames or ()):
            parser.error("Expected a Function column.")
        texts = [row["Function"] for row in reader]
    if not texts or any(text is None for text in texts):
        parser.error("Expected one or more function strings.")
    threshold = np.asarray(json.loads(args.thresholds.read_text()), dtype=float)
    if threshold.shape != (2,) or not np.isfinite(threshold).all() or not ((threshold >= 0) & (threshold <= 1)).all():
        parser.error("Expected two probability thresholds in [0, 1].")
    model = RobertaForSequenceClassification.from_pretrained(
        args.checkpoint, local_files_only=True, attn_implementation="eager").to(args.device)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.checkpoint, local_files_only=True)
    probabilities, _ = predict(model, tokenizer, texts, list(range(len(texts))),
                               dict(length=args.length, batch_size=args.batch_size, fixed=False,
                                    profile_reps=0, bucket_window=args.bucket_window))
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("vulnerability", "MAT"))
        writer.writerows((probabilities > threshold).astype(int))


if __name__ == "__main__":
    main()
