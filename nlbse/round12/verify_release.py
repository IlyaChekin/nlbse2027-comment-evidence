"""Verify the archived CLI against every stored public binary decision."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pyarrow.csv
import pyarrow.parquet


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decisions(path, annotated=False):
    prefix = "pred_" if annotated else ""
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        if not annotated and rows.fieldnames != ["vulnerability", "MAT"]:
            raise ValueError("The release writes unexpected column names")
        output = [(row[prefix+"vulnerability"], row[prefix+"MAT"]) for row in rows]
    if any(bit not in {"0", "1"} for pair in output for bit in pair):
        raise ValueError("The release writes nonbinary values")
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("development_parquet", type=Path)
    ap.add_argument("receipt_directory", type=Path)
    ap.add_argument("--entry",default="nlbse/round12/predict.py")
    ap.add_argument("--cuda-launch",action="store_true")
    args = ap.parse_args()
    receipt = args.receipt_directory.resolve()
    receipt.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="nlbse_release_check_") as temp:
        clean = Path(temp)
        with ZipFile(args.archive) as source:
            weights = [n for n in source.namelist() if n.endswith((".pt", ".safetensors", "pytorch_model.bin"))]
            if len(weights) != 1:
                raise ValueError("Expected exactly one encoder checkpoint in the release")
            source.extractall(clean)
        functions = pyarrow.parquet.read_table(args.development_parquet, columns=["Function"])
        input_csv = clean / "functions.csv"
        pyarrow.csv.write_csv(functions, input_csv)
        predictions = receipt / "release_predictions.csv"
        launcher=["nlbse/cuda_launch.py"] if args.cuda_launch else []
        command = [sys.executable, *launcher, args.entry, "--input", str(input_csv),
                   "--output", str(predictions)]
        run = subprocess.run(command, cwd=clean, text=True, capture_output=True, check=False)
        (receipt / "cli.log").write_text(run.stdout + run.stderr)
        run.check_returncode()
        expected = decisions(clean / "examples/public_expected.csv", annotated=True)
        actual = decisions(predictions)
        if actual != expected or len(actual) != functions.num_rows:
            raise ValueError("Release decisions do not exactly match the retained public predictions")
        record = dict(rows=len(actual), identical_binary_decisions=2 * len(actual),
                      archive_sha256=digest(args.archive), public_sha256=digest(args.development_parquet),
                      checkpoint_sha256=digest(clean / weights[0]), predictions_sha256=digest(predictions))
        (receipt / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
