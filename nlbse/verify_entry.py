"""Exercise label-free inference and verify its output against the frozen public run."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "data/raw/test-00000-of-00001.parquet"
LABELS = ["vulnerability", "MAT"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    selection = json.loads((ROOT / "reports/round2_selection.json").read_text())
    winner = selection["winner"]
    checkpoint = ROOT / selection["candidates"][winner]["checkpoint"]
    thresholds = ROOT / "reports" / f"{winner}_frozen_thresholds.json"
    prior = ROOT / "runs" / (winner + "_packed_public") / "predictions.csv"

    frame = pq.ParquetFile(PUBLIC)
    texts = next(frame.iter_batches(batch_size=113, columns=["Function"])).column(0).to_pylist()
    positions = list(range(0, len(texts), 2)) + list(range(1, len(texts), 2))
    # Repeat a known row after rearranging the inputs to exercise order restoration.
    positions.append(0)
    with prior.open(newline="") as file:
        saved = list(csv.DictReader(file))
    with tempfile.TemporaryDirectory(prefix="nlbse-entry-") as tmpdir:
        data = Path(tmpdir) / "input.csv"
        output = Path(tmpdir) / "predictions.csv"
        with data.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Function"])
            writer.writerows([texts[i]] for i in positions)
        subprocess.run(
            [sys.executable, str(ROOT / "infer.py"), "--input", str(data), "--output", str(output),
             "--checkpoint", str(checkpoint), "--thresholds", str(thresholds),
             "--length", "512", "--bucket-window", "2048", "--device", args.device],
            check=True, env={**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"},
        )
        with output.open(newline="") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == LABELS
            predictions = [[int(r[k]) for k in LABELS] for r in reader]
        assert len(predictions) == len(positions)
        assert set(v for pair in predictions for v in pair) <= {0, 1}
        assert predictions[0] == predictions[-1]
        expected = [[int(saved[i][k]) for k in LABELS] for i in positions]
        differences = sum(a != b for a, b in zip(predictions, expected, strict=True))
        pred_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    with (checkpoint / "model.safetensors").open("rb") as file:
        model_sha = hashlib.file_digest(file, "sha256").hexdigest()
    receipt = dict(model=winner, device=args.device, rows=len(positions), columns=LABELS,
                   repeated_row_passed=True, input_order_matched_rows=len(positions)-differences,
                   rows_different_from_saved_GPU=differences, weights_sha256=model_sha,
                   predictions_sha256=pred_sha)
    (ROOT / "reports/round2_entry_smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
