"""Package the measured MAT-only change over the verified round 20 archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_SHA = "0ca2a4076ba974485e6e279fdf36f866728caddc8eea49edbf4575f61a2a1cb5"
ENTRY = "nlbse/round21/mat/predict_mat.py"
FACTOR_ENTRY = "nlbse/round20/predict_factor.py"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prediction_bytes(path):
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(("row", "pred_vulnerability", "pred_MAT"))
    with path.open(newline="") as file:
        writer.writerows((r["row"], r["pred_vulnerability"], r["pred_MAT"])
                         for r in csv.DictReader(file))
    return table.getvalue().encode()


def build(base, output):
    record = json.loads((HERE / "profile.json").read_text())
    before, after = record["reference"], record["marker"]
    data = HERE / "profile_marker.csv"
    if digest(base) != BASE_SHA or output.resolve() == base.resolve():
        raise ValueError("Expected the unchanged round 20 archive and a separate output")
    required_hashes = {
        data: after["predictions_sha256"],
        HERE / "marker.py": record["provenance"]["marker_sha256"],
        HERE / "annotation_policy.json": record["provenance"]["audit_policy_sha256"],
    }
    if any(digest(path) != expected for path, expected in required_hashes.items()):
        raise ValueError("The measured policy, implementation, or predictions changed")
    if after["score"] <= before["score"]:
        raise ValueError("A slower MAT variant cannot replace the paired control")
    temp = output.with_suffix(".part")
    quality = after["quality"]
    with ZipFile(base) as old, ZipFile(temp, "w", ZIP_DEFLATED) as new:
        contents = {
            FACTOR_ENTRY: ROOT / "round20/predict_factor.py",
            ENTRY: HERE / "predict_mat.py",
            "nlbse/round21/mat/marker.py": HERE / "marker.py",
            "nlbse/round21/mat/profile.json": HERE / "profile.json",
            "nlbse/round21/mat/annotation_policy.json": HERE / "annotation_policy.json",
        }
        replaced = set(contents) | {"README.md", "examples/public_expected.csv"}
        for item in old.infolist():
            if item.filename not in replaced:
                new.writestr(item, old.read(item))
        for name, path in contents.items():
            new.write(path, name)
        new.writestr("examples/public_expected.csv", prediction_bytes(data))
        new.writestr("README.md", f"""# NLBSE 2027 · conditional rule-guided classifier

One compact encoder predicts vulnerability and auxiliary rule events.
Selective static checks refine the vulnerability decision.
MAT uses whole-word tags and also applies the published annotation's
marker normalization to lexically identified comments in the supplied function.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH. The timings used six CPU scanner workers.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Input is a CSV with a Function column. Output columns are vulnerability,MAT.
The paired local score on 30,655 public functions is {after["score"]:.6f}.
Vulnerability F1 is {quality["vulnerability"]["f1"]:.5f}, MAT F1 is
{quality["MAT"]["f1"]:.5f}. The RTX 4090 median elapsed time including rule checks is
{after["cost"]["wall_seconds"]:.2f} s. The organisers evaluate the hidden set.
The precise public profile and the marker selection policy are included.
Static weakness rule provenance is in nlbse/round19/weak_satd/RULE_SOURCE.md.
The MAT marker semantics were independently implemented from the MADE-WIC
replication scripts at doi:10.5281/zenodo.12567874.
""")
    temp.replace(output)
    sha = digest(output)
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(sha, output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "submission/NLBSE2027_round20_factor_tool.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "submission/NLBSE2027_round21_mat_tool.zip")
    args = parser.parse_args()
    build(args.base, args.output)
