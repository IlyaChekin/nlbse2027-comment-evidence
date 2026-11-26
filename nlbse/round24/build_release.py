"""Package the sparse expert if its frozen public comparison improves the score."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED,ZipFile

import joblib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/"submission/NLBSE2027_round22_union_tool.zip"
OUTPUT=ROOT/"submission/NLBSE2027_round24_sparse_tool.zip"
BASE_SHA="f66c19f420f0ff87230be04dd4034a82dbd24ab20a50ebf0638963f4c77b598f"
ENTRY="nlbse/round24/predict_sparse.py"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def predictions(path):
    buffer=io.StringIO(newline="")
    w=csv.writer(buffer)
    w.writerow(("row","pred_vulnerability","pred_MAT"))
    with path.open(newline="") as stream:
        w.writerows((r["row"],r["pred_vulnerability"],r["pred_MAT"]) for r in csv.DictReader(stream))
    return buffer.getvalue().encode()


def build(base,output):
    if digest(base)!=BASE_SHA or output.resolve()==base.resolve():
        raise ValueError("Expected the unchanged round 22 archive and a separate output path")
    profile=HERE/"public/results.json"
    report=json.loads(profile.read_text())
    key=report["selected"]
    internal=HERE/"official/tuned"
    p=internal/"policy.json"
    weights=internal/"models.joblib"
    policy=json.loads(p.read_text())
    if key!=policy["selected"] or report["internal_policy_sha256"]!=digest(p) or policy["models_sha256"]!=digest(weights):
        raise ValueError("Calibration or public selection changed")
    winner=report["results"][key]
    old=report["results"]["reference"]
    if winner["score"]<=old["score"]:
        raise ValueError("The sparse expert did not pass the paired public comparison")
    path=HERE/"public"/f"{key.replace(':','_')}.csv"
    if digest(path)!=winner["predictions_sha256"]:
        raise ValueError("Public predictions were modified")
    variant=policy["candidates"][key]
    state=joblib.load(weights)[variant["mode"]]
    temp=output.with_suffix(".part")
    with TemporaryDirectory(prefix="sparse_release_") as work:
        packed=Path(work)/"correction.joblib"
        joblib.dump(state,packed,compress=3)
        entry=dict(variant=variant,weights_sha256=digest(packed),scanner_sha256=report["scanner_sha256"])
        with ZipFile(base) as before,ZipFile(temp,"w",ZIP_DEFLATED) as after:
            for item in before.infolist():
                if item.filename not in ("README.md","examples/public_expected.csv"):
                    after.writestr(item,before.read(item))
            for fname in ("predict_sparse.py","fuse_sparse.py","sparse_model.py"):
                after.write(HERE/fname,"nlbse/round24/"+fname)
            after.write(p,"nlbse/round24/calibration_policy.json")
            after.write(profile,"nlbse/round24/public_profile.json")
            after.write(packed,"nlbse/round24/correction.joblib")
            after.writestr("nlbse/round24/entry.json",json.dumps(entry,indent=2)+"\n")
            after.writestr("examples/public_expected.csv",predictions(path))
            after.writestr("README.md",f"""# NLBSE 2027 · conditional classifier with a sparse correction

One compact model predicts vulnerability and MAT from each Function.
The vulnerability output combines a selective exact rule check with a small
sparse ranker. The ranker is calibrated on file-disjoint development rows
and uses a separately frozen threshold in its selected cohort.
MAT uses normalized tags in the supplied comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH. Timings used six scan workers.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Input is a CSV with a Function column. Output columns are vulnerability,MAT.
Local paired score {winner["score"]:.6f} versus {old["score"]:.6f}
on all 30,655 public functions. Mean F1 {winner["quality"]["mean"]["f1"]:.5f}.
The median elapsed time is {winner["cost"]["wall_seconds"]:.2f} s on RTX 4090.
Detailed timings and calibration are included. The organisers evaluate the hidden set.
""")
    temp.replace(output)
    sha=digest(output)
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(sha,output,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=BASE)
    p.add_argument("--output",type=Path,default=OUTPUT)
    a=p.parse_args()
    build(a.base,a.output)
