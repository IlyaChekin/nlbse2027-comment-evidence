"""Archive the frozen section correction after its paired benchmark."""
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
BASE=ROOT/"submission/NLBSE2027_round24_sparse_tool.zip"
OUT=ROOT/"submission/NLBSE2027_round25_sections_tool.zip"
BASE_HASH="e8ac0bfff7c2cc8038a899502028dad3f6ca9bf053586f03adfb8056e52d4989"
ENTRY="nlbse/round25/predict_sections.py"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def expected(path):
    target=io.StringIO(newline="")
    w=csv.writer(target)
    w.writerow(("row","pred_vulnerability","pred_MAT"))
    with path.open(newline="") as source:
        w.writerows((r["row"],r["pred_vulnerability"],r["pred_MAT"]) for r in csv.DictReader(source))
    return target.getvalue()


def build(before,after):
    if sha(before)!=BASE_HASH or before.resolve()==after.resolve():
        raise ValueError("Expected the verified Round 24 archive and a new output")
    pub=HERE/"views/public"
    result=json.loads((pub/"results.json").read_text())
    name=result["selected"]
    read=result["results"][name]
    ref=result["results"]["reference"]
    if read["score"]<=ref["score"] or read["quality"]["mean"]["f1"]<=ref["quality"]["mean"]["f1"]:
        raise ValueError("The section correction did not improve the paired result")
    calibration=pub/"policy.json"
    info=json.loads(calibration.read_text())
    if name!=info["selected"] or sha(calibration)!=result["policy_sha256"]:
        raise ValueError("Public comparison does not use the frozen calibration")
    weights=HERE/"views/official/calibrator.joblib"
    if sha(weights)!=info["calibrator_sha256"]:
        raise ValueError("Frozen calibrator changed")
    option=info["support"][name]
    chosen=joblib.load(weights)[option["region"]]
    p=pub/f"{name}.csv"
    if sha(p)!=read["predictions_sha256"]:
        raise ValueError("Saved predictions changed")
    tmp=after.with_suffix(".part")
    with TemporaryDirectory(prefix="nlbse_sections_") as folder:
        head=Path(folder)/"section_head.joblib"
        joblib.dump(chosen,head,compress=3)
        setup=dict(option=option,length_cutoff=info["length_cutoff"],head_sha256=sha(head),
                   scanner_sha256=result["scanner_sha256"])
        with ZipFile(before) as old,ZipFile(tmp,"w",ZIP_DEFLATED) as dst:
            for item in old.infolist():
                if item.filename not in ("README.md","examples/public_expected.csv"):
                    dst.writestr(item,old.read(item))
            for file in ("predict_sections.py","section_runtime.py"):
                dst.write(HERE/file,"nlbse/round25/"+file)
            dst.write(head,"nlbse/round25/section_head.joblib")
            dst.write(calibration,"nlbse/round25/calibration.json")
            dst.write(pub/"results.json","nlbse/round25/public_profile.json")
            dst.writestr("nlbse/round25/entry.json",json.dumps(setup,indent=2)+"\n")
            dst.writestr("examples/public_expected.csv",expected(p))
            dst.writestr("README.md",f"""# NLBSE 2027 · compact conditional code classifier

One compact encoder predicts vulnerability and MAT from each Function.
The vulnerability branch uses early-exit exact checks and separately
calibrated sparse evidence. Code signatures and bodies receive a separate
score near the threshold for the exact-negative cohort.
MAT uses normalized markers in the supplied comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH. The benchmark used six scan workers.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

The output columns are vulnerability,MAT.
Local paired score {read["score"]:.6f} versus {ref["score"]:.6f}
on all 30,655 public functions. Mean F1 {read["quality"]["mean"]["f1"]:.5f}.
The median elapsed time is {read["cost"]["wall_seconds"]:.2f} s on RTX 4090.
Timings and internal calibration are included. The organisers evaluate the hidden set.
""")
    tmp.replace(after)
    digest=sha(after)
    after.with_suffix(".sha256").write_text(f"{digest}  {after.name}\n")
    print(digest,after,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=BASE)
    ap.add_argument("--output",type=Path,default=OUT)
    a=ap.parse_args()
    build(a.base,a.output)
