"""Ship the new scan path after the paired run proves an improvement."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENTRY = "nlbse/round22/predict_union.py"
BASE = ROOT/"submission/NLBSE2027_round21_mat_tool.zip"
NEW = ROOT/"submission/NLBSE2027_round22_union_tool.zip"
BASE_SHA = "d35995eae113c47dbbd1ee9d0bd137f588755362397a41afe36bb919d79a664e"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def build(base, output):
    if digest(base)!=BASE_SHA or output.resolve()==base.resolve():
        raise ValueError("Expected the unchanged round 21 archive and a separate output file")
    results = HERE/"union_public/results.json"
    measure = json.loads(results.read_text())
    source = json.loads((HERE/"union_verification.json").read_text())
    winner = measure["stats"]["short_circuit"]
    old = measure["stats"]["reference"]
    if winner["score"]<=old["score"] or source["differences"]:
        raise ValueError("Release requires both exactness and a measured improvement")
    program = HERE/"scan_union.mjs"
    if not source["program_sha256"]==measure["program_sha256"]==digest(program):
        raise ValueError("The exact early-exit scanner changed")
    tmp = output.with_suffix(".part")
    quality = measure["quality"]
    with ZipFile(base) as before,ZipFile(tmp,"w",ZIP_DEFLATED) as after:
        for item in before.infolist():
            if item.filename!="README.md":
                after.writestr(item,before.read(item))
        for name in ("predict_union.py","scan_union.mjs","union_scanner.json","union_verification.json","union_public/results.json"):
            after.write(HERE/name,"nlbse/round22/"+name)
        after.writestr("README.md",f"""# NLBSE 2027 · conditional classifier with exact early-exit checks

The compact model estimates vulnerability separately for functions that do and
do not satisfy a WeakSATD rule. The acquisition policy requests an exact check
for selected functions. The checker returns on the first hit. A negative or
indeterminate answer still considers every rule. The MAT branch uses normalized
markers in the supplied comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH. The profile used six scanner workers.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Input is a CSV with a Function column. Output columns are vulnerability,MAT.
The score is {winner["score"]:.6f} in a paired local RTX 4090 measurement
against {old["score"]:.6f} for the full scan on 30,655 public functions.
Mean F1 is {quality["mean"]["f1"]:.5f} and all binary decisions agree.
Rule provenance is in nlbse/round19/weak_satd/RULE_SOURCE.md.
""")
    tmp.replace(output)
    sha = digest(output)
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(sha,output,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=BASE)
    p.add_argument("--output",type=Path,default=NEW)
    a=p.parse_args()
    build(a.base,a.output)
