"""Assemble a clean, single-checkpoint inference archive from the verified base package."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE_SHA="d147f4dce703425746fdc4a340e18a17e72fbf8bbd3e2a3c6a96b672e0a03223"
OLD_ENCODE="return small.encode(rows,tok),features"
NEW_ENCODE='return getattr(pipe,"encode",small.encode)(rows,tok),features'

README="""# NLBSE 2027 · compact encoder with full function context

The compact encoder reads the complete function and selectively applies a second identifier-normalized view with the same checkpoint. The MAT decision uses the original full-comment indicators. The checkpoint digest and operating thresholds are recorded in `nlbse/round16/entry.json`.

Install Python 3.11 or 3.12 and the CUDA build of PyTorch 2.8. The input CSV has a `Function` column. Predictions contain `vulnerability,MAT` in input order.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 nlbse/round16/predict.py --input functions.csv --output predictions.csv
```

The full public predictions and the profiled results are retained in `examples/public_expected.csv` and `nlbse/round16/public_profile.json`. The score is a local validation measurement. The organizers evaluate the competition on the hidden test set.
"""


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def expected(path):
    result=io.StringIO(newline="")
    w=csv.writer(result)
    w.writerow(("row","pred_vulnerability","pred_MAT"))
    i=-1
    with path.open(newline="") as stream:
        for i,row in enumerate(csv.DictReader(stream)):
            if int(row["row"])!=i or any(row[key] not in ("0","1") for key in ("vulnerability","mat")):
                raise ValueError("Malformed measured predictions")
            w.writerow((i,row["vulnerability"],row["mat"]))
    if i+1!=30655:
        raise ValueError("Incomplete public decisions")
    return result.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=ROOT/"submission/NLBSE2027_round12_tool.zip")
    p.add_argument("--output",type=Path,default=ROOT/"submission/NLBSE2027_round16_tool.zip")
    args=p.parse_args()
    if args.base.resolve()==args.output.resolve() or digest(args.base)!=BASE_SHA:
        raise ValueError("Expected the verified round 12 archive at a separate path")
    folder=HERE/"context_public"
    policy=read(folder/"policy.json")
    profile=read(folder/"results.json")
    if policy["selected"]!="full" or "full" not in profile:
        raise ValueError("The measured official configuration did not select full context")
    source=ROOT/"round15/official_distill"
    pinned=read(source/"fit.json")["variants"]["original_teacher"]
    weight=source/pinned["checkpoint"]
    if digest(weight)!=pinned["sha256"]:
        raise ValueError("The calibrated encoder changed")
    entry=dict(checkpoint="encoder.pt",checkpoint_sha256=pinned["sha256"],width=256,
               route=policy["route"],vulnerability_threshold=policy["modes"]["full"]["cutoff"],
               local_score=profile["full"]["score"])
    out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    pending=out.with_suffix(".part")
    with ZipFile(args.base) as old,ZipFile(pending,"w",ZIP_DEFLATED) as new:
        for member in old.infolist():
            name=member.filename
            if name in {"README.md","examples/public_expected.csv"} or name.endswith((".pt", ".safetensors", "pytorch_model.bin")):
                continue
            data=old.read(member)
            if name=="nlbse/round12/fast_mat.py":
                decoded=data.decode()
                if decoded.count(OLD_ENCODE)!=1:
                    raise ValueError("Unexpected compact inference interface in source archive")
                data=decoded.replace(OLD_ENCODE,NEW_ENCODE).encode()
            new.writestr(member,data)
        new.writestr("README.md",README)
        new.writestr("examples/public_expected.csv",expected(folder/"full_public.csv"))
        new.writestr("nlbse/round16/entry.json",json.dumps(entry,indent=2)+"\n")
        new.write(folder/"results.json","nlbse/round16/public_profile.json")
        new.write(HERE/"predict.py","nlbse/round16/predict.py")
        new.write(weight,"nlbse/round16/encoder.pt")
    pending.replace(out)
    print(json.dumps(dict(archive=str(out),archive_sha256=digest(out),**entry),indent=2),flush=True)


if __name__=="__main__":
    main()
