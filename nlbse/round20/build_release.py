"""Package the best profiled rule subset using the verified round 19 release."""
import argparse
from copy import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import joblib


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SUBMISSION=ROOT/"submission"
BASE_SHA="9a2cc872fcfed02ab0da6d0501a8cde4fb55f14c9d880d158e4f4420146d43a7"
PREFIX="nlbse/round20/"
RUNTIME="nlbse/round19/weak_satd/runtime.py"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def json_at(path):
    return json.loads(path.read_text())


def expected(path):
    stream=io.StringIO(newline="")
    writer=csv.writer(stream)
    writer.writerow(("row","pred_vulnerability","pred_MAT"))
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=30655 or any(int(r["row"])!=i for i,r in enumerate(rows)):
        raise ValueError("Frozen public predictions are incomplete or reordered")
    writer.writerows((i,r["vulnerability"],r["mat"]) for i,r in enumerate(rows))
    return stream.getvalue()


def readme(score,vulnerability,mat,wall):
    return f"""# NLBSE 2027 · compact encoder and selective WeakSATD rules

Input is a CSV with a Function column. Output is vulnerability,MAT in input order.
The exact rule subset and operating thresholds are frozen in nlbse/round20/entry.json.
The single compact encoder reads the complete function and optionally rereads a
canonical identifier view. The MAT decision uses the full comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8, and Node.js 22. A Node outside
PATH can be selected with NLBSE_NODE. The reported run used 6 CPU workers.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 nlbse/round20/predict.py --input functions.csv --output predictions.csv
```

The measured score on all 30,655 public functions was {score:.6f}.
The vulnerability and MAT F1 were {vulnerability:.5f} and {mat:.5f}.
Median end-to-end time on the recorded RTX 4090 host was {wall:.2f} s.
The rule processing time is included. The organizers rank submissions on their
hidden set. For rule provenance see nlbse/round19/weak_satd/RULE_SOURCE.md.
Round 20 uses a literal name index for the original static matcher wherever
possible. The optional numeric rule handles bounded literal syntax without
running any expressions from the input.
"""


def build(base,output,variant):
    if digest(base)!=BASE_SHA or base.resolve()==output.resolve():
        raise ValueError("Expected the unchanged round 19 release and a separate output")
    storage=HERE/variant/"official"
    rule=json_at(storage/"policy.json")
    winner=rule["selected"]
    record=json_at(storage/"results.json")
    quality=record[winner]["quality"]
    if winner=="reference" or record[winner]["score"]<=record["reference"]["score"]:
        raise ValueError("This variant did not improve its matched public measurement")
    spec=rule["modes"][winner]
    program=None
    if variant!="subsets":
        program="scan_extended.mjs" if variant=="subsets_numeric" else "scan.mjs"
        if digest(HERE/program)!=rule["program_sha256"] or digest(HERE/"scan.mjs")!=rule["scanner_sha256"]:
            raise ValueError("Fast scanner differs from the measured version")
    head=joblib.load(storage/"heads.joblib")[spec["head"]]
    serialized=io.BytesIO()
    joblib.dump(head,serialized,compress=3)
    payload=serialized.getvalue()
    temp=output.with_suffix(".part")
    with ZipFile(base) as old,ZipFile(temp,"w",ZIP_DEFLATED) as new:
        context=json.loads(old.read("nlbse/round19/weak_satd/entry.json"))
        encoder=json.loads(old.read("nlbse/round16/entry.json"))
        for file in old.infolist():
            if file.filename not in ("README.md","examples/public_expected.csv",RUNTIME):
                new.writestr(copy(file),old.read(file))
        cfg=dict(columns=[rule["columns"][i] for i in spec["columns"]],
                 gate=[spec["spec"]["left"],spec["spec"]["right"]],
                 threshold=spec["threshold"],route=encoder["route"],
                 checkpoint_sha256=context["checkpoint_sha256"],source_sha256=context["rule_source_sha256"],
                 program=program,program_sha256=rule.get("program_sha256"),
                 scanner_sha256=rule.get("scanner_sha256") or context["scanner_sha256"],
                 head_sha256=hashlib.sha256(payload).hexdigest(),worker_limit=context["worker_limit"],
                 score=record[winner]["score"],name=winner)
        new.writestr("README.md",readme(record[winner]["score"],quality["vulnerability"]["f1"],
                     quality["MAT"]["f1"],record[winner]["cost"]["wall_seconds"]))
        new.writestr("examples/public_expected.csv",expected(storage/f"{winner}_public.csv"))
        new.writestr(PREFIX+"entry.json",json.dumps(cfg,indent=2)+"\n")
        new.writestr(PREFIX+"head.joblib",payload)
        new.write(HERE/"predict.py",PREFIX+"predict.py")
        new.write(ROOT/"round19/weak_satd/runtime.py",RUNTIME)
        for name in set(("scan.mjs",program)) if program else ():
            new.write(HERE/name,PREFIX+name)
        new.write(storage/"results.json",PREFIX+"public_profile.json")
    temp.replace(output)
    sha=digest(output)
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(sha,output,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant",choices=("subsets","subsets_fast","subsets_numeric"))
    p.add_argument("--base",type=Path,default=SUBMISSION/"NLBSE2027_round19_tool.zip")
    p.add_argument("--output",type=Path,default=SUBMISSION/"NLBSE2027_round20_tool.zip")
    args=p.parse_args()
    build(args.base,args.output,args.variant)
