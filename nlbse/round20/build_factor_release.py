"""Package the best independently calibrated conditional rule model."""
import argparse
from copy import copy
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

from build_release import HERE,ROOT,SUBMISSION,BASE_SHA,PREFIX,RUNTIME,digest,expected,json_at

ENCODER="nlbse/round11/small.py"

def build(base,output,variant):
    if digest(base)!=BASE_SHA or base.resolve()==output.resolve():
        raise ValueError("Expected the unchanged round 19 release and a separate output")
    storage=HERE/variant/"official"
    policy=json_at(storage/"policy.json")
    record=json_at(storage/"results.json")
    name=policy["selected"]
    if name=="reference" or record[name]["score"]<=record["reference"]["score"]:
        raise ValueError("The conditionally selected model did not improve its paired public result")
    chosen=policy["modes"][name]
    fit=HERE/"official/42"
    model=json_at(fit/"fit.json")["modes"]["factor"]
    weight=fit/model["file"]
    if digest(weight)!=model["sha256"]:
        raise ValueError("Factorized weights differ from the profiled version")
    numeric=policy.get("numeric",False)
    program="scan_extended.mjs" if numeric else "scan.mjs"
    program_hash=policy.get("program_sha256",policy["scanner_sha256"])
    if digest(HERE/program)!=program_hash or digest(HERE/"scan.mjs")!=policy["scanner_sha256"]:
        raise ValueError("Rule scanner differs from the profiled version")
    heads=storage/"heads.joblib"
    score=record[name]["score"]
    quality=record[name]["quality"]
    wall=record[name]["cost"]["wall_seconds"]
    temp=output.with_suffix(".part")
    with ZipFile(base) as old,ZipFile(temp,"w",ZIP_DEFLATED) as new:
        ref=json.loads(old.read("nlbse/round19/weak_satd/entry.json"))
        config=dict(policy=chosen["policy"],gate=chosen["gate"],threshold=chosen["threshold"],
            route=ref["route"],checkpoint_sha256=model["sha256"],source_sha256=ref["rule_source_sha256"],
            program=program,program_sha256=program_hash,scanner_sha256=policy["scanner_sha256"],
            rule_columns=ref["rule_columns"]+(["rule_195"] if numeric else []),
            head_sha256=digest(heads),worker_limit=ref["worker_limit"],score=score)
        for entry in old.infolist():
            if entry.filename not in ("README.md","examples/public_expected.csv",RUNTIME,ENCODER) and not entry.filename.endswith(".pt"):
                new.writestr(copy(entry),old.read(entry))
        new.writestr("README.md",f"""# NLBSE 2027 · conditional rule-guided classifier

Input is a CSV with a Function column. Output is vulnerability,MAT in input order.
A single compact encoder predicts the rule event and vulnerability scores
conditional on this event. The uncertainty gate queries the exact rule scanner
for selected functions. The complete comments still determine MAT.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22. Set NLBSE_NODE if
Node is outside PATH. The recorded timings used 6 CPU workers.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 nlbse/round20/predict_factor.py --input functions.csv --output predictions.csv
```

Local score {score:.6f} on all 30,655 public functions. Vulnerability F1
{quality["vulnerability"]["f1"]:.5f}, MAT F1 {quality["MAT"]["f1"]:.5f}.
Measured median end-to-end time {wall:.2f} s on the RTX 4090 host.
The rule CPU time is included. The competition ranks tools on the hidden set.
Rule provenance is in nlbse/round19/weak_satd/RULE_SOURCE.md.
""")
        new.writestr("examples/public_expected.csv",expected(storage/f"{name}_public.csv"))
        new.writestr(PREFIX+"factor_entry.json",json.dumps(config,indent=2)+"\n")
        new.write(ROOT/"round19/weak_satd/runtime.py",RUNTIME)
        new.write(ROOT/"round11/small.py",ENCODER)
        for source in sorted({"predict_factor.py","rule_model.py","conditional.py","scan.mjs",program}):
            new.write(HERE/source,PREFIX+source)
        new.write(storage/"results.json",PREFIX+"public_profile.json")
        new.write(weight,PREFIX+"factor.pt")
        new.write(heads,PREFIX+"conditional_heads.joblib")
    temp.replace(output)
    sha=digest(output)
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(sha,output,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant",choices=("factor","factor_numeric"))
    p.add_argument("--base",type=Path,default=SUBMISSION/"NLBSE2027_round19_tool.zip")
    p.add_argument("--output",type=Path,default=SUBMISSION/"NLBSE2027_round20_factor_tool.zip")
    a=p.parse_args()
    build(a.base,a.output,a.variant)
