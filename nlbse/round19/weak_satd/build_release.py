"""Package the internally selected, profiled rule combiner with its frozen encoder."""
import argparse
from copy import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_SHA = "2f2ed530aa878b5c3a8f85d15adcc0bea9a4922d2816631728e9329e69e719b9"
PREFIX = "nlbse/round19/weak_satd/"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def json_at(path):
    return json.loads(path.read_text())


def expectations(path):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("row","pred_vulnerability","pred_MAT"))
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30655 or any(int(r["row"])!=i or any(r[k] not in ("0","1")
           for k in ("vulnerability","mat")) for i,r in enumerate(rows)):
        raise ValueError("Public result is incomplete or misordered")
    writer.writerows((i,r["vulnerability"],r["mat"]) for i,r in enumerate(rows))
    return stream.getvalue()


def build(base, output):
    if digest(base) != BASE_SHA or base.resolve() == output.resolve():
        raise ValueError("The verified round 16 archive is required as a separate input")
    folder = HERE/"official"
    measure = json_at(folder/"results.json")
    old = json_at(ROOT/"round16/context_public/results.json")["full"]["score"]
    config = json_at(folder/"policy.json")
    selected = config["selected"]
    if selected == "reference" or measure[selected]["score"] <= old:
        raise ValueError("There is no measured improvement to package")
    for name, expected in (("scan.mjs",config["scanner_sha256"]),
                           ("rules_manifest.json",config["rule_source_sha256"])):
        if digest(HERE/name) != expected:
            raise ValueError(f"Versioned release file changed: {name}")
    if digest(folder/"models.joblib") != config["model_sha256"]:
        raise ValueError("Frozen rule heads changed")
    manifest = json_at(HERE/"rules_manifest.json")
    for path, info in manifest["files"].items():
        if digest(HERE/"upstream"/path) != info["sha256"]:
            raise ValueError(f"Rule source hash changed: {path}")
    readme = f"""# NLBSE 2027 compact encoder with selective WeakSATD rules

Input is a CSV with a Function column. Output is vulnerability,MAT in input order.
The single compact encoder uses the full function and rereads a canonical view
where useful. The rule combiner receives only the function and model score.
The MAT decision is the original whole-comment rule.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8, and Node.js 22. The measured
scanner used Node.js 22.23.4. When Node is outside PATH, set NLBSE_NODE.

```bash
python3 -m pip install -r requirements-inference.txt
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
  python3 nlbse/round19/weak_satd/predict.py --input functions.csv --output predictions.csv
```

Public validation used 30,655 functions. The median RTX 4090 proxy score
from the measured trials is {measure[selected]["score"]:.6f}. The rule scanner
uses CPU workers and its time is included in the measured score. The organizer
ranks tools on the hidden set. See examples/public_expected.csv and the
release's public_profile.json for details. Rule provenance is recorded in
nlbse/round19/weak_satd/RULE_SOURCE.md.
"""
    notice = """# WeakSATD source attribution

The rule source and comment preprocessing are from Mock, Melegati, Kretschmann,
Díaz Ferreyra and Russo, MADE-WIC, 2024.
https://doi.org/10.5281/zenodo.12567874

The record labels the replication archive CC BY 4.0.
https://creativecommons.org/licenses/by/4.0/

Only rule source and its documentation were fetched. The runner adapts the
JSON import syntax, caches unchanged mitigation metadata, and evaluates rule
counts locally. The original source files and their digests are in this
directory. The research tool combines those counts with an NLBSE-trained
model. All labels used by the combiner are from the contest training data.
"""
    temp = output.with_suffix(".part")
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(base) as before, ZipFile(temp,"w",ZIP_DEFLATED) as archive:
        encoder = json.loads(before.read("nlbse/round16/entry.json"))
        for file in before.infolist():
            if file.filename not in {"README.md","examples/public_expected.csv"}:
                archive.writestr(copy(file),before.read(file))
        entry = {key:config[key] for key in ("selected","route","worker_limit","rule_columns",
                     "scanner_sha256","rule_source_sha256","model_sha256")}
        entry.update(checkpoint_sha256=encoder["checkpoint_sha256"],
                     threshold=config["modes"][selected]["threshold"],local_score=measure[selected]["score"])
        archive.writestr("README.md",readme)
        archive.writestr("examples/public_expected.csv",expectations(folder/f"{selected}_public.csv"))
        archive.writestr(PREFIX+"entry.json",json.dumps(entry,indent=2)+"\n")
        archive.writestr(PREFIX+"RULE_SOURCE.md",notice)
        for name in ("scan.mjs","rules_manifest.json","predict.py","runtime.py"):
            archive.write(HERE/name,PREFIX+name)
        for path in manifest["files"]:
            archive.write(HERE/"upstream"/path,PREFIX+"upstream/"+path)
        archive.write(folder/"models.joblib",PREFIX+"models.joblib")
        archive.write(folder/"results.json",PREFIX+"public_profile.json")
    temp.replace(output)
    sha = digest(output)
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(sha,output,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=ROOT/"submission/NLBSE2027_round16_tool.zip")
    p.add_argument("--output",type=Path,default=ROOT/"submission/NLBSE2027_round19_tool.zip")
    a=p.parse_args()
    build(a.base,a.output)
