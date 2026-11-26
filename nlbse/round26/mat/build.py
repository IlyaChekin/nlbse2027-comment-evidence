"""Archive the verified MAT change without modifying the Round 25 release."""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'submission/NLBSE2027_round25_sections_tool.zip'
OUT = ROOT/'submission/NLBSE2027_round26_mat_tool.zip'
BASE_HASH = '0750e06e9fda60df21a124dc39c0a43e008cb2a71fc20b7894767f2a226564a2'
ENTRY = 'nlbse/round25/predict_sections.py'


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def labels(path):
    sink = io.StringIO(newline='')
    writer = csv.writer(sink)
    fields = ('row', 'pred_vulnerability', 'pred_MAT')
    writer.writerow(fields)
    with path.open(newline='') as handle:
        writer.writerows([r[key] for key in fields] for r in csv.DictReader(handle))
    return sink.getvalue()


def build():
    pub = HERE/'public'
    result = json.loads((pub/'results.json').read_text())
    candidate, reference = (result['results'][n] for n in ('canonical', 'reference'))
    predictions = pub/'canonical.csv'
    policy = pub/'policy.json'
    frozen = json.loads(policy.read_text())
    runtime = HERE/'mat_runtime.py'
    if (sha(BASE) != BASE_HASH or sha(policy) != result['policy_sha256'] or
            sha(predictions) != candidate['predictions_sha256'] or sha(runtime) != frozen['canonical_sha256']):
        raise ValueError('The frozen package inputs changed')
    if (candidate['score'] <= reference['score'] or candidate['mat']['delta_tp'] < 0 or
            candidate['mat']['delta_fp'] > 0 or sum(d > 0 for d in result['paired_delta']) < 4):
        raise ValueError('Canonical MAT did not pass the paired benchmark')
    tmp = OUT.with_suffix('.part')
    with ZipFile(BASE) as src, ZipFile(tmp, 'w', ZIP_DEFLATED) as dst:
        for record in src.infolist():
            if record.filename not in (ENTRY, 'README.md', 'examples/public_expected.csv'):
                dst.writestr(record, src.read(record))
        if hashlib.sha256(src.read('nlbse/round21/mat/marker.py')).hexdigest() != frozen['marker_sha256']:
            raise ValueError('Archived marker differs from the measured version')
        previous = src.read(ENTRY).decode()
        expected = 'from predict_sparse import pf,PROGRAM,recover'
        if previous.count(expected) != 1 or previous.count('recover(texts,p[:,1]>.5)') != 1:
            raise ValueError('Unexpected Round 25 entry point')
        entry = previous.replace(expected, 'from predict_sparse import pf,PROGRAM\nfrom mat_runtime import canonical_mat')
        dst.writestr(ENTRY, entry.replace('recover(texts,p[:,1]>.5)', 'canonical_mat(texts)'))
        dst.write(runtime, 'nlbse/round25/mat_runtime.py')
        dst.write(pub/'results.json', 'nlbse/round26/mat/public_profile.json')
        dst.write(policy, 'nlbse/round26/mat/policy.json')
        dst.writestr('examples/public_expected.csv', labels(predictions))
        dst.writestr('README.md', f'''# NLBSE 2027 compact conditional code classifier

The input CSV contains a Function column. The output contains vulnerability and MAT.
Vulnerability predictions use a compact encoder, conditional static checks and sparse
scores for the function signature and body. MAT uses the normalized annotation tags
in the comments of the supplied function.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22 before running the tool.
Set NLBSE_NODE if Node is outside PATH.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

The local paired score is {candidate['score']:.6f} versus {reference['score']:.6f}
on the 30,655 public validation functions. Mean F1 is {candidate['quality']['mean']['f1']:.5f}.
The median end-to-end time is {candidate['cost']['wall_seconds']:.2f} s on RTX 4090.
Public predictions and measurement receipts are included.
''')
    tmp.replace(OUT)
    digest = sha(OUT)
    OUT.with_suffix('.sha256').write_text(f'{digest}  {OUT.name}\n')
    print(digest, OUT)


if __name__ == '__main__':
    build()
