"""Create a release after the frozen positive-cohort policy wins its paired comparison."""
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT/'submission/NLBSE2027_round26_mat_tool.zip'
OUT = ROOT/'submission/NLBSE2027_round27_positive_tool.zip'
BASE_HASH = '2ed64b204bb3e25ac3e00b820db9c1f347645ddcd352154a7136c97298f7748e'
ENTRY = 'nlbse/round25/predict_sections.py'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def predictions(path):
    sink = io.StringIO(newline='')
    w = csv.writer(sink)
    fields = ('row', 'pred_vulnerability', 'pred_MAT')
    w.writerow(fields)
    with path.open() as fd:
        w.writerows([r[k] for k in fields] for r in csv.DictReader(fd))
    return sink.getvalue()


def add_entrypoint(source):
    old = 'from section_runtime import adjust'
    before = '    return np.column_stack((vr,canonical_mat(texts))).astype(int)'
    if source.count(old) != 1 or source.count(before) != 1:
        raise ValueError('Round 26 entry point does not have the expected structure')
    source = source.replace(old, old+'\nfrom positive_runtime import apply as select_positive')
    readout = '''    positive = json.loads((HERE/"positive.json").read_text())
    weights = HERE/"positive.joblib"
    pf.require_hash(weights,positive["sha256"])
    vr,_,_,_ = select_positive(texts,values,state,config["threshold"],vr,positive["online"],joblib.load(weights))
'''
    return source.replace(before, readout+before)


def build():
    ref = HERE/'public'
    profile = json.loads((ref/'results.json').read_text())
    frozen = json.loads((ref/'policy.json').read_text())
    selected = frozen['selected']
    if selected != profile['selected'] or sha(ref/'policy.json') != profile['policy_sha256'] or sha(BASE) != BASE_HASH:
        raise ValueError('Signed benchmark inputs changed')
    new, old = [profile['results'][k] for k in (selected, 'reference')]
    if (new['score'] <= old['score'] or new['frozen']['f1_after'] <= new['frozen']['f1_before'] or
            sum(d > 0 for d in profile['paired_delta']) < 4):
        raise ValueError('The live candidate did not win the paired benchmark')
    artifact = HERE/'joint/official/weights.joblib'
    forecast = ref/'candidate.csv'
    if sha(artifact) != frozen['weights_sha256'] or sha(forecast) != new['predictions_sha256']:
        raise ValueError('Saved model or prediction file changed')
    option = frozen['supported'][selected]
    weights = joblib.load(artifact)[option['weights_key']]
    tmp = OUT.with_suffix('.part')
    with TemporaryDirectory(prefix='nlbse_positive_') as folder, ZipFile(BASE) as old_zip, ZipFile(tmp, 'w', ZIP_DEFLATED) as dst:
        for item in old_zip.infolist():
            if item.filename not in (ENTRY, 'README.md', 'examples/public_expected.csv'):
                dst.writestr(item, old_zip.read(item))
        expected = old_zip.read(ENTRY).decode()
        dst.writestr(ENTRY, add_entrypoint(expected))
        file = Path(folder)/'positive.joblib'
        joblib.dump(weights, file, compress=3)
        dst.write(file, 'nlbse/round25/positive.joblib')
        config = dict(online=option['online'], sha256=sha(file), provenance=profile['policy_sha256'])
        dst.writestr('nlbse/round25/positive.json', json.dumps(config, indent=2)+'\n')
        dst.write(HERE/'positive_runtime.py', 'nlbse/round25/positive_runtime.py')
        dst.write(ref/'results.json', 'nlbse/round27/public_profile.json')
        dst.writestr('examples/public_expected.csv', predictions(forecast))
        dst.writestr('README.md', f'''# NLBSE 2027 compact conditional code classifier

The input CSV contains one Function column. The output contains vulnerability and MAT.
Vulnerability uses one compact encoder, selective static checks and sparse token evidence.
The verified negative and positive outcomes have separate calibrated scores.
MAT reads tags in the comments of the supplied function.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22. Set NLBSE_NODE if Node is outside PATH.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

The local paired score is {new["score"]:.6f} versus {old["score"]:.6f} on all 30,655 public validation functions.
The mean F1 is {new["quality"]["mean"]["f1"]:.5f}. Median end-to-end time is {new["cost"]["wall_seconds"]:.2f} s on RTX 4090.
Public predictions and the paired measurement receipt are included.
''')
    tmp.replace(OUT)
    digest = sha(OUT)
    OUT.with_suffix('.sha256').write_text(f'{digest}  {OUT.name}\n')
    print(digest, OUT, flush=True)


if __name__ == '__main__':
    build()
