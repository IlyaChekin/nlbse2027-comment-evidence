"""Package a threshold-only variant if it wins the separate measured comparison."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import csv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SIGNED_SOURCE='97195d9378ca531ef12f5ea74404e216c883c08ac7d3cf243942e8e649a083c6'
OLD=ROOT/'submission/NLBSE2027_round28_negative_tool.zip'
NEW=ROOT/'submission/NLBSE2027_round28_calibrated_tool.zip'
ENTRY='nlbse/round25/predict_sections.py'


def sha(path):
    with path.open('rb') as s:
        return hashlib.file_digest(s,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def make():
    src=HERE/'threshold'
    meta=read(src/'policy.json')
    measured=read(src/'results.json')
    if sha(src/'policy.json')!=measured['policy_sha256'] or sha(OLD)!=SIGNED_SOURCE:
        raise ValueError('The benchmark or the source release changed')
    chosen=measured['results']['calibration_only']
    baseline=measured['results'][meta['family']['score']]
    contrast=[a-b for a,b in zip(measured['paired']['calibration_only'],
                                 measured['paired'][meta['family']['score']],strict=True)]
    replace=(chosen['score']>baseline['score'] and chosen['quality']['mean']['f1']>=baseline['quality']['mean']['f1']
             and sum(d>0 for d in contrast)>=4)
    result=dict(selected='calibration_only' if replace else meta['family']['score'],
                point_model=meta['family']['score'],pairs_minus_point=contrast,
                results_sha256=sha(src/'results.json'),policy_sha256=measured['policy_sha256'])
    (src/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    if not replace:
        print('Threshold-only variant did not pass the paired release gate.',flush=True)
        return
    source=src/chosen['csv']
    if sha(source)!=chosen['predictions_sha256']:
        raise ValueError('Public predictions changed')
    buf=io.StringIO(newline='')
    w=csv.writer(buf)
    fields=('row','pred_vulnerability','pred_MAT')
    w.writerow(fields)
    with source.open(newline='') as f:
        w.writerows([r[k] for k in fields] for r in csv.DictReader(f))
    before='''    rankers = HERE/"negative.joblib"
    pf.require_hash(rankers,negative["sha256"])
    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,
                    settings["option"],head,negative["online"],joblib.load(rankers))'''
    after='''    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,
                    settings["option"],head,negative["online"])'''
    omit={ENTRY,'README.md','examples/public_expected.csv','nlbse/round25/negative.joblib',
          'nlbse/round25/negative_runtime.py','nlbse/round25/negative.json'}
    temp=NEW.with_suffix('.part')
    with ZipFile(OLD) as old, ZipFile(temp,'w',ZIP_DEFLATED) as new:
        for i in old.infolist():
            if i.filename not in omit:
                new.writestr(i,old.read(i))
        entry=old.read(ENTRY).decode()
        if entry.count(before)!=1:
            raise ValueError('The signed entry point does not match')
        new.writestr(ENTRY,entry.replace(before,after))
        settings=meta['variants']['calibration_only']['setting']
        new.writestr('nlbse/round25/negative.json',json.dumps(dict(online=settings,source=measured['policy_sha256']),indent=2)+'\n')
        new.write(HERE/'negative_runtime.py','nlbse/round25/negative_runtime.py')
        new.write(src/'results.json','nlbse/round28/threshold_profile.json')
        new.writestr('examples/public_expected.csv',buf.getvalue())
        new.writestr('README.md',f'''# NLBSE 2027 compact conditional code classifier

Reads a Function column and writes vulnerability and MAT. The compact encoder,
static auxiliary outcome and sparse score are combined conditionally. A separate
threshold near the decision boundary applies to observed negative outcomes.
MAT scans supplied comments for normalized annotation markers.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

On 30,655 public validation functions, mean F1 is {chosen["quality"]["mean"]["f1"]:.5f}.
The score is {chosen["score"]:.6f} versus {baseline["score"]:.6f} with extra ranking.
Median end-to-end time is {chosen["cost"]["wall_seconds"]:.2f} s on RTX 4090.
''')
    temp.replace(NEW)
    digest=sha(NEW)
    NEW.with_suffix('.sha256').write_text(f'{digest}  {NEW.name}\n')
    print(digest,NEW,flush=True)


if __name__=='__main__':
    make()
