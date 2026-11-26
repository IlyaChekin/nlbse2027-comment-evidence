"""Package the compact recall correction only after a signed paired improvement."""
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import joblib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'submission/NLBSE2027_round28_calibrated_tool.zip'
NEW=ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
ENTRY='nlbse/round25/predict_sections.py'
OLD_HASH='f7c247d5c7570c503865a76735fb4f023a7beaa4d6b2124fa88c16bd9bbe93d7'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def build():
    base=HERE/'recover_public'
    plan=read(base/'policy.json')
    report=read(base/'results.json')
    if sha(OLD)!=OLD_HASH or sha(base/'policy.json')!=report['policy_sha256']:
        raise ValueError('Signed benchmark or baseline changed')
    name,=plan['options']
    before=report['results']['reference']
    after=report['results'][name]
    replace=(after['score']>before['score'] and after['quality']['mean']['f1']>before['quality']['mean']['f1']
             and sum(x>0 for x in report['paired'][name])>=4)
    (base/'release_selection.json').write_text(json.dumps(dict(selected=name if replace else 'reference',
                                                          comparison_sha256=sha(base/'results.json')),indent=2)+'\n')
    if not replace:
        print('No robust public improvement over the calibrated release.',flush=True)
        return
    option=plan['variants'][name]
    weight=HERE/'recover_calibrated/official/heads.joblib'
    source=base/after['csv']
    if sha(weight)!=option['source_sha256'] or sha(source)!=after['predictions_sha256']:
        raise ValueError('Frozen new weights or predictions changed')
    reply=io.StringIO(newline='')
    writer=csv.writer(reply)
    fields=('row','pred_vulnerability','pred_MAT')
    writer.writerow(fields)
    with source.open(newline='') as fd:
        writer.writerows([r[k] for k in fields] for r in csv.DictReader(fd))
    skip={ENTRY,'README.md','examples/public_expected.csv','nlbse/round25/negative_runtime.py',
          'nlbse/round25/negative.json'}
    old_call='''    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,
                    settings["option"],head,negative["online"])'''
    new_call='''    weights = HERE/"recall.joblib"
    pf.require_hash(weights,negative["sha256"])
    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,
                    settings["option"],head,negative["online"],joblib.load(weights))'''
    temp=NEW.with_suffix('.part')
    with TemporaryDirectory(prefix='nlbse_recall_') as folder, ZipFile(OLD) as old, ZipFile(temp,'w',ZIP_DEFLATED) as new:
        for record in old.infolist():
            if record.filename not in skip:
                new.writestr(record,old.read(record))
        entry=old.read(ENTRY).decode()
        if entry.count(old_call)!=1:
            raise ValueError('The signed entry point changed')
        new.writestr(ENTRY,entry.replace(old_call,new_call))
        file=Path(folder)/'recall.joblib'
        joblib.dump(joblib.load(weight)['compact'],file,compress=3)
        new.write(file,'nlbse/round25/recall.joblib')
        cfg=dict(online=option['setting'],sha256=sha(file),source=report['policy_sha256'])
        new.writestr('nlbse/round25/negative.json',json.dumps(cfg,indent=2)+'\n')
        new.write(HERE/'negative_runtime.py','nlbse/round25/negative_runtime.py')
        new.write(base/'results.json','nlbse/round28/recall_public_profile.json')
        new.writestr('examples/public_expected.csv',reply.getvalue())
        new.writestr('README.md',f'''# NLBSE 2027 compact conditional code classifier

Reads a Function column and writes vulnerability and MAT. The verified
negative outcome uses a calibrated decision and a small recall correction
over scores computed by the base tool. MAT scans the supplied comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Local score {after["score"]:.6f} versus {before["score"]:.6f} in five paired trials.
Mean F1 {after["quality"]["mean"]["f1"]:.5f} on 30,655 public validation rows.
Median elapsed time on RTX 4090 {after["cost"]["wall_seconds"]:.2f} s.
''')
    temp.replace(NEW)
    digest=sha(NEW)
    NEW.with_suffix('.sha256').write_text(f'{digest}  {NEW.name}\n')
    print(digest,NEW,flush=True)


if __name__=='__main__':
    build()
