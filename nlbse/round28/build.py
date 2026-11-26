"""Build the released tool only when a preselected policy passes the live comparison."""
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
BASE=ROOT/'submission/NLBSE2027_round27_positive_tool.zip'
OUT=ROOT/'submission/NLBSE2027_round28_negative_tool.zip'
REFERENCE='c3e79dedfac2195a62ffa22f0c458dd1823108448cbc2e247c907e82ac4ca1fd'
ENTRY='nlbse/round25/predict_sections.py'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def text(old):
    take='from section_runtime import adjust'
    find='''    vr,_,_,_,_=adjust(texts,values,state,config["threshold"],earlier["variant"],package,
                         settings["option"],settings["length_cutoff"],head)'''
    instead='''    negative = json.loads((HERE/"negative.json").read_text())
    rankers = HERE/"negative.joblib"
    pf.require_hash(rankers,negative["sha256"])
    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,
                    settings["option"],head,negative["online"],joblib.load(rankers))'''
    if old.count(take)!=1 or old.count(find)!=1:
        raise ValueError('Round 27 entry structure changed')
    return old.replace(take,'from negative_runtime import rerank').replace(find,instead)


def predictions(path):
    stream=io.StringIO(newline='')
    sink=csv.writer(stream)
    fields=('row','pred_vulnerability','pred_MAT')
    sink.writerow(fields)
    with path.open(newline='') as fd:
        sink.writerows([r[k] for k in fields] for r in csv.DictReader(fd))
    return stream.getvalue()


def main():
    table=HERE/'public'
    frozen=json.loads((table/'policy.json').read_text())
    benchmark=json.loads((table/'results.json').read_text())
    if sha(table/'policy.json')!=benchmark['policy_sha256'] or sha(BASE)!=REFERENCE:
        raise ValueError('The baseline or candidate policy changed')
    profiles=benchmark['results']
    reference=profiles['reference']
    picked='reference'
    for k in frozen['options']:
        row=profiles[k]
        if (row['score']>reference['score'] and row['quality']['mean']['f1']>reference['quality']['mean']['f1']
                and sum(d>0 for d in benchmark['paired'][k])>=4):
            picked=k
            break
    receipt=table/'release_selection.json'
    receipt.write_text(json.dumps(dict(selected=picked,policy_sha256=benchmark['policy_sha256'],
                                       comparison_sha256=sha(table/'results.json')),indent=2)+'\n')
    if picked=='reference':
        print('No live candidate beat the signed Round 27 archive.',flush=True)
        return
    chosen=profiles[picked]
    score=frozen['variants'][picked]
    path=HERE/'studies'/score['setting']['kind']/'official/readouts.joblib'
    forecast=table/chosen['csv']
    if sha(path)!=score['source_sha256'] or sha(forecast)!=chosen['predictions_sha256']:
        raise ValueError('Public predictions or learned weights changed')
    bundle=joblib.load(path)[score['mode']]
    tmp=OUT.with_suffix('.part')
    with TemporaryDirectory(prefix='nlbse_negative_') as d, ZipFile(BASE) as old, ZipFile(tmp,'w',ZIP_DEFLATED) as new:
        for row in old.infolist():
            if row.filename not in (ENTRY,'examples/public_expected.csv','README.md'):
                new.writestr(row,old.read(row))
        new.writestr(ENTRY,text(old.read(ENTRY).decode()))
        new.write(HERE/'negative_runtime.py','nlbse/round25/negative_runtime.py')
        destination=Path(d)/'negative.joblib'
        joblib.dump(bundle,destination,compress=3)
        new.write(destination,'nlbse/round25/negative.joblib')
        config=dict(online=score['setting'],sha256=sha(destination),source=benchmark['policy_sha256'])
        new.writestr('nlbse/round25/negative.json',json.dumps(config,indent=2)+'\n')
        new.write(table/'results.json','nlbse/round28/public_profile.json')
        new.writestr('examples/public_expected.csv',predictions(forecast))
        new.writestr('README.md',f'''# NLBSE 2027 compact conditional code classifier

Reads a Function column and writes vulnerability and MAT. The compact
encoder, static auxiliary outcome and sparse token classifier produce
a separate ranking for each observed scanner outcome. MAT uses a
canonical marker detector over the supplied comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

On 30,655 public validation functions, mean F1 is {chosen["quality"]["mean"]["f1"]:.5f}.
The score is {chosen["score"]:.6f} versus {reference["score"]:.6f} in the paired benchmark.
Median end-to-end time is {chosen["cost"]["wall_seconds"]:.2f} s on RTX 4090.
''')
    tmp.replace(OUT)
    digest=sha(OUT)
    OUT.with_suffix('.sha256').write_text(f'{digest}  {OUT.name}\n')
    print(digest,OUT,flush=True)


if __name__=='__main__':
    main()
