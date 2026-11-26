"""Export one distilled readout after the selected variant passes the live benchmark."""
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
OLD=ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
OUT=ROOT/'submission/NLBSE2027_round30_distill_tool.zip'
OLD_HASH='ff131957ac0c7a1205bf62a22921bade0b51473c4e78209a6ea1fe89a46ed86d'
ENTRY='nlbse/round25/predict_sections.py'


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def replace(entry,before,after):
    if entry.count(before)!=1:
        raise ValueError('The verified entry point changed')
    return entry.replace(before,after)


def entrypoint(text):
    text=replace(text,'import numpy as np','import numpy as np\nimport torch')
    text=replace(text,'from negative_runtime import rerank',
                 'from rank30 import rerank, student_bits\nfrom student_runtime import head as student_head, predict as with_student')
    before='    p,aux=pf.infer(model,tokenizer,texts,auxiliary=True)'
    after='''    u=json.loads((HERE/"distill.json").read_text())
    path=HERE/"student.bin"
    readout_path=HERE/"student.joblib"
    pf.require_hash(path,u["head_sha256"])
    pf.require_hash(readout_path,u["decoder_sha256"])
    learned=student_head(model.width).cuda().eval()
    learned.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    p,aux,extra,_=with_student(model,tokenizer,texts,(learned,))'''
    text=replace(text,before,after)
    text=replace(text,'    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,',
                       '    margins=np.full(len(texts),np.nan)\n    vr,_,_,_=rerank(texts,values,state,config["threshold"],earlier["variant"],package,')
    text=replace(text,'settings["option"],head,negative["online"],joblib.load(weights))',
                 'settings["option"],head,negative["online"],joblib.load(weights),margins)')
    text=replace(text,'    return np.column_stack((vr,canonical_mat(texts))).astype(int)',
                 '    vr,_,_=student_bits(vr,margins,extra[:,0],aux,joblib.load(readout_path),u["online"])\n'
                 '    return np.column_stack((vr,canonical_mat(texts))).astype(int)')
    return text


def main():
    public=HERE/'public'
    plan=read(public/'policy.json')
    name=plan['selected']
    selected='reference'
    result=dict(selected=selected,policy_sha256=sha(public/'policy.json'))
    if name=='reference':
        (public/'release_selection.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    report=read(public/'results.json')
    if sha(OLD)!=OLD_HASH or result['policy_sha256']!=report['policy_sha256']:
        raise ValueError('Frozen policy or archived tool changed')
    before,after=[report['results'][k] for k in ('reference',name)]
    if after['score']>before['score'] and after['quality']['mean']['f1']>before['quality']['mean']['f1'] and sum(
            x>0 for x in report['pairs'][name])>=4:
        selected=name
    result.update(selected=selected,results_sha256=sha(public/'results.json'))
    (public/'release_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    if selected=='reference':
        print('No public improvement over the archived Round 29 tool.',flush=True)
        return
    option=plan['options'][name]
    model=HERE/f"student/official/{option['mode']}.pt"
    bank=HERE/'comparisons'/option['stage']/'official/readouts.joblib'
    decisions=public/'candidate.csv'
    if sha(model)!=option['head_sha256'] or sha(bank)!=option['readout_sha256'] or sha(decisions)!=after['predictions_sha256']:
        raise ValueError('Selected predictions or learned weights changed')
    buf=io.StringIO(newline='')
    out=csv.writer(buf)
    names=('row','pred_vulnerability','pred_MAT')
    out.writerow(names)
    with decisions.open(newline='') as fd:
        out.writerows([r[k] for k in names] for r in csv.DictReader(fd))
    keep=('width','mix','cut')
    tmp=OUT.with_suffix('.part')
    with TemporaryDirectory(prefix='nlbse_distill_') as d, ZipFile(OLD) as old, ZipFile(tmp,'w',ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in (ENTRY,'README.md','examples/public_expected.csv'):
                new.writestr(item,old.read(item))
        new.writestr(ENTRY,entrypoint(old.read(ENTRY).decode()))
        file=Path(d)/'student.joblib'
        joblib.dump(joblib.load(bank)[option['mode']],file,compress=3)
        new.write(file,'nlbse/round25/student.joblib')
        new.write(model,'nlbse/round25/student.bin')
        for f in ('student_runtime.py','rank30.py'):
            new.write(HERE/f,'nlbse/round25/'+f)
        receipt=dict(online={k:option[k] for k in keep},head_sha256=sha(model),decoder_sha256=sha(file),
                     source=result['policy_sha256'])
        new.writestr('nlbse/round25/distill.json',json.dumps(receipt,indent=2)+'\n')
        new.write(public/'results.json','nlbse/round30/public_profile.json')
        new.writestr('examples/public_expected.csv',buf.getvalue())
        new.writestr('README.md',f'''# NLBSE 2027 compact conditional code classifier

Reads a Function column and returns vulnerability and MAT. The compact
encoder has an additional readout over existing token states. The negative
scanner outcome uses this readout within the frozen decision band. MAT
uses normalized markers in the supplied comments.

Install Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Set NLBSE_NODE if Node is outside PATH.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Local score {after["score"]:.6f} versus {before["score"]:.6f} in five pairs.
Mean F1 {after["quality"]["mean"]["f1"]:.5f} on 30,655 rows.
Median elapsed time on RTX 4090 {after["timing"]["wall_seconds"]:.2f} s.
''')
    tmp.replace(OUT)
    digest=sha(OUT)
    OUT.with_suffix('.sha256').write_text(f'{digest}  {OUT.name}\n')
    print(digest,OUT,flush=True)


if __name__=='__main__':
    main()
