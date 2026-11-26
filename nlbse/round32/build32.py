"""Package the verified score adjustment with the released compact checkpoint."""
import csv
import io
import json
from zipfile import ZipFile, ZIP_DEFLATED

import joblib
import numpy as np

from summarize32 import HERE, ROOT, sha, read

OLD=ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
NEW=ROOT/'submission/NLBSE2027_round32_rule_calibration_tool.zip'
ENTRY='nlbse/round25/predict_sections.py'
CONFIG='nlbse/round25/entry.json'
SCANNER='nlbse/round22/scan_first.mjs'
CALIBRATION='nlbse/round25/rule_calibration.json'
BASE_SHA='ff131957ac0c7a1205bf62a22921bade0b51473c4e78209a6ea1fe89a46ed86d'

REPLACEMENTS={
    'from positive_runtime import apply as select_positive':
        'from positive_runtime import apply as select_positive\nfrom rule_margin32 import adjust',
    '    pf.require_hash(PROGRAM,settings["scanner_sha256"])':
        '''    rule_path=HERE/"rule_calibration.json"
    pf.require_hash(rule_path,settings["rule_calibration_sha256"])
    rule_calibration=json.loads(rule_path.read_text())
    scan_first=HERE.parent/"round22"/"scan_first.mjs"
    pf.require_hash(scan_first,rule_calibration["scanner_sha256"])''',
    '    flag=np.zeros_like(known)':'''    flag=np.zeros_like(known)
    evidence=np.zeros((len(texts),2),dtype=np.int32)''',
    'if on],program=PROGRAM)':'if on],program=scan_first)',
    'if cols!=["rule_union"]:':'if cols!=["rule_union","first_rule","line_count"]:',
    '        flag[asked]=raw[:,0]==1':'''        flag[asked]=raw[:,0]==1
        evidence[asked]=raw[:,1:]''',
    '    vr,_,_,_ = select_positive(texts,values,state,config["threshold"],vr,positive["online"],joblib.load(weights))':
        '''    vr,corrected,score,_ = select_positive(texts,values,state,config["threshold"],vr,positive["online"],joblib.load(weights))
    vr=adjust(texts,state,config["threshold"],vr,corrected,score,positive["online"]["cut"],evidence,rule_calibration)''',
}


def main():
    public=HERE/'public'
    src=public/'results.json'
    profile=read(src)
    chosen=profile['policy']
    verified=profile['receipt']
    ref, candidate=(profile['results'][k] for k in ('reference','candidate'))
    if sha(HERE/'selection.json')!=verified['selection_sha256'] or sha(OLD)!=BASE_SHA:
        raise ValueError('Reference tool or frozen selection changed')
    if sha(HERE/'live32.py')!=profile['code_sha256'] or sha(HERE/'scan_first.mjs')!=verified['scanner_sha256']:
        raise ValueError('Benchmarked implementation changed')
    if not (candidate['score']>ref['score'] and candidate['score']>read(HERE/'selection.json')['inventory']['current_score']
            and candidate['quality']['mean']['f1']>ref['quality']['mean']['f1']
            and sum(delta>0 for delta in profile['paired'])>=4):
        raise ValueError('Public scores did not show a repeatable increase')
    truth_file=public/'candidate.npz'
    assert sha(truth_file)==candidate['predictions_sha256']
    with np.load(truth_file) as z:
        predicted=z['pred'].astype(np.int8)
    rows=io.StringIO(newline='')
    writer=csv.writer(rows)
    writer.writerow(('row','pred_vulnerability','pred_MAT'))
    writer.writerows((i,int(v),int(m)) for i,(v,m) in enumerate(predicted))
    weights=HERE/chosen['stage']/'official/models.joblib'
    assert sha(weights)==verified['weights_sha256']
    cfg=dict(width=chosen['width'],strength=chosen['strength'],cut=chosen['cut'],
             weights=joblib.load(weights)[chosen['style']].tolist(),scanner_sha256=verified['scanner_sha256'])
    assert chosen['style']=='counts' and len(cfg['weights'])==62
    payload=json.dumps(cfg,indent=2)+'\n'
    import hashlib
    digest=hashlib.sha256(payload.encode()).hexdigest()
    temporary=NEW.with_suffix('.part')
    skip={ENTRY,CONFIG,'README.md','examples/public_expected.csv'}
    with ZipFile(OLD) as previous, ZipFile(temporary,'w',ZIP_DEFLATED) as output:
        for member in previous.infolist():
            if member.filename not in skip:
                output.writestr(member,previous.read(member))
        body=previous.read(ENTRY).decode()
        for search,replacement in REPLACEMENTS.items():
            if body.count(search)!=1:
                raise ValueError('The archived entry point changed')
            body=body.replace(search,replacement)
        output.writestr(ENTRY,body)
        earlier=json.loads(previous.read(CONFIG))
        earlier['rule_calibration_sha256']=digest
        output.writestr(CONFIG,json.dumps(earlier,indent=2)+'\n')
        output.writestr(CALIBRATION,payload)
        output.write(HERE/'rule_margin32.py','nlbse/round25/rule_margin32.py')
        output.write(HERE/'scan_first.mjs', SCANNER)
        output.write(src,'nlbse/round32/calibrated_rules_profile.json')
        output.writestr('examples/public_expected.csv',rows.getvalue())
        output.writestr('README.md',f"""# NLBSE 2027 compact conditional code classifier

Reads a Function column and writes vulnerability and MAT. The exact scanner
reports its first rule and count at the usual early exit. A small calibration
uses these values near the positive scanner decision boundary.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Use Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.
Measured local score {candidate['score']:.6f} versus {ref['score']:.6f} in five pairs.
Mean F1 {candidate['quality']['mean']['f1']:.5f} on 30,655 public functions.
Median elapsed time on RTX 4090 {candidate['cost']['wall_seconds']:.2f} s.
""")
    temporary.replace(NEW)
    digest=sha(NEW)
    NEW.with_suffix('.sha256').write_text(f'{digest}  {NEW.name}\n')
    print(digest,NEW,flush=True)


if __name__=='__main__':
    main()
