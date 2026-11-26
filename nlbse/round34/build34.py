"""Package the reproducible pooled projection once it passes the paired public test."""
import csv
import hashlib
from inspect import getsource
import io
import json
from zipfile import ZipFile, ZIP_DEFLATED

import joblib
import numpy as np
import torch

from sections34 import HERE, ROOT, SPANS
from select34 import read, sha
from compare34 import design
from policy33 import logit, matrix
from rescue33 import descriptors

BASE=ROOT/'submission/NLBSE2027_round32_rule_calibration_tool.zip'
NEW=ROOT/'submission/NLBSE2027_round34_pooled_projection_tool.zip'
ENTRY='nlbse/round25/predict_sections.py'
ENTRY_CONFIG='nlbse/round25/entry.json'
NEGATIVE='nlbse/round25/negative_runtime.py'
MODULE='nlbse/round25/approx_runtime34.py'
CHECKPOINT='nlbse/round25/pooled_adjustment.joblib'


def make_entry(src):
    changes={
        'from predict_sparse import pf,PROGRAM':'from predict_sparse import pf,PROGRAM\nimport torch\nfrom fuse_sparse import clip_logit',
        'from rule_margin32 import adjust':'from rule_margin32 import adjust\nfrom approx_runtime34 import encode, regional_adjust',
        '    p,aux=pf.infer(model,tokenizer,texts,auxiliary=True)':'''    projected_path=HERE/"pooled_adjustment.joblib"
    pf.require_hash(projected_path,settings["pooled_adjustment_sha256"])
    trained=joblib.load(projected_path)
    projection=torch.nn.Linear(2*model.width,2).cuda().eval()
    projection.load_state_dict({k:torch.as_tensor(v) for k,v in trained["weights"].items()})
    p,aux,extra=encode(model,tokenizer,texts,projection)''',
        '    q,no,yes,asked=pf.choose(values,est,pf.Policy(**config["policy"]),config["gate"])':
        '''    policy=pf.Policy(**config["policy"])
    q,no,yes,asked=pf.choose(values,est,policy,config["gate"])
    prior=q.copy()
    literal_cut=clip_logit(np.asarray(np.float32(config["threshold"]),dtype=np.float64))
    first_margin=clip_logit(prior.astype(np.float32).astype(np.float64))-literal_cut
    negative_margin=np.full(len(texts),np.nan)''',
        'settings["option"],head,negative["online"],joblib.load(weights))':
            'settings["option"],head,negative["online"],joblib.load(weights),negative_margin)',
        '    return np.column_stack((vr,canonical_mat(texts))).astype(int)':
            '''    vr=regional_adjust(texts,first_margin,negative_margin,values,est,no,yes,prior,known,flag,
                       vr,config,policy,extra,trained)
    return np.column_stack((vr,canonical_mat(texts))).astype(int)''',
    }
    for old,new in changes.items():
        if src.count(old)!=1:
            raise ValueError('The released CLI no longer matches the signed anchors')
        src=src.replace(old,new)
    return src


def build():
    source=HERE/'public_approx/results.json'
    record=read(source)
    audit=read(HERE/'audit.json')
    if audit['release_candidate']!='public_approx':
        raise ValueError('The projected head has not passed the paired public tests')
    assert sha(BASE)==audit['round32_archive_sha256']
    assert audit['outcomes']['public_approx']['check']==sha(source)
    assert record['receipt']['selection_sha256']==sha(HERE/'selection_approx.json')
    exact=HERE/'public_approx/candidate.npz'
    result=record['comparison']['candidate']
    assert sha(exact)==result['predictions_sha256']
    settings=record['chosen']
    assert settings['stage']=='approx_comparison'
    weights=HERE/'approximation/official'/f"{settings['mode']}.pt"
    head=HERE/'approx_comparison/official/calibrators.joblib'
    assert sha(weights)==record['receipt']['attention_sha256']
    assert sha(head)==record['receipt']['calibration_sha256']
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    trained=dict(width=settings['width'],strength=settings['strength'],cut=settings['cut'],
                 weights={k:v.numpy() for k,v in checkpoint.items()},
                 calibrator=joblib.load(head)[settings['mode'],settings['source']])
    data=io.BytesIO()
    joblib.dump(trained,data,compress=3)
    payload=data.getvalue()
    runtime=(HERE/'approx_runtime34.py').read_text()+f'\n\nSPANS = {SPANS!r}\n'
    for function in (logit,matrix,descriptors,design):
        runtime+='\n'+getsource(function)+'\n'
    temporary=NEW.with_suffix('.part')
    routine=ROOT/'round30/rank30.py'
    code=routine.read_text().split('\ndef student_bits',1)[0]
    expected=io.StringIO(newline='')
    with np.load(exact) as z:
        w=csv.writer(expected)
        w.writerow(('row','pred_vulnerability','pred_MAT'))
        w.writerows((i,int(x),int(y)) for i,(x,y) in enumerate(z['pred']))
    excluded={ENTRY,ENTRY_CONFIG,NEGATIVE,'README.md','examples/public_expected.csv'}
    with ZipFile(BASE) as before,ZipFile(temporary,'w',ZIP_DEFLATED) as after:
        for item in before.infolist():
            if item.filename not in excluded:
                after.writestr(item,before.read(item))
        after.writestr(ENTRY,make_entry(before.read(ENTRY).decode()))
        config=json.loads(before.read(ENTRY_CONFIG))
        config['pooled_adjustment_sha256']=hashlib.sha256(payload).hexdigest()
        after.writestr(ENTRY_CONFIG,json.dumps(config,indent=2)+'\n')
        after.writestr(NEGATIVE,code)
        after.writestr(MODULE,runtime)
        after.writestr(CHECKPOINT,payload)
        after.writestr('examples/public_expected.csv',expected.getvalue())
        after.write(source,'nlbse/round34/public_profile.json')
        after.writestr('README.md',f"""# NLBSE 2027 compact conditional code classifier

The tool reads a Function column and writes vulnerability and MAT. A linear
projection of the compact encoder's pooled state refines selected decisions
with an observed negative scanner result.

Use Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \\
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Measured local score {result['score']:.6f}. Mean F1 {result['quality']['mean']['f1']:.5f}
on 30,655 public functions. Median elapsed time on RTX 4090 {result['cost']['wall_seconds']:.2f} s.
""")
    temporary.replace(NEW)
    digest=sha(NEW)
    NEW.with_suffix('.sha256').write_text(f'{digest}  {NEW.name}\n')
    print(digest,NEW,flush=True)


if __name__=='__main__':
    build()
