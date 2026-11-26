"""Package the verified deferred regional correction as a self-contained tool."""
from io import BytesIO, StringIO
from inspect import getsource
from pathlib import Path
import csv
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

import joblib
import numpy as np

from section37 import HERE, ROOT, NATIVE, SECTIONS, OUTPUTS, SEAM_TOKENS, checksum, prepare, section_scores
from live37 import encode
from eval37 import VIEWS, features
from broad35 import features as score_features, regimes
from deferred37 import ScopedStates
from fast37 import TwoRegions
from prefilter34 import GateFilter

BASE = ROOT/'submission/NLBSE2027_round34_pooled_projection_tool.zip'
ARCHIVE = ROOT/'submission/NLBSE2027_round37_deferred_sections_tool.zip'
ENTRY = 'nlbse/round25/predict_sections.py'
SETTINGS = 'nlbse/round25/entry.json'
CAPSULE = 'nlbse/round25/section_adjust37.py'
ADDITION = 'nlbse/round25/deferred_sections.joblib'

HEADER = """import numpy as np
import torch
from bisect import bisect_left
from time import perf_counter

from approx_runtime34 import design, SPANS
from conditional import query_value
from section_runtime import opening
from predict_factor import lexical_features, ordered_batches, batch_plan, batch, small, full_context
from rule_model import pooling
"""

APPLY = """
def refine(texts, original, margin, values, estimate, no, yes, prior, known, flag,
           bits, config, policy, output, pooled, settings, retained):
    observed_negative = known & ~flag
    scope = observed_negative & (((original>0.) & (margin<=0.)) | ((margin>0.) & (margin<SPANS)))
    if pooled['width'] is not None:
        scope &= abs(margin)<pooled['width']
    active = observed_negative.copy()
    spec = settings['config']
    if not active.any():
        return bits
    t = config['threshold']
    utility = query_value(policy,prior,no,yes,estimate['w'],config['gate']['base'])
    pre = np.column_stack((values,estimate['w'],no,yes,prior,utility))
    state = dict(pre=pre, unqueried=original, queried=margin,
                 length=np.fromiter(map(len,texts),np.int32))
    rank = margin.copy()
    if scope.any():
        view = design({k:x[scope] for k,x in state.items()},t,output[scope,:2])
        w = pooled['strength']
        rank[scope] = (1.-w)*rank[scope]+w*pooled['calibrator'].decision_function(view)-pooled['cut']
    result = bits.copy()
    result[scope] = rank[scope]>0.
    if spec['width'] is not None:
        active &= abs(rank)<spec['width']
    if active.any():
        view = retained.features({k:x[active] for k,x in state.items()},t,spec['mode'],output[active,2:])
        w = spec['strength']
        result[active] = (1.-w)*rank[active]+w*settings['model'].decision_function(view)>spec['cut']
    return result
"""


def replace(entry):
    patches = {
        'from approx_runtime34 import encode, regional_adjust':
            'from section_adjust37 import ScopedStates, encode, refine',
        '    p,aux,extra=encode(model,tokenizer,texts,projection)':
            '''    selected=HERE/"deferred_sections.joblib"
    pf.require_hash(selected,settings["deferred_sections_sha256"])
    regional=joblib.load(selected)
    capture=ScopedStates((config,pf.Policy(**config["policy"]),heads),config["route"])
    p,aux,extra,_=encode(model,tokenizer,texts,projection,True,deferred=capture)''',
        '''    vr=regional_adjust(texts,first_margin,negative_margin,values,est,no,yes,prior,known,flag,
                       vr,config,policy,extra,trained)''':
            '''    vr=refine(texts,first_margin,negative_margin,values,est,no,yes,prior,known,flag,
              vr,config,policy,extra,trained,regional,capture)''',
    }
    for old,new in patches.items():
        if entry.count(old)!=1:
            raise ValueError('The frozen CLI changed')
        entry = entry.replace(old,new)
    return entry


def main():
    plan = HERE/'selection.json'
    audit = json.loads((HERE/'audit.json').read_text())
    selected = audit['selected']
    if selected!='public_semantic_deferred' or not audit['results'][selected]['passed']:
        raise ValueError('The deferred implementation has no verified gain')
    destination = HERE/selected
    profile = destination/'results.json'
    c = audit['results'][selected]
    assert c['public_profile_sha256']==checksum(profile)
    verified = json.loads(profile.read_text())
    proof = verified['receipt']
    assert proof['archive_sha256']==checksum(BASE)
    assert c['heldout_stage_sha256']==proof['selection_sha256']==checksum(plan)
    coeff = HERE/'comparison/official/readouts.joblib'
    assert proof['model_sha256']==checksum(coeff)
    config = verified['selected']
    model = joblib.load(coeff)[config['source'],config['mode']]
    packed = BytesIO()
    joblib.dump(dict(config=config,model=model),packed,compress=3)
    payload = packed.getvalue()
    code = [HEADER,f'NATIVE = {NATIVE!r}\nSECTIONS = {SECTIONS!r}\nOUTPUTS = {OUTPUTS!r}\nSEAM_TOKENS = {SEAM_TOKENS!r}\nVIEWS = {VIEWS!r}']
    code += [getsource(x) for x in (GateFilter,prepare,regimes,score_features)]
    code += ['score_features = features']
    code += [getsource(x) for x in (features,section_scores,TwoRegions,ScopedStates,encode)]
    code.append(APPLY)
    module = '\n\n'.join(code)
    compile(module,CAPSULE,'exec')
    public = destination/'candidate.npz'
    assert checksum(public)==verified['comparison']['candidate']['predictions_sha256']
    expected = StringIO(newline='')
    with np.load(public) as z:
        w = csv.writer(expected)
        w.writerow(('row','pred_vulnerability','pred_MAT'))
        w.writerows((i,int(a),int(b)) for i,(a,b) in enumerate(z['pred']))
    avoid = {ENTRY,SETTINGS,'README.md','examples/public_expected.csv'}
    target = ARCHIVE.with_suffix('.part')
    with ZipFile(BASE) as old,ZipFile(target,'w',ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in avoid:
                new.writestr(item,old.read(item))
        new.writestr(ENTRY,replace(old.read(ENTRY).decode()))
        info = json.loads(old.read(SETTINGS))
        info['deferred_sections_sha256'] = hashlib.sha256(payload).hexdigest()
        new.writestr(SETTINGS,json.dumps(info,indent=2)+'\n')
        new.writestr(CAPSULE,module)
        new.writestr(ADDITION,payload)
        new.writestr('examples/public_expected.csv',expected.getvalue())
        new.write(profile,'nlbse/round37/public_profile.json')
        new.writestr('README.md',f"""# NLBSE 2027 compact conditional code classifier

Predicts vulnerability and MAT from the Function column. The tool retains
bounded first-pass token states and computes the declaration and body
projections only for selected observed negative scanner results.

Use Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Measured local score {c['score']:.6f} on 30,655 public functions with an RTX 4090.
The retained state buffer is capped at 7 GiB and also at 58% of currently free
GPU memory, with per-batch eager projection when the buffer is full.
""")
    target.replace(ARCHIVE)
    digest = checksum(ARCHIVE)
    ARCHIVE.with_suffix('.sha256').write_text(f'{digest}  {ARCHIVE.name}\n')
    print(ARCHIVE,digest,flush=True)


if __name__=='__main__':
    main()
