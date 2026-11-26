"""Package the supported positive-branch correction in the verified single-encoder tool."""
import csv
from hashlib import sha256
from inspect import getsource
from io import BytesIO, StringIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import joblib
import numpy as np

from data38 import HERE, ROOT, checksum, json_at
from rank38 import OFFSET, positive_features, predict, rule_order
from agreement32 import features as agreement

BASE = ROOT/'submission/NLBSE2027_round37_deferred_sections_tool.zip'
ARCHIVE = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
ENTRY = 'nlbse/round25/predict_sections.py'
CONFIG = 'nlbse/round25/entry.json'
RULE = 'nlbse/round25/rule_margin32.py'
MODULE = 'nlbse/round25/paired_rule38.py'
WEIGHTS = 'nlbse/round25/paired_rule38.joblib'

WRAPPER = """
def on_positive(texts, config, policy, values, estimate, no, yes, prior, first_margin,
                current, state, raw, aux, bits, settings):
    selected = settings['selected']
    mask = state['known'] & state['flag']
    width = selected['width_positive']
    if width is not None:
        mask &= abs(current)<width
    if not mask.any():
        return bits
    utility = query_value(policy,prior,no,yes,estimate['w'],config['gate']['base'])
    pre = np.column_stack((values,estimate['w'],no,yes,prior,utility))
    length = np.fromiter(map(len,texts),np.int32)
    data = dict(queried=current[mask],unqueried=first_margin[mask],pre=pre[mask],
                latest=current[mask],length=length[mask])
    x = positive_features(data,config['threshold'],aux[mask,:20],raw[mask],selected['positive']!='control')
    rank = predict(settings['model'],x,current[mask])
    m = selected['mix']
    answer = bits.copy()
    answer[mask] = (1.-m)*current[mask]+m*rank>selected['cut']
    return answer
"""


def patch(source, changes):
    for before,after in changes:
        if source.count(before)!=1:
            raise ValueError('An expected archive entry changed')
        source = source.replace(before,after)
    return source


def adjust_rule(src):
    return patch(src,(
        ('counts, settings):','counts, settings, details=False):'),
        ('        return original','        return (original,margin) if details else original'),
        ("    result[relevant] = candidate>settings['cut']\n    return result",
         "    margin[relevant] = candidate-settings['cut']\n    result[relevant] = margin[relevant]>0.\n    return (result,margin) if details else result")
    ))


def patch_entry(src):
    return patch(src,(
        ('from rule_margin32 import adjust','from rule_margin32 import adjust\nfrom paired_rule38 import on_positive'),
        ('    vr=adjust(texts,state,config["threshold"],vr,corrected,score,positive["online"]["cut"],evidence,rule_calibration)',
         '    vr,positive_margin=adjust(texts,state,config["threshold"],vr,corrected,score,positive["online"]["cut"],evidence,rule_calibration,details=True)'),
        ('    return np.column_stack((vr,canonical_mat(texts))).astype(int)',
         '''    path=HERE/"paired_rule38.joblib"
    pf.require_hash(path,settings["paired_rule38_sha256"])
    latest=joblib.load(path)
    vr=on_positive(texts,config,policy,values,est,no,yes,prior,first_margin,positive_margin,state,evidence,aux,vr,latest)
    return np.column_stack((vr,canonical_mat(texts))).astype(int)''')
    ))


def main():
    audit = json_at(HERE/'audit.json')
    path = HERE/'public_primary/results.json'
    assert audit['promoted'] and audit['profile_sha256']==checksum(path)
    profile = json_at(path)
    chosen = profile['selected']
    assert checksum(BASE)==profile['receipt']['archive_sha256']
    assert checksum(HERE/'selection.json')==profile['receipt']['plan_sha256']
    models = HERE/'comparison/official/models.joblib'
    assert checksum(models)==profile['receipt']['model_sha256']
    assert chosen['positive']!='off' and chosen['negative']=='off' and chosen['deployable']
    coef = joblib.load(models)['positive',chosen['positive']]
    p = BytesIO()
    joblib.dump(dict(selected=chosen,model=coef),p,compress=3)
    frozen = p.getvalue()
    expected = StringIO(newline='')
    decisions = path.with_name('candidate.npz')
    assert checksum(decisions)==profile['comparison']['candidate']['predictions_sha256']
    with np.load(decisions) as z:
        table = csv.writer(expected)
        table.writerow(('row','pred_vulnerability','pred_MAT'))
        table.writerows((i,int(a),int(b)) for i,(a,b) in enumerate(z['pred']))
    _,indices = rule_order()
    head = f'''import numpy as np
from scipy.special import logit
from conditional import query_value
from section_adjust37 import score_features

OFFSET = {OFFSET!r}

def rule_order():
    return None, {indices!r}
'''
    capsule = '\n\n'.join((head,getsource(agreement),'agreement = features',getsource(positive_features),
                          getsource(predict),WRAPPER))
    compile(capsule,MODULE,'exec')
    skipped = {ENTRY,CONFIG,RULE,'README.md','examples/public_expected.csv'}
    scratch = ARCHIVE.with_suffix('.part')
    with ZipFile(BASE) as old,ZipFile(scratch,'w',ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in skipped:
                new.writestr(item,old.read(item))
        entry = patch_entry(old.read(ENTRY).decode())
        rule = adjust_rule(old.read(RULE).decode())
        compile(entry,ENTRY,'exec')
        compile(rule,RULE,'exec')
        new.writestr(ENTRY,entry)
        new.writestr(RULE,rule)
        conf = json.loads(old.read(CONFIG))
        conf['paired_rule38_sha256'] = sha256(frozen).hexdigest()
        new.writestr(CONFIG,json.dumps(conf,indent=2)+'\n')
        new.writestr(MODULE,capsule)
        new.writestr(WEIGHTS,frozen)
        new.writestr('examples/public_expected.csv',expected.getvalue())
        new.write(path,'nlbse/round38/public_profile.json')
        new.writestr('README.md',f"""# NLBSE 2027 compact conditional code classifier

Reads the Function column and predicts vulnerability and MAT. The model retains
first-pass token states for selected negative scanner results. A linear residual
ranker trained on within-file pairs refines positive results using existing
rule predictions and first-hit information.

Use Python 3.11 or 3.12, CUDA PyTorch 2.8 and Node.js 22.

```bash
python3 -m pip install -r requirements-inference.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \
  python3 {ENTRY} --input functions.csv --output predictions.csv
```

Measured local score {audit['score']:.6f} on 30,655 public functions on an RTX 4090.
""")
    scratch.replace(ARCHIVE)
    digest = checksum(ARCHIVE)
    ARCHIVE.with_suffix('.sha256').write_text(f'{digest}  {ARCHIVE.name}\n')
    print(ARCHIVE,digest,flush=True)


if __name__=='__main__':
    main()
