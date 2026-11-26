"""Verify pinned weights and recheck the held-out FP and TP counts."""
import csv
from pathlib import Path
import sys

import numpy as np

from embed47 import HERE,ROOT,MODEL,DEST as EMB,WINDOWS,source,REVISION,f,checksum,json_at,write_json
from fit47 import PARTS,DEST as FIT,CONTROL,load,parts
from compare47 import DEST
sys.path.insert(0,str(ROOT/'round42'))
from audit42 import assert_equal


def assets():
    metadata=source()
    weights=MODEL/'pytorch_model.bin'
    view=EMB/'views.json'
    token=json_at(view)
    assert token['revision']==REVISION and token['upstream']==checksum(MODEL/'upstream.json')
    assert token['lfs']==metadata[weights.name]['lfs']['sha256']==checksum(weights)
    assert token['code_sha256']==checksum(HERE/'embed47.py')
    assert token['window_receipt']==checksum(WINDOWS/'receipt.json')
    assert token['roster_sha256']==checksum(WINDOWS/'rows.npz')
    for name,digest in token['assets'].items():
        assert digest==checksum(MODEL/name)
    proof={}
    for c,k in token['views'].items():
        p=EMB/f'tokens_{c}.npz'
        assert k['file_sha256']==checksum(p)
        with np.load(p) as z,np.load(WINDOWS/f'qwen_{c}.npz') as old:
            assert np.array_equal(z['full'],old['full'])
            assert np.array_equal(np.diff(z['ends']),z['lengths'])
            assert int(z['full'].sum())==k['full'] and int(z['lengths'].sum())==k['total_tokens']
        note=EMB/c/'receipt.json'
        r=json_at(note)
        assert r['views_sha256']==checksum(view)
        assert r['model_sha256']==token['lfs']
        assert r['rows']==token['rows']
        assert r['example_max_error']<.0008
        for item,d in r['hashes'].items():
            assert checksum(EMB/c/item)==d
        proof[c]=checksum(note)
    return proof


def validate(rotation,plan,proof):
    filepath=FIT/rotation/'receipt.json'
    ticket=json_at(filepath)
    assert ticket['code_sha256']==checksum(HERE/'fit47.py')
    assert ticket['score_sha256']==checksum(filepath.with_name('scores.npz'))
    assert ticket['weights_sha256']==checksum(filepath.with_name('weights.joblib'))
    assert ticket['prior_sha256']==checksum(CONTROL/rotation/'receipt.json')
    assert ticket['pairs_sha256']==checksum(ROOT/'round37/matched'/f'{rotation}.npz')
    assert ticket['embeddings']==proof
    matrix,_,stamp=load(rotation)
    assert stamp==ticket['matrix_sha256']
    _,held=parts(matrix)
    target=DEST/rotation/'results.json'
    assert checksum(target)==plan['sources'][rotation]
    study=json_at(target)
    assert study['code_sha256']==checksum(HERE/'compare47.py')
    assert study['fit_sha256']==checksum(filepath)
    long=ROOT/'round46/length'/rotation/'results.json'
    assert study['technical_length_sha256']==checksum(long)
    prediction=target.with_name('predictions.npz')
    assert study['predictions_sha256']==checksum(prediction)
    base=held['signed']>0.
    y=held['y']
    universe=np.ones(len(base),bool)
    groups={k:held['project']==k for k in np.unique(held['project'])}
    with np.load(prediction) as h:
        assert np.array_equal(h['ids'],held['ids'])
        assert np.array_equal(h['base'],base) and np.array_equal(h['truth'],y)
        for key,v in study['options'].items():
            for mode in ('fixed','fp','tp'):
                assert_equal(v[mode]['held'],key+'/'+mode,universe,y,base,h[key+'_'+mode])
                if rotation!='official' and mode!='fixed':
                    candidate=h[key+'_same_'+mode]
                    selected=v['matched'][mode]
                    assert_equal(selected['change'],key+'/'+mode,universe,y,base,candidate)
                    for group,local in groups.items():
                        out=assert_equal(selected['projects'][group],key+'/'+group,local,y,base,candidate)
                        assert out['delta_'+mode]==0
    return dict(options=len(study['options']),sha256=checksum(prediction))


def main():
    proof=assets()
    p=DEST/'selection.json'
    plan=json_at(p)
    assert plan['code_sha256']==checksum(HERE/'compare47.py')
    assert plan['csv_sha256']==checksum(DEST/'comparison.csv')
    with (DEST/'comparison.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==plan['tested']
    assert any(r['eligible']=='True' for r in rows)==any(plan['selection'].values())
    base=json_at(ROOT/'round38/release_summary.json')
    assert checksum(ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip')==base['archive_sha256']==plan['archive_sha256']
    result=dict(previous=base['score'],gap=base['target_gap'],model_revision=REVISION,
                selection=plan['selection'],tested=plan['tested'],
                roles={k:validate(k,plan,proof) for k in PARTS},table_sha256=checksum(p),
                code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
