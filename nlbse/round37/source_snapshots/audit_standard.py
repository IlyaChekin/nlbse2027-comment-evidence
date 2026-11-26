"""Recompute five public scores and compare binary decisions to the released archive."""
from pathlib import Path
from statistics import median

import numpy as np

from section37 import HERE, ROOT, f, checksum, json_at, write_json
from live37 import DEST, PUBLIC
from pipeline import load
from round2_report import bootstrap_delta
from students import group_codes
from eval35 import cut


def check(output):
    path = output/'results.json'
    info = json_at(path)
    signed = info['receipt']
    plan = HERE/'selection.json'
    choice = json_at(plan)
    reference = ROOT/'round34/public_approx/candidate.npz'
    release = json_at(ROOT/'round34/release_summary.json')
    archive = ROOT/'submission/NLBSE2027_round34_pooled_projection_tool.zip'
    arm = info['arm']
    expected = choice['selected'] if arm=='primary' else choice['best'][arm]
    assert info['selected']==expected
    for name,file in (('selection',plan),('reference',reference),('public',PUBLIC),('archive',archive),
                      ('model',HERE/'comparison/official/readouts.joblib'),('code',HERE/'live37.py'),
                      ('core',ROOT/'round35/live35.py')):
        assert signed[name+'_sha256']==checksum(file)
    assert signed['archive_sha256']==choice['base_archive_sha256']==release['archive_sha256']
    _,y,m = load(PUBLIC,audit=True)
    predicted = {}
    individual = {}
    for role,item in info['comparison'].items():
        file = output/f'{role}.npz'
        assert checksum(file)==item['predictions_sha256']
        with np.load(file) as z:
            assert np.array_equal(z['truth'],y)
            bits,mask,margin = z['pred'],z['scope'],z['margin']
        scores = []
        metric = f.metric(y,bits)
        assert metric==item['quality']
        assert len(item['trials'])==5
        for t in item['trials']:
            s = f.competition_formula(metric['mean']['f1'],t['forward_seconds'],t['analytic_gflops'])
            assert s==t['score']
            scores.append(s)
        for key,value in item['cost'].items():
            assert median(t[key] for t in item['trials'])==value
        assert median(scores)==item['score']
        predicted[role] = bits,mask,margin
        individual[role] = scores
    base,candidate = predicted['reference'][0],predicted['candidate'][0]
    field,margin = predicted['candidate'][1:]
    with np.load(reference) as original:
        assert np.array_equal(base,original['pred'])
    assert np.array_equal(base[:,1],candidate[:,1])
    assert np.array_equal(base[~field],candidate[~field])
    gold,prior = y[:,0].astype(bool),base[:,0]
    assert f.comparison(gold,prior,candidate[:,0])==info['quality']['frozen']
    for scheme in ('fp','tp'):
        bit = prior.copy()
        c = cut(gold[field],prior[field],margin[field],scheme)
        bit[field] = margin[field]>c
        assert f.comparison(gold,prior,bit)==info['quality'][scheme]
    paired = [v-u for u,v in zip(individual['reference'],individual['candidate'],strict=True)]
    assert paired==info['paired']
    wins = sum(x>0 for x in paired)
    h = info['comparison']['candidate']
    r = info['comparison']['reference']
    qualifies = wins>=4 and h['score']>max(r['score'],release['score']) and h['quality']['mean']['f1']>r['quality']['mean']['f1']
    clusters = group_codes(zip(m['Projectname'],m['Filepath'],strict=True))
    check_ = dict(passed=qualifies,reached_target=qualifies and h['score']>=.97,rows=len(base),binary_decisions=base.size,
                  selected=info['selected'],score=h['score'],control=r['score'],wins=wins,pairs=len(paired),
                  changed_predictions=int((base!=candidate).sum()),interval=bootstrap_delta(y,base,candidate,clusters),
                  heldout_stage_sha256=checksum(plan),public_profile_sha256=checksum(path),code_sha256=checksum(Path(__file__)))
    return check_


if __name__=='__main__':
    outputs = (DEST,HERE/'public_native',HERE/'public_semantic')
    records = {path.name:check(path) for path in outputs if (path/'results.json').is_file()}
    if not records:
        raise ValueError('No finished public comparison')
    promoted = max((name for name,r in records.items() if r['passed']),
                   key=lambda name:records[name]['score'],default=None)
    write_json(HERE/'audit.json',dict(selected=promoted,results=records,code_sha256=checksum(Path(__file__))))
    print(promoted,records,flush=True)
