"""Recount scored decisions and verify the public comparison against the Round 37 archive."""
from pathlib import Path
from statistics import median

import numpy as np

from data38 import HERE, ROOT, R37, f, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes


def main():
    scan = HERE/'public_primary'
    path = scan/'results.json'
    source = json_at(path)
    plan = HERE/'selection.json'
    assert source['selected']==json_at(plan)['selected']
    archive = ROOT/'submission/NLBSE2027_round37_deferred_sections_tool.zip'
    previous = R37/'public_semantic_deferred/candidate.npz'
    baseline = json_at(R37/'release_summary.json')
    proof = source['receipt']
    refs = {'archive':archive,'plan':plan,'data':PUBLIC,'anchor':previous,'code':HERE/'live38.py',
            'runtime':ROOT/'round35/live35.py','model':HERE/'comparison/official/models.joblib'}
    for key,entry in refs.items():
        assert checksum(entry)==proof[key+'_sha256']
    assert proof['archive_sha256']==baseline['archive_sha256']
    _,y,meta = load(PUBLIC,audit=True)
    ordered = {}
    paid = {}
    for name,item in source['comparison'].items():
        p = scan/f'{name}.npz'
        assert checksum(p)==item['predictions_sha256']
        with np.load(p) as z:
            output,touched=z['pred'],z['scope']
            assert np.array_equal(z['truth'],y)
        metric = f.metric(y,output)
        assert metric==item['quality']
        paid[name] = []
        for trial in item['trials']:
            spot = f.competition_formula(metric['mean']['f1'],trial['forward_seconds'],trial['analytic_gflops'])
            assert spot==trial['score']
            paid[name].append(spot)
        assert len(paid[name])==5 and median(paid[name])==item['score']
        for key,v in item['cost'].items():
            assert median(q[key] for q in item['trials'])==v
        ordered[name] = output,touched
    old = ordered['reference'][0]
    new,where = ordered['candidate']
    with np.load(previous) as z:
        assert np.array_equal(old,z['pred'])
    assert np.array_equal(old[:,1],new[:,1]) and np.array_equal(old[~where],new[~where])
    assert f.comparison(y[:,0],old[:,0],new[:,0])==source['changes']['frozen']
    paired = [b-a for a,b in zip(paid['reference'],paid['candidate'],strict=True)]
    assert paired==source['paired']
    same=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    v = source['comparison']['candidate']
    ref = source['comparison']['reference']
    wins = sum(x>0. for x in paired)
    improved = wins>=4 and v['score']>max(ref['score'],baseline['score']) and v['quality']['mean']['f1']>ref['quality']['mean']['f1']
    result = dict(rows=len(old),binary_decisions=old.size,changed_bits=int((old!=new).sum()),
                  wins=wins,pairs=len(paired),score=v['score'],control=ref['score'],prior=baseline['score'],
                  changed_region=int(where.sum()),interval=bootstrap_delta(y,old,new,same),
                  promoted=improved,reached_target=improved and v['score']>=.97,
                  profile_sha256=checksum(path),code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
