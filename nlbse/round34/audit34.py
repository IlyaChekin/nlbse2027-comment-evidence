"""Verify the held-out selection, public bit-for-bit replay and profiling evidence."""
from statistics import median

import numpy as np

from sections34 import HERE, ROOT, f, checksum, json_at, write_json
from pipeline import PUBLIC, load
from round2_report import bootstrap_delta
from students import group_codes


def main():
    prior=ROOT/'round32/public/candidate.npz'
    saved=ROOT/'round32/release_summary.json'
    proof=json_at(saved)
    archive=ROOT/'submission/NLBSE2027_round32_rule_calibration_tool.zip'
    assert checksum(archive)==proof['archive_sha256']
    _,target,meta=load(PUBLIC,audit=True)
    with np.load(prior) as z:
        expected=z['pred']
        assert np.array_equal(z['truth'],target)
    selection=HERE/'selection.json'
    choice=json_at(selection)['selected']
    grouped=group_codes(zip(meta['Projectname'],meta['Filepath'],strict=True))
    verified={}
    prior_candidate=None
    for stage in ('public','public_gated','public_trim','public_deferred','public_approx'):
        folder=HERE/stage
        path=folder/'results.json'
        result=json_at(path)
        is_approx=stage=='public_approx'
        stage_selection=HERE/'selection_approx.json' if is_approx else selection
        assert result['chosen']==(json_at(stage_selection)['selected'] if is_approx else choice)
        receipt=result['receipt']
        if stage=='public':
            source=HERE/'public_unfiltered/live_source.py'
        elif is_approx:
            source=HERE/'live34.py'
        else:
            source=folder/'live_source.py'
        assert receipt['code_sha256']==checksum(source)
        assert receipt['public_sha256']==checksum(PUBLIC)
        assert receipt['selection_sha256']==checksum(stage_selection)
        assert receipt['reference_sha256']==checksum(prior)
        if stage=='public_gated':
            assert receipt['prefilter_sha256']==checksum(HERE/'prefilter34.py')
        if stage=='public_deferred':
            assert receipt['deferred_sha256']==checksum(HERE/'defer34.py')
        series={}
        for role,case in result['comparison'].items():
            raw=folder/f'{role}.npz'
            assert checksum(raw)==case['predictions_sha256']
            with np.load(raw) as z:
                value=z['pred']
                use=z['scope']
                assert np.array_equal(z['truth'],target)
            assert f.metric(target,value)==case['quality']
            cost=case['cost']
            assert all(median(x[k] for x in case['trials'])==v for k,v in cost.items())
            assert f.competition_formula(case['quality']['mean']['f1'],cost['forward_seconds'],
                                         cost['analytic_gflops'])==case['score']
            series[role]=(value,use)
        reference,_=series['reference']
        candidate,scope=series['candidate']
        assert np.array_equal(reference,expected)
        assert np.array_equal(candidate[:,1],expected[:,1])
        assert np.array_equal(reference[~scope],candidate[~scope])
        effect=f.comparison(target[:,0],reference[:,0],candidate[:,0])
        assert effect==result['quality']['frozen']
        if not is_approx:
            if prior_candidate is not None:
                assert np.array_equal(candidate,prior_candidate)
            prior_candidate=candidate
        verified[stage]=dict(candidate=result['comparison']['candidate']['score'],
                             reference=result['comparison']['reference']['score'],active=int(scope.sum()),
                             changed=int((reference!=candidate).sum()),check=checksum(path),
                             pairs=len(result['paired']),wins=sum(x>0. for x in result['paired']),
                             interval=bootstrap_delta(target,reference,candidate,grouped),delta_tp=effect['delta_tp'],
                             delta_fp=effect['delta_fp'],
                             mean_f1_gain=result['comparison']['candidate']['quality']['mean']['f1']-
                                 result['comparison']['reference']['quality']['mean']['f1'])
    compatible=[key for key,item in verified.items() if item['candidate']>max(proof['score'],item['reference']) and
                item['wins']*5>=item['pairs']*4 and item['mean_f1_gain']>0.]
    selected=max(compatible,key=lambda k:verified[k]['candidate'],default=None)
    write_json(HERE/'audit.json',dict(round32_archive_sha256=checksum(archive),latest_release=proof['score'],
               release_candidate=selected,
               selection_sha256=checksum(selection),outcomes=verified))
    print(verified,flush=True)


if __name__=='__main__':
    main()
