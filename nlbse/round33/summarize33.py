"""Verify published measurements, held-out artifacts and diagnostic ceilings."""
import json

from select33 import HERE, ROOT, PARTS, read, sha


def main():
    baseline=read(ROOT/'round32/release_summary.json')
    archive=ROOT/'submission/NLBSE2027_round32_rule_calibration_tool.zip'
    assert sha(archive)==baseline['archive_sha256']
    selector=read(HERE/'selection.json')
    assert sha(HERE/'table.csv')==selector['table_sha256']
    assert selector['eligible']==0 and selector['selected'] is None
    for split in PARTS:
        area=HERE/'points'/split
        record=read(area/'receipt.json')
        assert record['exact_replay'] and sha(HERE/'points33.py')==record['code_sha256']
        for name,value in record['phases'].items():
            assert sha(area/f'{name}.npz')==value
        folder=HERE/'comparison'/split
        assert sha(folder/'results.json')==selector['tables'][split]
        assert sha(folder/'predictions.npz')==read(folder/'results.json')['predictions_sha256']
    pairs=HERE/'matched_veto'
    fit=read(pairs/'results.json')
    assert sha(HERE/'matched_veto33.py')==fit['code_sha256']
    for split,record in fit['analysis'].items():
        assert sha(pairs/f'{split}.npz')==record['predictions_sha256']
        assert sha(HERE/'points'/split/'receipt.json')==record['points_sha256']
        assert sha(ROOT/'round30/tuned_teacher'/split/'oof.npz')==record['teacher_sha256']
    target=HERE/'public'
    report=read(target/'results.json')
    audit=read(target/'audit.json')
    ceiling=read(target/'ceiling.json')
    assert audit['benchmark_sha256']==sha(target/'results.json')
    assert audit['oracle_sha256']==sha(target/'ceiling.json')
    assert ceiling['endpoints_sha256']==sha(target/'full_endpoints.npz')
    assert report['source']['selection_sha256']==sha(HERE/'joint_selection.json')
    for name,values in report['comparison'].items():
        assert sha(target/f'{name}.npz')==values['predictions_sha256']
    assert report['quality']['frozen']['delta_tp']==0 and report['quality']['frozen']['delta_fp']==1
    assert sum(x>0. for x in report['paired'])==1 and audit['attempted_extra_queries']==181
    assert ceiling['signed_reference'] and ceiling['diagnostic_only']
    best=ceiling['scores']['any_budget_any_shared_threshold']
    profile=read(ROOT/'round32/public/results.json')['results']['candidate']
    pay=.2*profile['cost']['analytic_gflops']/3_000_000
    seconds_limit=(best['with_zero_cost']-.97-pay)/.0002
    summary=dict(release_score=baseline['score'],round33_scores={k:v['score'] for k,v in report['comparison'].items()},
                 candidate_winning_pairs=1,candidate_total_pairs=len(report['paired']),target=.97,
                 oracle=ceiling['scores'],matched_veto_sha256=sha(pairs/'results.json'),maximum_measured_seconds_at_oracle_f1_and_current_gflops=seconds_limit,
                 sha256={k:sha(target/k) for k in ('results.json','audit.json','ceiling.json')})
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
