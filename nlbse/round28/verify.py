"""Check persisted forecasts and report the verified operating point."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as fd:
        return hashlib.file_digest(fd,'sha256').hexdigest()


def verified_table(where):
    events=read(where/'results.json')
    assert sha(where/'policy.json')==events['policy_sha256']
    for v in events['results'].values():
        assert sha(where/v['csv'])==v['predictions_sha256']
        assert all(median(t[k] for t in v['trials'])==amount for k,amount in v['cost'].items())
    return events


def main():
    selected='calibration_only'
    source=HERE/'threshold'
    archive=ROOT/'submission/NLBSE2027_round28_calibrated_tool.zip'
    report=verified_table(source)
    assert all(x>0 for x in read(source/'selection.json')['pairs_minus_point'])
    extra=HERE/'recover_public'
    if (extra/'release_selection.json').exists():
        r=verified_table(extra)
        choice=read(extra/'release_selection.json')['selected']
        if choice!='reference':
            selected,source,report=choice,extra,r
            archive=ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
    baseline=report['results']['reference']
    current=report['results'][selected]
    summary=read(source/'release_verification/receipt.json')
    assert summary['archive_sha256']==sha(archive)
    with (source/'release_verification/release_predictions.csv').open(newline='') as f, (
            source/current['csv']).open(newline='') as g:
        outcomes=0
        for a,b in zip(csv.DictReader(f),csv.DictReader(g),strict=True):
            assert (a['vulnerability'],a['MAT'])==(b['pred_vulnerability'],b['pred_MAT'])
            outcomes+=2
    assert outcomes==summary['identical_binary_decisions']==61310
    for tag in ('head','sparse','canonical'):
        for part in ('test_a','test_b','official'):
            path=HERE/'studies'/tag/part
            p=read(path/'results.json')
            assert p['predictions_sha256']==sha(path/'predictions.npz')
            assert p['weights_sha256']==sha(path/'readouts.joblib')
            if part!='official':
                for row in p['options'].values():
                    assert all(v['delta_fp']==0 for v in row['project'].values())
                    assert sum(v['delta_tp'] for v in row['project'].values())==row['matched']['delta_tp']
    confusion=current['frozen']['after']
    points=current['score']
    if source.name=='threshold':
        fp,fn=81,389
    else:
        fp=81+current['frozen']['delta_fp']
        fn=389-current['frozen']['delta_tp']
    def at(t,p,n):
        return points+.3*(2*t/(2*t+p+n)-current['quality']['vulnerability']['f1'])
    value=dict(selected=selected,archive=archive.name,archive_sha256=sha(archive),score=points,
               paired_reference_score=baseline['score'],remaining_goal_gap=.97-points,
               quality=current['quality'],remaining_exact_negative=dict(fp=fp,fn=fn),
               score_when_all_those_fn_recovered=at(confusion['tp']+fn,confusion['fp'],confusion['fn']-fn),
               score_when_outcome_cohort_is_perfect=at(confusion['tp']+fn,confusion['fp']-fp,confusion['fn']-fn),
               identical_binary_decisions=outcomes)
    (HERE/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    main()
