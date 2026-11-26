"""Summarize signed Round 31 results and score sensitivity without relabeling data."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARTS = ('test_a','test_b','official')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def audit_materials():
    paths = []
    for part in PARTS:
        raw = HERE/'trained'/part
        cfg = read(raw/'fit.json')
        assert cfg['config']['code_sha256']==sha(HERE/'continue_fit.py')
        for name,info in cfg['arms'].items():
            assert sha(raw/f'{name}.pt')==info['sha256']
        dest = HERE/'continuation'/part
        data = read(dest/'results.json')
        assert data['reference_replayed'] and data['fit_receipt_sha256']==sha(raw/'fit.json')
        assert data['code_sha256']==sha(HERE/'continue_eval.py')
        assert data['predictions_sha256']==sha(dest/'predictions.npz')
        assert data['readout_sha256']==sha(dest/'readouts.joblib')
        paths.append(dest/'results.json')
    for mode, runner in ((HERE/'mat','transfer.py'),(HERE/'mat/placement','placement.py')):
        config = read(mode/'policy.json')
        assert config['partition_sha256']==sha(ROOT/'round13/partition.npz')
        assert config['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
        assert config['code_sha256']==sha(HERE/'mat'/runner)
        if 'heads_sha256' in config:
            assert config['heads_sha256']==sha(mode/'heads.joblib')
        for name, info in read(mode/'results.json').items():
            assert info['predictions_sha256']==sha(mode/f'{name}_predictions.npz')
    mat = HERE/'mat'
    provenance = read(mat/'fit_provenance.json')
    assert provenance['partition_sha256']==sha(ROOT/'round13/partition.npz')
    assert provenance['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
    assert provenance['code_sha256']==sha(mat/'audit_fit.py')
    assert provenance['source_audit_code_sha256']==sha(ROOT/'round18/mat_residual/source_audit.py')
    pair = mat/'paired_provenance'
    model = read(pair/'policy.json')
    assert model['audit_sha256']==sha(mat/'fit_provenance.json')
    assert model['residual_sha256']==sha(mat/'placement/heads.joblib')
    assert model['code_sha256']==sha(mat/'paired_provenance.py')
    assert model['weights_sha256']==sha(pair/'weights.joblib')
    for name, info in read(pair/'results.json').items():
        assert info['predictions_sha256']==sha(pair/f'{name}_predictions.npz')
    return [read(path)['arms'] for path in paths]


def summarize_models(studies):
    a,b,official = studies
    a_fp = a['head']['core']['matched']['delta_tp']
    b_fp = b['head']['core']['matched']['delta_tp']
    rows = []
    feasible = []
    for arm, options in official.items():
        for name, policy in options.items():
            if name in ('held','cal'):
                continue
            first, second = a[arm][name]['matched'], b[arm][name]['matched']
            if first['delta_fp']!=0 or second['delta_fp']!=0:
                raise ValueError('The project comparison is not matched')
            for mode, result in policy.items():
                internal, review = result['cal'], result['frozen']
                row = dict(encoder=arm,readout=name,policy=mode,test_a_tp=first['delta_tp'],
                           test_b_tp=second['delta_tp'],test_a_tp_over_retuned=first['delta_tp']-a_fp,
                           test_b_tp_over_retuned=second['delta_tp']-b_fp,cal_tp=internal['delta_tp'],
                           cal_fp=internal['delta_fp'],review_tp=review['delta_tp'],review_fp=review['delta_fp'],
                           cal_f1_gain=internal['f1_after']-internal['f1_before'],
                           review_f1_gain=review['f1_after']-review['f1_before'],
                           test_a_scanner_changes=a[arm]['held']['scan_changes'],
                           test_b_scanner_changes=b[arm]['held']['scan_changes'])
                rows.append(row)
                if (min(row['test_a_tp_over_retuned'],row['test_b_tp_over_retuned'])>0 and
                    row['cal_f1_gain']>0 and internal['delta_fp']<=0 and row['review_f1_gain']>=0):
                    feasible.append(row)
    with (HERE/'continuation_comparison.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return dict(candidates=len(rows),retuned_reference_tp=[a_fp,b_fp],public_eligible=feasible)


def sensitivity():
    live = ROOT/'round28/recover_public'
    approved = read(live/'release_selection.json')['selected']
    validation = read(live/'results.json')['results'][approved]
    path = live/validation['csv']
    assert sha(path)==validation['predictions_sha256']
    release = ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
    assert sha(release)==release.with_suffix('.sha256').read_text().split()[0]
    cells = {label:{'tp':0,'fp':0,'fn':0,'tn':0} for label in ('vulnerability','MAT')}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for gold,pred in (('label_vulnerability','pred_vulnerability'),('label_mat','pred_MAT')):
                key = pred.removeprefix('pred_')
                y, bit = int(row[gold]),int(row[pred])
                cells[key][('tn','fp','fn','tp')[2*y+bit]] += 1
    vf,mf = cells.values()
    assert sum(vf.values())==sum(mf.values())==30_655
    time = validation['cost']['forward_seconds']
    gflops = validation['cost']['analytic_gflops']
    efficiency = .2*(1.-time/1000)+.2*(1.-gflops/3_000_000)

    def f1(tp, fp, total):
        return 2*tp/(tp+fp+total)

    def score(mat=0, vulnerability=0, remove_fp=0, budget=efficiency):
        return budget+.3*(f1(vf['tp']+vulnerability,vf['fp']-remove_fp,vf['tp']+vf['fn'])+
                          f1(mf['tp']+mat,mf['fp'],mf['tp']+mf['fn']))

    assert abs(score()-validation['score'])<1e-12
    thresholds = {
        'additional_mat_tp':next(i for i in range(mf['fn']+1) if score(mat=i)>=.97),
        'additional_vulnerability_tp':next(i for i in range(vf['fn']+1) if score(vulnerability=i)>=.97),
        'remove_vulnerability_fp':next(i for i in range(vf['fp']+1) if score(remove_fp=i)>=.97),
    }
    return dict(public_score=score(),counts=cells,target=.97,target_gap=.97-score(),same_cost_minimum_changes=thresholds,
                no_cost_at_current_f1=score(budget=.4),condition_18_mat_tp=score(mat=18),
                condition_19_mat_tp=score(mat=19),condition_20_mat_tp=score(mat=20),
                condition_10_mat_tp_minimum_vulnerability_fp_removal=next(i for i in range(vf['fp']+1)
                            if score(mat=10,remove_fp=i)>=.97),
                scored_seconds=time,gflops=gflops,archived_predictions_sha256=sha(path),archive_sha256=sha(release))


def main():
    models = summarize_models(audit_materials())
    audit = read(HERE/'reference_audit.json')
    assert audit['checked_decisions']==61_310
    assert audit['code_sha256']==sha(HERE/'audit29.py')
    assert audit['readout_sha256']==sha(HERE/'reference_audit.npz')
    result = dict(continuation=models, target=sensitivity(),
                  reference_audit_sha256=sha(HERE/'reference_audit.json'))
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
