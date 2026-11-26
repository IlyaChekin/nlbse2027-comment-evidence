"""Verify experiment receipts and select a candidate before viewing public labels."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=('test_a','test_b','official')
STAGES={
    'evaluation':('eval32.py','models.joblib'),
    'agreement':('eval32.py','models.joblib'),
    'dense_evaluation':('dense_rank.py','rankers.joblib'),
    'calibrated_rules':('subgroups32.py','models.joblib'),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read_stage(stage):
    code,weight=STAGES[stage]
    all_runs={}
    for split in PARTS:
        directory=HERE/stage/split
        path=directory/'results.json'
        detail=read(path)
        assert detail['rotation']==split
        assert detail['code_sha256']==sha(HERE/code)
        assert detail['predictions_sha256']==sha(directory/'predictions.npz')
        assert detail['weights_sha256']==sha(directory/weight)
        if 'runtime_sha256' in detail:
            assert detail['runtime_sha256']==sha(HERE/'risk32.py')
        if stage=='agreement':
            raw=HERE/'aux'/split
            receipt=read(raw/'receipt.json')
            assert receipt['code_sha256']==sha(HERE/'collect_aux.py')
            assert detail['training']['aux_sha256']==receipt['predictions_sha256']==sha(raw/'scores.npz')
        if stage=='dense_evaluation':
            raw=HERE/'dense'
            receipt=read(raw/'receipt.json')
            assert receipt['code_sha256']==sha(HERE/'collect_dense.py')
            assert detail['feature_source']==receipt['archive_sha256']
            assert receipt['rows_sha256']==sha(raw/'rows.npz')
        all_runs[split]=detail['options']
    return all_runs


def f1_gain(data):
    return data['f1_after']-data['f1_before']


def summarize(runs):
    a,b,inside=(runs[k] for k in PARTS)
    measurement=[]
    for key,candidate in inside.items():
        opts=candidate['setting']
        if key not in a or key not in b:
            continue
        limit='all' if opts['width'] is None else f"{opts['width']:g}"
        base_key=f'threshold_{limit}_0'
        results=[v[key]['matched'] for v in (a,b)]
        reference=[v[base_key]['matched']['tp']['change']['delta_fp'] for v in (a,b)]
        change=[v['tp']['change']['delta_fp'] for v in results]
        matched_fp=[v['fp']['change']['delta_tp'] for v in results]
        for outer in results:
            assert all(v['delta_tp']==0 for v in outer['tp']['projects'].values())
            assert all(v['delta_fp']==0 for v in outer['fp']['projects'].values())
        gain=[ref-new for ref,new in zip(reference,change,strict=True)]
        for protocol in ('tp','fp','fixed'):
            row=candidate[protocol]
            calibration,review=row['cal'],row['frozen']
            item=dict(option=key,style=opts['style'],width=opts['width'],strength=opts['strength'],
                      protocol=protocol,publishable=opts['publishable'],matched_a_delta_fp=change[0],
                      matched_b_delta_fp=change[1],matched_a_fp_saved_over_retuned=gain[0],
                      matched_b_fp_saved_over_retuned=gain[1],matched_a_delta_tp=matched_fp[0],
                      matched_b_delta_tp=matched_fp[1],cal_delta_tp=calibration['delta_tp'],
                      cal_delta_fp=calibration['delta_fp'],review_delta_tp=review['delta_tp'],
                      review_delta_fp=review['delta_fp'],cal_f1_gain=f1_gain(calibration),
                      review_f1_gain=f1_gain(review),cut=row['cut'])
            measurement.append(item)
    return measurement


def score_inventory():
    first=read(HERE/'first_hit_public.json')
    assert first['same_ternary']==first['rows']==7279
    assert first['code_sha256']==sha(HERE/'first_hit32.py')
    assert first['program_sha256']==sha(HERE/'scan_first.mjs')
    assert first['predictions_sha256']==sha(HERE/'first_hit_public.npz')
    assert first['source_scanner_sha256']==sha(ROOT/'round22/scan_union.mjs')
    audit=read(ROOT/'round31/summary.json')['target']
    released=ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
    assert audit['archive_sha256']==sha(released)
    counts=audit['counts']['vulnerability']
    fp=sum(r['fp'] for r in first['by_first_rule'].values())
    assert fp==274
    tp=counts['tp']
    quality=2*tp/(2*tp+counts['fp']-fp+counts['fn'])
    current=2*tp/(2*tp+counts['fp']+counts['fn'])
    timing=first['scan_only_seconds']
    return dict(current_score=audit['public_score'],optimistic_if_all_274_fp_removed=audit['public_score']+.3*(quality-current),
                observations_by_first_rule=first['by_first_rule'],
                median_scan_only={k:median(v) for k,v in timing.items()})


def main():
    stage_reports={stage:read_stage(stage) for stage in STAGES}
    measurements=[]
    for stage,run in stage_reports.items():
        measurements.extend(dict(stage=stage,**item) for item in summarize(run))
    selection=[row for row in measurements if row['publishable'] and
               min(row['matched_a_fp_saved_over_retuned'],row['matched_b_fp_saved_over_retuned'])>0 and
               row['cal_f1_gain']>0 and row['review_f1_gain']>=0]
    winner=max(selection,key=lambda r:(r['cal_f1_gain'],r['review_f1_gain']),default=None)
    # A neutral project arm can proceed to a runtime benchmark when the other group improves.
    # The matched-FP diagnostic must also improve recall in both groups.
    benchmark=[r for r in measurements if r['stage']=='calibrated_rules' and r['protocol']=='tp' and
               min(r['matched_a_fp_saved_over_retuned'],r['matched_b_fp_saved_over_retuned'])>=0 and
               r['matched_a_fp_saved_over_retuned']+r['matched_b_fp_saved_over_retuned']>0 and
               min(r['matched_a_delta_tp'],r['matched_b_delta_tp'])>0 and
               r['cal_f1_gain']>0 and r['review_f1_gain']>=0]
    final=max(benchmark,key=lambda r:(r['cal_f1_gain'],r['review_f1_gain'],
              -(float('inf') if r['width'] is None else r['width'])),default=None)
    cmp=HERE/'comparison.csv'
    with cmp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)
    diagnosis={}
    for family in STAGES:
        selected=[row for row in measurements if row['stage']==family and row['protocol']=='tp']
        best=max(selected,key=lambda r:(min(r['matched_a_fp_saved_over_retuned'],r['matched_b_fp_saved_over_retuned']),
                                       r['matched_a_fp_saved_over_retuned']+r['matched_b_fp_saved_over_retuned'],
                                       r['cal_f1_gain'],r['review_f1_gain']))
        diagnosis[family]=best
    result=dict(configurations=len(measurements),eligible=len(selection),selected=winner,benchmark_candidate=final,
                best_matched_by_stage=diagnosis,option_tables={stage:{k:sha(HERE/stage/k/'results.json') for k in PARTS}
                        for stage in STAGES},comparison_sha256=sha(cmp),inventory=score_inventory())
    (HERE/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('configurations','eligible','selected','benchmark_candidate')},indent=2))


if __name__=='__main__':
    main()
