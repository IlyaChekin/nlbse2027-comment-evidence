"""Verify the shipped measurements and compute remaining changes to reach 0.97."""
import json

from summarize32 import HERE, ROOT, read, sha


def main():
    archive=ROOT/'submission/NLBSE2027_round32_rule_calibration_tool.zip'
    receipt=read(HERE/'release_verification/receipt.json')
    public=HERE/'public/results.json'
    benchmark=read(public)
    audit=read(HERE/'public/audit.json')
    assert sha(archive)==receipt['archive_sha256']==archive.with_suffix('.sha256').read_text().split()[0]
    assert receipt['identical_binary_decisions']==61_310
    assert receipt['predictions_sha256']==sha(HERE/'release_verification/release_predictions.csv')
    assert sha(public)==audit['benchmark_sha256']
    assert benchmark['receipt']['selection_sha256']==sha(HERE/'selection.json')
    assert benchmark['code_sha256']==sha(HERE/'live32.py')
    assert benchmark['public_sha256']==receipt['public_sha256']
    assert all(sha(HERE/'public'/f'{k}.npz')==v['predictions_sha256'] for k,v in benchmark['results'].items())
    assert audit['scope']==425 and audit['changed']==31
    score=benchmark['results']['candidate']['score']
    prior=benchmark['results']['reference']['score']
    assert score>prior and all(x>0. for x in benchmark['paired'])
    v=benchmark['quality']['frozen']['after']
    m=read(ROOT/'round31/summary.json')['target']['counts']['MAT']
    assert v['tp']==9538 and v['fp']==433
    def f1(tp,fp,fn):
        return 2*tp/(2*tp+fp+fn)
    v1=f1(v['tp'],v['fp'],v['fn'])
    m1=f1(m['tp'],m['fp'],m['fn'])
    def target(fn_in_mat=0, fn_in_vuln=0, fp_in_vuln=0):
        fm=f1(m['tp']+fn_in_mat,m['fp'],m['fn']-fn_in_mat)
        fv=f1(v['tp']+fn_in_vuln,v['fp']-fp_in_vuln,v['fn']-fn_in_vuln)
        return score+.3*(fm+fv-m1-v1)

    quality={
        'mat_tp_added':next(i for i in range(m['fn']+1) if target(fn_in_mat=i)>=.97),
        'vulnerability_tp_added':next(i for i in range(v['fn']+1) if target(fn_in_vuln=i)>=.97),
        'vulnerability_fp_removed':next(i for i in range(v['fp']+1) if target(fp_in_vuln=i)>=.97),
    }
    result=dict(score=score,reference_score=prior,raw_gap=.97-score,
                cost_fixed_minimum_changes=quality,vulnerability=v,MAT=m,
                remaining_fp_by_gate={'rule_positive':247,'rule_negative':96,'not_requested':90},
                remaining_fn_by_gate={'rule_positive':18,'rule_negative':364,'not_requested':132},
                archive_sha256=receipt['archive_sha256'],benchmark_sha256=sha(public),
                verified_binary_decisions=receipt['identical_binary_decisions'])
    (HERE/'release_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
