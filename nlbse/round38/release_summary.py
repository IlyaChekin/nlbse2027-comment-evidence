"""Check the signed published decisions and calculate the remaining target gap."""
from pathlib import Path
from select38 import read, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'


def main():
    audit = read(HERE/'audit.json')
    profile = HERE/'public_primary/results.json'
    assert audit['promoted'] and audit['wins']==5 and audit['profile_sha256']==sha(profile)
    result = read(profile)
    winner = result['comparison']['candidate']
    assert winner['predictions_sha256']==sha(profile.with_name('candidate.npz'))
    receipt = read(HERE/'release_verification/receipt.json')
    assert receipt['rows']==audit['rows']==30655
    assert receipt['identical_binary_decisions']==audit['binary_decisions']==61310
    assert receipt['archive_sha256']==sha(ARCHIVE)==ARCHIVE.with_suffix('.sha256').read_text().split()[0]
    assert receipt['predictions_sha256']==sha(HERE/'release_verification/release_predictions.csv')
    assert receipt['public_sha256']==result['receipt']['data_sha256']
    past = read(ROOT/'round37/release_summary.json')
    assert winner['score']==audit['score']>past['score']
    assert past['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert past['archive_sha256']==result['receipt']['archive_sha256']
    v = result['changes']['frozen']['after']
    m = past['MAT']
    def f1(tp,fp,fn):
        return 2*tp/(2*tp+fp+fn)
    o,u = f1(v['tp'],v['fp'],v['fn']),f1(m['tp'],m['fp'],m['fn'])
    assert abs(winner['quality']['mean']['f1']-(o+u)/2)<1e-13
    def score(t=0,p=0,d=0):
        return winner['score']+.3*(f1(v['tp']+t,v['fp']-p,v['fn']-t)-o+f1(m['tp']+d,m['fp'],m['fn']-d)-u)
    demand = dict(tp_v=next(i for i in range(v['fn']+1) if score(t=i)>=.97),
                  fp_v=next(i for i in range(v['fp']+1) if score(p=i)>=.97),
                  tp_m=next(i for i in range(m['fn']+1) if score(d=i)>=.97))
    record = dict(score=winner['score'],paired_reference_score=audit['control'],target_gap=.97-winner['score'],
                  quality=winner['quality'],cost=winner['cost'],vulnerability=v,MAT=m,
                  changes_to_target_at_fixed_cost=demand,archive_sha256=receipt['archive_sha256'],
                  checkpoint_sha256=receipt['checkpoint_sha256'],public_sha256=receipt['public_sha256'],
                  scored_decisions=receipt['identical_binary_decisions'],measurement_sha256=sha(profile),
                  audit_sha256=sha(HERE/'audit.json'))
    import json
    (HERE/'release_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
