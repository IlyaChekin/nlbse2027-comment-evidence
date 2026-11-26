"""Verify the promoted archive and report the exact remaining target gap."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = ROOT/'submission/NLBSE2027_round37_deferred_sections_tool.zip'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    audit = read(HERE/'audit.json')
    chosen = audit['selected']
    assert chosen=='public_semantic_deferred'
    test = audit['results'][chosen]
    profile = HERE/chosen/'results.json'
    assert test['passed'] and test['wins']==5 and test['public_profile_sha256']==sha(profile)
    result = read(profile)
    paired = result['comparison']['reference']
    winner = result['comparison']['candidate']
    assert test['score']==winner['score'] and test['control']==paired['score']
    assert winner['predictions_sha256']==sha(profile.with_name('candidate.npz'))
    receipt = read(HERE/'release_verification/receipt.json')
    assert receipt['rows']==test['rows']==30655
    assert receipt['identical_binary_decisions']==test['binary_decisions']==61310
    assert receipt['archive_sha256']==sha(ARCHIVE)==ARCHIVE.with_suffix('.sha256').read_text().split()[0]
    assert receipt['predictions_sha256']==sha(HERE/'release_verification/release_predictions.csv')
    assert receipt['public_sha256']==result['receipt']['public_sha256']
    old = read(ROOT/'round34/release_summary.json')
    assert winner['score']>old['score']
    assert old['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert old['archive_sha256']==result['receipt']['archive_sha256']
    old_v = result['quality']['frozen']['after']
    old_mat = old['MAT']

    def f1(tp,fp,fn):
        return 2*tp/(2*tp+fp+fn)

    vuln = f1(old_v['tp'],old_v['fp'],old_v['fn'])
    mat = f1(old_mat['tp'],old_mat['fp'],old_mat['fn'])
    assert abs(winner['quality']['mean']['f1']-(vuln+mat)/2)<1e-14
    def hypothetical(tp_v=0,fp_v=0,tp_m=0):
        return winner['score']+.3*(f1(old_v['tp']+tp_v,old_v['fp']-fp_v,old_v['fn']-tp_v)-vuln+
                     f1(old_mat['tp']+tp_m,old_mat['fp'],old_mat['fn']-tp_m)-mat)

    gaps = dict(tp_v=next(i for i in range(old_v['fn']+1) if hypothetical(tp_v=i)>=.97),
                fp_v=next(i for i in range(old_v['fp']+1) if hypothetical(fp_v=i)>=.97),
                tp_m=next(i for i in range(old_mat['fn']+1) if hypothetical(tp_m=i)>=.97))
    report = dict(score=winner['score'],paired_reference_score=paired['score'],target_gap=.97-winner['score'],
                  quality=winner['quality'],cost=winner['cost'],vulnerability=old_v,MAT=old_mat,
                  changes_to_target_at_fixed_cost=gaps,archive_sha256=receipt['archive_sha256'],
                  checkpoint_sha256=receipt['checkpoint_sha256'],public_sha256=receipt['public_sha256'],
                  scored_decisions=receipt['identical_binary_decisions'],measurement_sha256=sha(profile),
                  audit_sha256=sha(HERE/'audit.json'))
    (HERE/'release_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
