"""Verify the selected archive against the archived public predictions."""
import json

from select34 import HERE, ROOT, read, sha


def main():
    benchmark=HERE/'public_approx/results.json'
    profile=read(benchmark)
    c=profile['comparison']['candidate']
    b=profile['comparison']['reference']
    audit=read(HERE/'audit.json')
    assert audit['release_candidate']=='public_approx'
    assert audit['outcomes']['public_approx']['check']==sha(benchmark)
    assert all(x>0. for x in profile['paired']) and len(profile['paired'])==5
    assert sha(HERE/'selection_approx.json')==profile['receipt']['selection_sha256']
    assert sha(HERE/'public_approx/candidate.npz')==c['predictions_sha256']
    past=read(ROOT/'round32/release_summary.json')
    assert c['score']>past['score']
    archive=ROOT/'submission/NLBSE2027_round34_pooled_projection_tool.zip'
    receipt=read(HERE/'release_verification/receipt.json')
    assert receipt['rows']==30655 and receipt['identical_binary_decisions']==61310
    assert receipt['public_sha256']==profile['receipt']['public_sha256']
    assert receipt['archive_sha256']==sha(archive)==archive.with_suffix('.sha256').read_text().split()[0]
    assert sha(HERE/'release_verification/release_predictions.csv')==receipt['predictions_sha256']
    m=past['MAT']
    v=profile['quality']['frozen']['after']
    def f1(tp,fp,fn):
        return 2*tp/(2*tp+fp+fn)
    p=f1(v['tp'],v['fp'],v['fn'])
    q=f1(m['tp'],m['fp'],m['fn'])
    assert abs(c['quality']['mean']['f1']-.5*(p+q))<1e-13

    def hypothetical(tp_v=0,fp_v=0,tp_m=0):
        return c['score']+.3*(f1(v['tp']+tp_v,v['fp']-fp_v,v['fn']-tp_v)-p+
                             f1(m['tp']+tp_m,m['fp'],m['fn']-tp_m)-q)

    counts=dict(tp_v=next(i for i in range(v['fn']+1) if hypothetical(tp_v=i)>=.97),
                fp_v=next(i for i in range(v['fp']+1) if hypothetical(fp_v=i)>=.97),
                tp_m=next(i for i in range(m['fn']+1) if hypothetical(tp_m=i)>=.97))
    data=dict(score=c['score'],paired_reference_score=b['score'],target_gap=.97-c['score'],quality=c['quality'],
              cost=c['cost'],vulnerability=v,MAT=m,changes_to_target_at_fixed_cost=counts,
              archive_sha256=receipt['archive_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],
              public_sha256=receipt['public_sha256'],scored_decisions=receipt['identical_binary_decisions'],
              measurement_sha256=sha(benchmark),audit_sha256=sha(HERE/'audit.json'))
    (HERE/'release_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
