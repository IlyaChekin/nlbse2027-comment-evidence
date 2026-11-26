"""Locate the remaining score headroom using only signed public decisions."""
from pathlib import Path

import numpy as np

from section37 import HERE, ROOT, f, checksum, json_at, write_json


def main():
    last = ROOT/'round34/public_approx'
    proof = json_at(last/'results.json')['comparison']['candidate']
    signed = last/'candidate.npz'
    assert proof['predictions_sha256']==checksum(signed)
    path = ROOT/'round33/public/full_endpoints.npz'
    assert checksum(path)==json_at(path.with_name('ceiling.json'))['endpoints_sha256']
    with np.load(signed) as current,np.load(path) as seen:
        pred,truth,margin = [current[k] for k in ('pred','truth','rank')]
        assert np.array_equal(truth,seen['y'])
        ran,flag = seen['old'],seen['exact'][:,0]
        initial = seen['prior']>0.
    gold,report = truth[:,0].astype(bool),pred[:,0]
    assert np.array_equal(report,margin>0.)
    negative = ran&(flag==0)
    positive = ran&(flag==1)
    unknown = ~negative & ~positive
    assert int(negative.sum())==4476
    assert int(positive.sum())==2803

    def score(bit,zero_cost=False):
        proposed = pred.copy()
        proposed[:,0] = bit
        m = f.metric(truth,proposed)
        expense = proof['cost']
        return f.competition_formula(m['mean']['f1'],
                   0. if zero_cost else expense['forward_seconds'],
                   0. if zero_cost else expense['analytic_gflops'])

    def bound(mask):
        fn = mask & gold & ~report
        fp = mask & ~gold & report
        props = dict(rows=int(mask.sum()),fn=int(fn.sum()),fp=int(fp.sum()),
                     recall_only=score(report | fn),precision_only=score(report & ~fp),
                     both=score(np.where(mask,gold,report)),
                     score_at_zero_cost=score(np.where(mask,gold,report),True))
        return props

    strata = {f'prior_{a}_final_{b}':bound(negative&(initial==a)&(report==b))
              for a in (False,True) for b in (False,True)}
    widths = {str(x):bound(negative&(abs(margin)<x)) for x in (.40,.75,1.25,1.8,3.,5.)}
    budget = dict(known_negative=bound(negative),known_positive=bound(positive),unobserved=bound(unknown),
                  both_observed_branches=bound(negative|positive),all_rows=bound(np.ones(len(gold),bool)))
    target = .97
    positive_errors = negative&gold&~report
    n = int(positive_errors.sum())
    fp = report&~gold
    initial_tp = int((report&gold).sum())
    actual_positive = int(gold.sum())
    initial_fp = int(fp.sum())
    expense = proof['cost']
    mat = proof['quality']['MAT']['f1']
    corrections = []
    for true_positives in (0,50,100,150,200,250,300,n):
        tp = initial_tp+true_positives
        needed = None
        for removed in range(initial_fp+1):
            f1 = 2*tp/(actual_positive+tp+initial_fp-removed)
            total = f.competition_formula((f1+mat)/2,expense['forward_seconds'],expense['analytic_gflops'])
            if total>=target:
                needed = removed
                break
        corrections.append(dict(recovered_known_negative=true_positives,any_fp_to_remove=needed,
                                attainable_only_known_negative=needed is not None and needed<=int((negative&fp).sum())))
    result = dict(released_score=proof['score'], released_score_at_zero_cost=score(report,True),target=target,
                  fixed_cost=proof['cost'], branches=budget,negative_breakdown=strata,all_negative_margin_caps=widths,
                  combinations_at_target=corrections, recipe='oracle using public labels, fixed MAT and decisions outside scope',
                  old_archive_sha256=checksum(signed),endpoint_sha256=checksum(path),
                  code_sha256=checksum(Path(__file__)))
    write_json(HERE/'scope.json',result)
    print(result['released_score_at_zero_cost'],widths,strata,corrections,flush=True)


if __name__=='__main__':
    main()
