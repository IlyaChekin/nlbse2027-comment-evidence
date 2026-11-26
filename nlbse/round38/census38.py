"""Cross-check the remaining public errors by observed branch and first-hit rule."""
from pathlib import Path

import numpy as np

from data38 import HERE, ROOT, f, checksum, json_at, write_json


def main():
    profile = HERE/'public_primary/results.json'
    view = json_at(profile)
    candidate = profile.with_name('candidate.npz')
    result = view['comparison']['candidate']
    assert checksum(candidate)==result['predictions_sha256']
    points = ROOT/'round33/public/full_endpoints.npz'
    assert checksum(points)==json_at(points.with_name('ceiling.json'))['endpoints_sha256']
    rule = ROOT/'round32/first_hit_public.npz'
    source = json_at(rule.with_suffix('.json'))
    assert checksum(rule)==source['predictions_sha256']
    with np.load(candidate) as z,np.load(points) as w,np.load(rule) as r:
        truth, bits, margin = [z[k] for k in ('truth','pred','margin')]
        active,flag = w['old'],w['exact'][:,0]
        assert np.array_equal(truth,w['y'])
        order = np.flatnonzero(active)
        assert np.array_equal(order,r['ids']) and np.array_equal(flag[order],r['results'][:,0])
        first = np.zeros(len(active),np.int32)
        first[order] = r['results'][:,1]
        pre = w['prior']>0.
    y,p = truth[:,0].astype(bool),bits[:,0]
    negative = active&(flag==0)
    positive = active&(flag==1)
    unknown = ~negative&~positive
    quality = result['quality']

    def hypot(mask):
        revised = bits.copy()
        revised[mask,0] = y[mask]
        mean = f.metric(truth,revised)['mean']['f1']
        t = result['cost']
        return f.competition_formula(mean,t['forward_seconds'],t['analytic_gflops'])

    def observe(mask):
        fp = mask & ~y & p
        fn = mask & y & ~p
        return dict(rows=int(mask.sum()),false_positives=int(fp.sum()),false_negatives=int(fn.sum()),
                    fp_over_1_8=int((fp&(margin>=1.8)).sum()),
                    fn_under_minus_1_8=int((fn&(margin<=-1.8)).sum()),score_if_perfect_in_scope=hypot(mask))

    branches = {k:observe(v) for k,v in [('observed_positive',positive),('observed_negative',negative),
              ('negative_prior_negative',negative & ~pre),('negative_prior_positive',negative & pre),
              ('not_observed',unknown),('queried_and_observed',positive|negative)]}
    statistics = {name:observe(positive&(first==i)) for i,name in enumerate(source['order'],1)}
    report = dict(score=result['score'],at_zero_cost=f.competition_formula(quality['mean']['f1'],0.,0.),
                  branches=branches,rules=statistics,public_sha256=view['receipt']['data_sha256'],
                  predictions_sha256=checksum(candidate),endpoints_sha256=checksum(points),
                  first_hit_sha256=checksum(rule),code_sha256=checksum(Path(__file__)))
    write_json(HERE/'census.json',report)
    print(branches,flush=True)


if __name__=='__main__':
    main()
