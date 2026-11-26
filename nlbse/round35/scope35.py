"""Bound the improvement possible by changing decisions in the scoped correction."""
import numpy as np

from teacher35 import HERE, ROOT, f, checksum, json_at, write_json
from sections34 import support


def main():
    r33=ROOT/'round33/public'
    r34=ROOT/'round34'
    proof=json_at(r33/'ceiling.json')
    input_file=r33/'full_endpoints.npz'
    assert checksum(input_file)==proof['endpoints_sha256']
    source=r34/'public_approx/candidate.npz'
    result=json_at(r34/'public_approx/results.json')['comparison']['candidate']
    assert checksum(source)==result['predictions_sha256']
    with np.load(source) as z,np.load(input_file) as p:
        gold=z['truth']
        assert np.array_equal(gold,p['y'])
        y,b=gold[:,0].astype(bool),z['pred']
        cut={'unqueried':p['prior'],'queried':p['queried'],'flag':p['exact'][:,0],'old':p['old']}
        areas={'wide':support(cut),'known_negative':p['old']&(p['exact'][:,0]==0),
               'known_positive':p['old']&(p['exact'][:,0]==1),'unqueried':~p['old']}
        for w in (.40,.75,1.25,1.80):
            areas[f'within_{w:g}']=areas['wide']&(abs(z['rank'])<w)
        data={}
        cost=result['cost']
        for name,mask in areas.items():
            ideal=b.copy()
            ideal[mask,0]=y[mask]
            quality=f.metric(gold,ideal)
            perf=f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            data[name]=dict(rows=int(mask.sum()),score_upper_bound=perf,quality=quality,
                            maximum_change=f.comparison(y,b[:,0],ideal[:,0]))
    write_json(HERE/'scope_ceiling.json',dict(analysis=data,data_sha256=checksum(input_file),
               baseline_predictions_sha256=checksum(source),code_sha256=checksum(HERE/'scope35.py')))
    print({k:(v['rows'],v['maximum_change']['delta_tp'],v['maximum_change']['delta_fp'],v['score_upper_bound'])
           for k,v in data.items()},flush=True)


if __name__=='__main__':
    main()
