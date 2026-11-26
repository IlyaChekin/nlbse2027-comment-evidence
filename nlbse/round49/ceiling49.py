"""Reconstruct the released decisions under both scanner actions using a signed cache."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'round40'),str(ROOT/'round33')]
from live40 import anchor, pipeline, f, checksum, json_at, write_json
from live35 import forward
from pipeline import PUBLIC, load
from policy33 import reassign
from experiment33 import budget
from ceiling33 import optimize_threshold

DEST = HERE/'public_ceiling'
R33 = ROOT/'round33/public'


def outcomes(y, base, a, b, mask):
    valid = (a != b) & mask
    better = valid & (b == y)
    worse = valid & ~better
    toward = np.where(better,b,base)
    away = np.where(worse,b,base)
    return dict(rows=int(mask.sum()), switches=int(valid.sum()),
                beneficial=f.comparison(y,base,toward),harmful=f.comparison(y,base,away))


def main():
    DEST.mkdir(exist_ok=True)
    archive, predicted_file, gold, expected = anchor()
    profile = ROOT/'round38/release_summary.json'
    stats = json_at(profile)
    texts,y,_ = load(PUBLIC)
    assert np.array_equal(y,gold)
    src = R33/'full_endpoints.npz'
    assert checksum(src)==json_at(R33/'ceiling.json')['endpoints_sha256']
    with np.load(src) as tape:
        exact,old,truth = (tape[k] for k in ('exact','old','y'))
    assert np.array_equal(truth,gold) and int(old.sum()) == stats['cost']['scanner_calls']
    _,state,correction,schedule = pipeline()
    previous = f.choose
    captures, runs = {}, {}
    for action in ('release','skip','all'):
        def choose(*args):
            before,no,yes,mask = previous(*args)
            assert np.array_equal(mask,old)
            chosen = {'release':mask, 'skip':np.zeros_like(mask), 'all':np.ones_like(mask)}[action]
            captures['scope'] = chosen
            return before,no,yes,chosen

        def cached(subset):
            assert len(subset)==captures['scope'].sum()
            return exact[captures['scope']]

        def capture(live,rank,first,aux,context):
            captures['pre'] = live['pre'].copy()
            return correction(live,rank,first,aux,context)

        with patch.object(f,'choose',choose):
            pred, rank, _, cost = forward(texts,*schedule,capture,scanner_fn=cached)
        state.finish(cost)
        assert np.array_equal(pred[:,1],expected[:,1])
        runs[action] = (rank,captures.pop('pre'),cost,pred)
        print(action,len(rank),int((pred[:,0]!=expected[:,0]).sum()),flush=True)
    release, no, yes = (runs[k][0] for k in ('release','skip','all'))
    current = expected[:,0]
    assert np.array_equal(runs['release'][3],expected)
    assert np.array_equal(np.where(old,yes,no)>0., current)
    assert np.allclose(release,np.where(old,yes,no),atol=2e-5,rtol=0)
    assert np.array_equal(runs['release'][1],runs['skip'][1])
    lengths = np.fromiter(map(len,texts),np.int32)
    y = gold[:,0].astype(bool)
    a,b = no>0., yes>0.
    winning = (a!=b)&(b==y)
    risk = (a!=b)&(b!=y)
    utility = winning.astype(np.int8)-risk
    allocated = reassign(old,lengths,utility,runs['release'][1][:,8],1.,1.)
    freeze = f.comparison(y,current,np.where(b==y,b,a))
    full = optimize_threshold(y,no,yes,current)
    same_cost = f.comparison(y,current,np.where(allocated,b,a))
    added = outcomes(y,current,a,b,~old)
    removed = outcomes(y,current,b,a,old)
    changed = release-np.where(old,yes,no)
    F = stats['quality']['MAT']['f1']
    seconds,gflops = (stats['cost'][k] for k in ('forward_seconds','analytic_gflops'))
    thresholds = dict(any_route_fixed_threshold=freeze,same_budget_witness=same_cost,
                      any_route_any_shared_threshold=full['comparison'],only_add_oracle=added['beneficial'])
    for item in thresholds.values():
        q = .5*(item['f1_after']+F)
        item['score_reusing_release_cost']=f.competition_formula(q,seconds,gflops)
        item['score_with_zero_cost']=f.competition_formula(q,0.,0.)
    p = DEST/'endpoints.npz'
    np.savez_compressed(p,truth=gold,old=old,flag=exact,queried=yes,unqueried=no,
                        signed=release,length=lengths,pre=runs['release'][1],same_budget=allocated)
    result = dict(archive_sha256=checksum(archive),release_predictions_sha256=checksum(predicted_file),
                  release_summary_sha256=checksum(profile),source_scanner_sha256=checksum(src),
                  dataset_sha256=checksum(PUBLIC),code_sha256=checksum(Path(__file__)),
                  runtime_sha256=checksum(ROOT/'round35/live35.py'),predictions_sha256=checksum(p),
                  reference_binary_decisions=int(expected.size),rank_max_abs_diff=float(np.max(abs(changed))),
                  fixed=stats['score'],shared_oracle_threshold=full['threshold'],
                  warm_replay_with_cached_scans={k:r[2] for k,r in runs.items()},
                  route_limits=budget({'old':old,'length':lengths},allocated),thresholds=thresholds,
                  only_add=added,only_remove=removed)
    write_json(DEST/'results.json',result)
    print({k:(r['delta_tp'],r['delta_fp'],r['score_reusing_release_cost']) for k,r in thresholds.items()},flush=True)


if __name__=='__main__':
    main()
