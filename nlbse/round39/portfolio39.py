"""Separate an oracle's complementarity from unanimously missed public cases."""
import csv
from pathlib import Path

import numpy as np

from location39 import HERE, ROOT, f, checksum, json_at, write_json
from pipeline import PUBLIC


def from_profile(folder, gold, proof):
    candidates=folder/'candidate.npz'
    receipt=json_at(folder/'results.json')
    assert checksum(candidates)==receipt['comparison']['candidate']['predictions_sha256']
    assert any(receipt['receipt'].get(k)==proof for k in ('data_sha256','public_sha256'))
    with np.load(candidates) as z:
        assert np.array_equal(z['truth'],gold)
        return z['pred'][:,0].astype(bool),checksum(candidates)


def csv_bits(source,rows):
    with source.open(newline='') as r:
        reader=csv.DictReader(r)
        values=list(reader)
    ids=np.asarray([int(v['row']) for v in values])
    flag=np.asarray([int(v['pred_vulnerability']) for v in values])
    assert np.array_equal(ids,np.arange(rows)) and np.isin(flag,(0,1)).all()
    return flag.astype(bool),checksum(source)


def main():
    path=ROOT/'round38/public_primary'
    profile=json_at(path/'results.json')
    data=profile['receipt']['data_sha256']
    assert checksum(PUBLIC)==data
    with np.load(path/'candidate.npz') as z:
        gold=z['truth']
        base=z['pred']
    truth=gold[:,0].astype(bool)
    mat=base[:,1]
    candidates={}
    proofs={}
    for name,fld in [
        ('round39_layout',HERE/'public_chosen'),('round39_local',HERE/'public_best_local'),
        ('round38',path),('round37',ROOT/'round37/public_semantic_deferred'),
        ('round35_broad',ROOT/'round35/public_broad'),('round35_sparse',ROOT/'round35/public_sparse'),
        ('round34',ROOT/'round34/public_approx')]:
        candidates[name],proofs[name]=from_profile(fld,gold,data)
    for name,route in [
        ('early_compact','round10/compact_exit/lexical_deferred/predictions.csv'),
        ('unixcoder','round10/unix_context/unix_lexical/predictions.csv'),
        ('unixcoder_neighbors','round11/neighbors/unixcoder9/predictions.csv'),
        ('sparse_bm25','portfolio/simple/runs/bm25_comments_bigram/public/predictions.csv'),
        ('sparse_learned','portfolio/learned_sparse/runs/distilled_int8/public/predictions.csv')]:
        candidates[name],proofs[name]=csv_bits(ROOT/route,len(truth))
    source=ROOT/'round33/public/full_endpoints.npz'
    assert checksum(source)==json_at(source.with_name('ceiling.json'))['endpoints_sha256']
    with np.load(source) as z:
        assert np.array_equal(z['y'],gold)
        before=z['prior']
        after=z['queried']
        flag=z['exact'][:,0]
        old=z['old']
    candidates['without_scanning']=before>0.
    candidates['full_static_route']=np.where(flag>=0,after,before)>0.
    proofs['counterfactual_routes']=checksum(source)
    t=profile['comparison']['candidate']['cost']
    new_types=list(candidates)[:7]
    portfolios=dict(local_variants=new_types,add_encoders=new_types+['early_compact','unixcoder','unixcoder_neighbors'],
                    all=list(candidates))

    def score(new):
        q=f.metric(gold,np.column_stack((new,mat)))
        return dict(quality=q,reference_cost=f.competition_formula(q['mean']['f1'],t['forward_seconds'],t['analytic_gflops']),
                    zero_cost=f.competition_formula(q['mean']['f1'],0.,0.))

    strata={'known_negative':old&(flag==0),'known_positive':old&(flag==1),'unobserved':~old|(flag<0)}
    result={}
    for name,members in portfolios.items():
        reach=np.logical_or.reduce([candidates[m]==truth for m in members])
        oracle=np.where(reach,truth,~truth)
        blocked={k:{'fn':int((~reach&truth&z).sum()),'fp':int((~reach&~truth&z).sum())}
                 for k,z in strata.items()}
        result[name]=dict(members=members,unanimous_fn=int((~reach&truth).sum()),unanimous_fp=int((~reach&~truth).sum()),
                          blocked_by_scope=blocked,**score(oracle))
    individual={}
    for name,p in candidates.items():
        scope=base[:,0]!=truth
        individual[name]=dict(fixable_fn=int((scope&(p==truth)&truth).sum()),
                              fixable_fp=int((scope&(p==truth)&~truth).sum()),**score(p))
    write_json(HERE/'portfolio.json',dict(public_sha256=data,source_sha256=proofs,portfolios=result,
              candidates=individual,code_sha256=checksum(Path(__file__)),
              purpose='oracle analysis only; selection needs file-disjoint held-out forecasts'))
    print({k:{n:v[n] for n in ('reference_cost','unanimous_fn','unanimous_fp')} for k,v in result.items()},flush=True)


if __name__=='__main__':
    main()
