"""Measure a diagnostic bound for every possible scanner policy using the frozen endpoints."""
import numpy as np

from points33 import HERE, R32, f, checksum, json_at, write_json
from audit29 import packages
from disagreement import clip_logit
from experiment33 import oracle
from first_hit32 import scan
from length_eval import recipe
from live32 import load_choice
from pipeline import PUBLIC, load
from positive_runtime import apply as correct_positive
from rank30 import rerank
from subgroups32 import features as group_features


def exact_scores(texts, values, observed, p, no, yes, w, core, package, cfg, coef):
    flag=observed[:,0]==1
    known=observed[:,0]>=0
    prob=np.where(known,np.where(flag,yes,no),p).astype(np.float32)
    state=dict(p=prob,w=w,no=no,yes=yes,known=known,flag=flag)
    risk=np.full(len(p),np.nan)
    q,_,_,_=rerank(texts,values,state,core['threshold'],package['variant'],package['weights'],
                   package['section'],package['section_head'],package['negative'],package['recall'],risk)
    bits,extra,extra_q,_=correct_positive(texts,values,state,core['threshold'],q,package['correction'],package['positive'])
    t=clip_logit(np.asarray(np.float32(core['threshold']),dtype=np.float64))
    margins=clip_logit(prob.astype(np.float64))-t
    negative=known & ~flag
    margins[negative]=risk[negative]
    margins[extra]=extra_q-package['correction']['cut']
    assert np.array_equal(margins>0.,bits)
    positive=flag & (abs(margins)<cfg['width'])
    x=group_features(margins[positive],observed[positive,1:3].astype(np.int32),cfg['style'])
    margins[positive]+=cfg['strength']*(x@coef)-cfg['cut']
    return margins,t


def optimize_threshold(y,a,b,reference):
    # For each possible threshold, choose the more favorable endpoint using the hidden labels.
    # The resulting ranking bounds every query subset at that threshold, including an unlimited budget.
    effective=np.where(y,np.maximum(a,b),np.minimum(a,b))
    order=np.argsort(-effective,kind='stable')
    z=effective[order]
    at=np.flatnonzero(np.r_[z[:-1]!=z[1:],True])
    tp=np.cumsum(y[order])[at]
    fp=(at+1)-tp
    f1=2*tp/(y.sum()+tp+fp)
    k=at[int(f1.argmax())]
    cut=float(np.nextafter(z[k],-np.inf))
    return dict(threshold=cut,comparison=f.comparison(y,reference,effective>cut))


def main():
    file=R32/'public/candidate.npz'
    proof=json_at(R32/'public/results.json')
    stats=proof['results']['candidate']
    assert checksum(file)==stats['predictions_sha256'] and checksum(PUBLIC)==proof['public_sha256']
    with np.load(file) as source:
        current=source['pred'][:,0]
        oof=source['truth']
    texts,gold,_=load(PUBLIC)
    assert np.array_equal(gold,oof)
    exact,seconds=scan(texts,True)
    tok=f.small.tokenizer()
    p,aux,_=f.infer_student(f.pipe('official',tok),texts,f.pinned()[0],columns=22,return_all=True)
    values=np.column_stack((p[:,0],aux[:,0],aux[:,-2:]))
    core,route,trained=recipe()
    est=f.estimate(trained,values)
    prior,no,yes,old=f.choose(values,est,route,core['gate'])
    assert int(old.sum())==7279
    scout=R32/'first_hit_public.npz'
    assert checksum(scout)==json_at(R32/'first_hit_public.json')['predictions_sha256']
    with np.load(scout) as release:
        assert np.array_equal(release['ids'],np.flatnonzero(old))
        assert np.array_equal(release['results'],exact[old])
    cfg,coef,_=load_choice()
    ranks,cut=exact_scores(texts,values,exact,prior,no,yes,est['w'],core,packages(),cfg,coef)
    no_scan=clip_logit(prior.astype(np.float32).astype(np.float64))-cut
    archived=np.where(old,ranks,no_scan)
    assert np.array_equal(archived>0.,current)
    size=np.fromiter(map(len,texts),dtype=np.int32)
    raw=f.query_value(route,prior,no,yes,est['w'],core['gate']['base'])
    data=dict(pre=np.column_stack((values,est['w'],no,yes,prior,raw)),old=old,queried=ranks,unqueried=no_scan,
              replay=archived,y=gold[:,0].astype(bool),length=size)
    report=oracle(data)
    best=optimize_threshold(data['y'],no_scan,ranks,current)
    bounds={}
    for k,d in (('same_budget_witness',report['same_budget']),('any_budget_published_threshold',report['any_budget']),
                ('any_budget_any_shared_threshold',best['comparison'])):
        q=.5*(d['f1_after']+stats['quality']['MAT']['f1'])
        bounds[k]=dict(with_previous_cost=f.competition_formula(q,stats['cost']['forward_seconds'],stats['cost']['analytic_gflops']),
                       with_zero_cost=f.competition_formula(q,0.,0.),delta_tp=d['delta_tp'],delta_fp=d['delta_fp'])
    out=HERE/'public/full_endpoints.npz'
    np.savez_compressed(out,exact=exact,old=old,y=gold,prior=no_scan,queried=ranks)
    report=dict(signed_reference=True,full_scans=len(texts),scan_seconds=seconds,diagnostic_only=True,
                scores=bounds,label_selected_threshold=best,protocol=report,source_predictions_sha256=checksum(file),
                endpoints_sha256=checksum(out),code_sha256=checksum(HERE/'ceiling33.py'))
    write_json(HERE/'public/ceiling.json',report)
    print(bounds,flush=True)


if __name__=='__main__':
    main()
