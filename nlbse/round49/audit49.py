"""Verify the public diagnostic ceiling and archived-annotation comparisons."""
from pathlib import Path

import numpy as np

from route49 import HERE, ROOT, PARTS, f, checksum, json_at, load, write_json
from source41 import parts
from ceiling33 import optimize_threshold


def verify_public():
    proof=HERE/'public_ceiling/results.json'
    table=proof.with_name('endpoints.npz')
    info=json_at(proof)
    stamped=ROOT/'round38/release_summary.json'
    release=json_at(stamped)
    assert info['release_summary_sha256']==checksum(stamped)
    assert info['code_sha256']==checksum(HERE/'ceiling49.py')
    assert info['predictions_sha256']==checksum(table)
    assert info['archive_sha256']==release['archive_sha256']
    assert info['source_scanner_sha256']==checksum(ROOT/'round33/public/full_endpoints.npz')
    with np.load(table) as frame:
        y=frame['truth'][:,0].astype(bool)
        no,yes,allocated,old,size=(frame[k] for k in ('unqueried','queried','same_budget','old','length'))
        base=frame['signed']>0.
        assert np.array_equal(np.where(old,yes,no)>0.,base)
        assert int(allocated.sum())==int(old.sum())==info['route_limits']['calls']
        assert int(size[allocated].sum())==info['route_limits']['characters']<=int(size[old].sum())
    typical=optimize_threshold(y,no,yes,base)
    a,b=no>0.,yes>0.
    fixed=f.comparison(y,base,np.where(b==y,b,a))
    fixed_budget=f.comparison(y,base,np.where(allocated,b,a))
    stats=info['thresholds']
    fee=release['cost']
    for k,result in [('any_route_fixed_threshold',fixed),('any_route_any_shared_threshold',typical['comparison']),
                     ('same_budget_witness',fixed_budget)]:
        for outcome,value in result.items():
            assert value==stats[k][outcome]
        F=.5*(release['quality']['MAT']['f1']+result['f1_after'])
        assert f.competition_formula(F,fee['forward_seconds'],fee['analytic_gflops'])==stats[k]['score_reusing_release_cost']
    bound=stats['any_route_any_shared_threshold']['score_reusing_release_cost']
    speed=fee['forward_seconds']-(.97-bound)/.0002
    return dict(win_fixed=fixed_budget,ceiling=bound,seconds_needed_even_at_ceiling=speed,
                archive_sha256=release['archive_sha256'],results_sha256=checksum(proof))


def verify_audit():
    p=HERE/'format_audit/results.json'
    report=json_at(p)
    scan=p.with_name('predictions.npz')
    assert report['code_sha256']==checksum(HERE/'format_audit49.py') and report['predictions_sha256']==checksum(scan)
    source=HERE/'components/receipt.json'
    root=json_at(source)
    assert root['code_sha256']==checksum(HERE/'components49.py') and report['inputs']==checksum(source)
    assert checksum(source.with_name('labels.npz'))==root['predictions_sha256']
    with np.load(scan) as z,np.load(source.with_name('labels.npz')) as c:
        rows,y,project=z['ids'],z['annotation'],z['group']
        assert len(rows)==len(np.unique(rows))
        assert c['selected'][rows].all() and np.array_equal(y,c['component'][rows]&1)
        assert len(report['record_hashes'])==len(rows)
        wrong=int((z['contest'][:,0]!=y).sum())
        residual=int((z['original_masked'][:,0]!=y).sum())
        for name,r in report['grouped'].items():
            m=project==name
            assert int(m.sum())==r['unique']
            for k,record in r['scores'].items():
                assert int(((z[k][:,0]==y)&m).sum())==record['agreement']
                assert int(((z[k][:,0]!=z['contest'][:,0])&m).sum())==record['changed_from_contest']
        assert np.array_equal(z['original'][:,0],y)
    return dict(correspondence_rows=root['usable'],audit_rows=len(rows),formatted_mismatches=wrong,
                masked_original_mismatches=residual,results_sha256=checksum(p))


def verify_styles():
    p=HERE/'styles/receipt.json'
    data=json_at(p)
    table=p.with_name('predictions.npz')
    assert data['code_sha256']==checksum(HERE/'styles49.py')
    assert data['predictions_sha256']==checksum(table)
    assert data['comparison_sha256']==checksum(p.with_name('comparison.csv'))
    assert data['formatter_sha256']==checksum(HERE/'formatter/receipt.json')
    assert data['annotation_sha256']==checksum(HERE/'components/labels.npz')
    kinks={}
    with np.load(table) as z:
        ids,base,truth,eligible=(z[k] for k in ('ids','baseline','source_w','matched'))
        assert len(ids)==data['total_calls_per_style']
        for rotation in PARTS:
            source,_,stamp=load(rotation)
            assert stamp==data['sources'][rotation]
            for name,r in zip(('cal','held'),parts(source),strict=True):
                focus=r['old']&(r['flag']>=0)&(abs(r['unqueried'])<data['threshold'])
                rows=r['ids'][focus]
                i=np.searchsorted(ids,rows)
                assert np.array_equal(ids[i],rows) and np.array_equal(base[i,0],r['flag'][focus])
                i=i[eligible[i]]
                for style in data['measurements']:
                    new=z[style][i,0]==1
                    expected=data['records'][f'{rotation}/{name}/{style}']
                    metrics=f.comparison(truth[i],base[i,0]==1,new)
                    assert all(expected[k]==v for k,v in metrics.items())
                    if name=='held':
                        kinks.setdefault(style,[]).append((metrics['delta_tp'],metrics['delta_fp']))
    return dict(matched_test_changes=kinks,trace_sha256=checksum(p))


def main():
    routes=HERE/'comparison/selection.json'
    decomposition=HERE/'component_comparison/selection.json'
    for policy in (routes,decomposition):
        result=json_at(policy)
        tab=policy.with_name('comparison.csv')
        assert checksum(tab)==result.get('table_sha256',result.get('comparison_sha256'))
    report=dict(public=verify_public(),format_audit=verify_audit(),styles=verify_styles(),
                routing_selection=checksum(routes),component_selection=checksum(decomposition),
                code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
