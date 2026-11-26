"""Audit frozen project comparisons and select a policy before any public run."""
import csv
from pathlib import Path

import numpy as np

from study50 import HERE, ROOT, DEST, PARTS, HEADS, f, checksum, json_at, load, scope, write_json
from source41 import parts

OUT=HERE/'comparison'


def checked(rotation):
    x,_,stamp=load(rotation)
    cal,hold=parts(x)
    source=DEST/rotation/'results.json'
    info=json_at(source)
    assert stamp==info['proof']['matrix_sha256']
    assert info['code_sha256']==checksum(HERE/'study50.py')
    for key,name in [('predictions_sha256','predictions.npz'),('annotation_sha256','annotations.npz'),
                     ('weights_sha256','weights.joblib')]:
        assert info[key]==checksum(source.with_name(name))
    y=hold['y']
    base=hold['signed']>0.
    # Project memberships are used only in diagnostic budget matching.
    groups={s:hold['project']==s for s in np.unique(hold['project'])}
    with np.load(source.with_name('predictions.npz')) as z:
        assert np.array_equal(z['ids'],hold['ids']) and np.array_equal(z['cal_ids'],cal['ids'])
        assert np.array_equal(z['truth'],y) and np.array_equal(z['base'],base)
        assert not np.intersect1d(z['cal_ids'],z['ids']).size
        for key,r in info['results'].items():
            use=scope(hold,r['panel'],r['width'])
            assert int(use.sum())==r['held_rows']
            for h in ('fp','tp','fixed'):
                prediction=z[key+'_'+h]
                a=f.comparison(y,base,prediction)
                assert all(r[h]['held'][k]==v for k,v in a.items())
                assert np.array_equal(prediction[~use],base[~use])
                if rotation!='official' and h!='fixed':
                    view=z[key+'_same_'+h]
                    assert f.comparison(y,base,view)==r['matched'][h]['change']
                    for name,m in groups.items():
                        outcome=f.comparison(y[m],base[m],view[m])
                        assert outcome['delta_'+h]==0 and r['matched'][h]['projects'][name]==outcome
    return info,checksum(source)


def delta(r):
    return r['f1_after']-r['f1_before']


def get_change(results,arm,kind,target):
    mode,key,sign=('fp','tp',1) if kind=='fp' else ('tp','fp',-1)
    gain=sign*results[arm]['matched'][mode]['change']['delta_'+key]
    rival=max([0]+[sign*results[k]['matched'][mode]['change']['delta_'+key] for k in target])
    return gain,gain-rival


def select():
    OUT.mkdir(exist_ok=True)
    reports={}
    tickets={}
    for part in PARTS:
        reports[part],tickets[part]=checked(part)
    a,b,c=(reports[r]['results'] for r in PARTS)
    options=[]
    for path,r in c.items():
        if r['head']=='threshold' or path not in a or path not in b:
            continue
        panel,size,view,model,blend=path.split('/')
        pref='/'.join((panel,size))
        style=r['head']!='direct'
        if style:
            heads=[f'{pref}/{v}/direct/{blend}' for v in ('numeric',view)]
        else:
            heads=[] if view=='numeric' else [f'{pref}/numeric/direct/{blend}']
        controls=[f'{pref}/numeric/threshold/0',*heads]
        g=[[get_change(d,path,k,controls) for d in (a,b)] for k in ('fp','tp')]
        matched=any(min(t[1] for t in group)>=0 and sum(t[1] for t in group)>0 for group in g)
        for mode in ('fixed','fp','tp'):
            dev,tv=r[mode]['cal'],r[mode]['held']
            cal,held=delta(dev),delta(tv)
            frontier=cal-max([0.]+[delta(c[k][mode]['cal']) for k in controls])
            options.append(dict(path=path,kind=model,view=view,panel=panel,width=r['width'],mix=r['blend'],
                rule=mode,cut=r[mode]['cut'],a_tp=g[0][0][0],b_tp=g[0][1][0],
                a_extra_tp=g[0][0][1],b_extra_tp=g[0][1][1],a_removed_fp=g[1][0][0],b_removed_fp=g[1][1][0],
                a_extra_fp=g[1][0][1],b_extra_fp=g[1][1][1],cal_f1=cal,review_f1=held,cal_over=frontier,
                cal_tp=dev['delta_tp'],cal_fp=dev['delta_fp'],review_tp=tv['delta_tp'],review_fp=tv['delta_fp'],
                cal_rows=r['cal_rows'],review_rows=r['held_rows'],
                eligible=matched and cal>0. and held>0. and frontier>0.))
    table=OUT/'comparison.csv'
    with table.open('w',newline='') as io:
        rows=csv.DictWriter(io,fieldnames=list(options[0]))
        rows.writeheader()
        rows.writerows(options)
    order=sorted(options,key=lambda r:(r['eligible'],min(r['cal_f1'],r['review_f1']),r['cal_over']),reverse=True)
    pick=next((r for r in order if r['eligible']),None)
    archive=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert checksum(archive)==json_at(ROOT/'round38/release_summary.json')['archive_sha256']
    top={head:next((v for v in order if v['kind']==head),None) for head in HEADS}
    output=dict(selected=pick,tested=len(options),qualified=sum(r['eligible'] for r in options),
                examples=order[:8],best_by_model=top,recipes={k:v['fit'] for k,v in reports.items()},
                components={k:v['report'] for k,v in reports.items()},sources=tickets,
                archive_sha256=checksum(archive),comparison_sha256=checksum(table),
                code_sha256=checksum(Path(__file__)))
    write_json(OUT/'selection.json',output)
    print('selected',pick,'tested',len(options),'qualified',output['qualified'],flush=True)
    print('examples',order[:5],flush=True)


if __name__=='__main__':
    select()
