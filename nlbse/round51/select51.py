"""Audit component supervision at the release operating point before public evaluation."""
import csv
from pathlib import Path

import numpy as np

from eval51 import HERE, ROOT, DEST as EVAL, FIT, PARTS, ARMS, f, checksum, json_at, load, write_json, geometry, training_code
from source41 import parts

DEST=HERE/'comparison'


def load_study(rotation):
    frame,_,stamp=load(rotation)
    cal,held=parts(frame)
    base=held['signed']>0.
    label=held['y']
    table=EVAL/rotation/'results.json'
    card=json_at(table)
    train=FIT/rotation/'fit.json'
    recipe=json_at(train)
    assert card['fit_sha256']==checksum(train)
    assert card['matrix_sha256']==recipe['config']['sources']['matrix_sha256']==stamp
    assert card['code_sha256']==checksum(HERE/'eval51.py')
    assert recipe['config']['code_sha256']==checksum(training_code(rotation))
    assert recipe['config']['roster_sha256']==checksum(train.with_name('rows.npz'))
    for mode,g in recipe['arms'].items():
        assert g['sha256']==checksum(train.with_name(mode+'.pt'))
    assert len({g['steps'] for g in recipe['arms'].values()})==1
    forecast=table.with_name('predictions.npz')
    assert card['prediction_sha256']==checksum(forecast)
    assert card['scores_sha256']==checksum(table.with_name('scores.npz'))
    with np.load(forecast) as p:
        assert np.array_equal(p['ids'],held['ids']) and np.array_equal(p['cal_ids'],cal['ids'])
        assert np.array_equal(p['truth'],label) and np.array_equal(p['base'],base)
        assert not np.intersect1d(p['cal_ids'],p['ids']).size
        projects={k:held['project']==k for k in np.unique(held['project'])}
        for key,s in card['results'].items():
            scope=geometry(held,s['panel'],s['width'])
            assert s['held_rows']==int(scope.sum())
            for mode in ('fp','tp','fixed'):
                q=p[key+'_'+mode]
                inherited=f.comparison(label,base,q)
                assert all(s[mode]['held'][k]==v for k,v in inherited.items())
                assert np.array_equal(q[~scope],base[~scope])
                if rotation!='official' and mode!='fixed':
                    q=p[key+'_same_'+mode]
                    assert f.comparison(label,base,q)==s['matched'][mode]['change']
                    for p0,m in projects.items():
                        values=f.comparison(label[m],base[m],q[m])
                        assert values['delta_'+mode]==0 and values==s['matched'][mode]['projects'][p0]
    return card,checksum(table)


def f1(report):
    return report['f1_after']-report['f1_before']


def differences(this,selection,controls):
    q=this[selection]['matched']
    o=[]
    for kind,metric,way in [('fp','tp',1),('tp','fp',-1)]:
        gain=way*q[kind]['change']['delta_'+metric]
        at=max([0]+[way*this[key]['matched'][kind]['change']['delta_'+metric] for key in controls])
        o.append((gain,gain-at))
    return o


def clean_subset(candidates):
    source,_,_=load('official')
    cal,held=parts(source)
    _,_,project,path,_=f.inputs()
    train=source['oof']['ids']
    owners={(project[i],path[i]) for i in train if path[i]}
    def unseen(ids):
        return np.fromiter((bool(path[i]) and (project[i],path[i]) not in owners for i in ids),bool)
    cm,hm=unseen(cal['ids']),unseen(held['ids'])
    old=held['signed'][hm]>0.
    gold=held['y'][hm]
    results=dict(cal_rows=len(cm),cal_unseen_files_rows=int(cm.sum()),
                 review_rows=len(hm),review_unseen_files_rows=int(hm.sum()),
                 review_positive=int(gold.sum()),scope='available paths absent from the initial official train')
    with np.load(EVAL/'official/predictions.npz') as z:
        results['predictions']={r['key']+'/'+r['rule']:f.comparison(gold,old,z[r['key']+'_'+r['rule']][hm])
                                for r in candidates}
    return results


def select():
    DEST.mkdir(exist_ok=True)
    reports,tickets={},{}
    for part in PARTS:
        reports[part],tickets[part]=load_study(part)
    A,B,O=(reports[p]['results'] for p in PARTS)
    results=[]
    for key,r in O.items():
        if r['arm']=='reference' or key not in A or key not in B:
            continue
        panel,width,arm,v,weight=key.split('/')
        base=f'{panel}/{width}'
        controls=[f'{base}/reference/threshold/0']
        if arm.endswith('components'):
            benchmark=arm.replace('components','union')
            controls.append(f'{base}/{benchmark}/main/{weight}')
        use_a,use_b=[differences(rows,key,controls) for rows in (A,B)]
        equal_fp = min(use_a[0][1],use_b[0][1])>=0 and use_a[0][1]+use_b[0][1]>0
        equal_tp = min(use_a[1][1],use_b[1][1])>=0 and use_a[1][1]+use_b[1][1]>0
        for mode in ('fixed','fp','tp'):
            d=r[mode]['cal']
            h=r[mode]['held']
            a,b=f1(d),f1(h)
            top=max([0.]+[f1(O[k][mode]['cal']) for k in controls])
            results.append(dict(key=key,arm=arm,view=v,panel=panel,width=r['width'],mix=r['mix'],rule=mode,
                        cut=r[mode]['cut'],a_tp=use_a[0][0],b_tp=use_b[0][0],
                        a_over_tp=use_a[0][1],b_over_tp=use_b[0][1],a_removed_fp=use_a[1][0],b_removed_fp=use_b[1][0],
                        a_over_fp=use_a[1][1],b_over_fp=use_b[1][1],cal_tp=d['delta_tp'],cal_fp=d['delta_fp'],
                        review_tp=h['delta_tp'],review_fp=h['delta_fp'],cal_f1=a,review_f1=b,cal_over=a-top,
                        cal_rows=r['cal_rows'],review_rows=r['held_rows'],
                        qualified=(equal_fp or equal_tp) and a>top and b>0.))
    csv_file=DEST/'comparison.csv'
    with csv_file.open('w',newline='') as target:
        table=csv.DictWriter(target,fieldnames=list(results[0]))
        table.writeheader()
        table.writerows(results)
    ranked=sorted(results,key=lambda r:(r['qualified'],min(r['cal_f1'],r['review_f1']),r['cal_over']),reverse=True)
    candidates={p:next((r for r in ranked if r['qualified'] and r['arm']==p),None) for p in ARMS}
    incumbent=ROOT/'round38/release_summary.json'
    archive=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert checksum(archive)==json_at(incumbent)['archive_sha256']
    summary=dict(selected=candidates,options=len(ranked),qualified=sum(r['qualified'] for r in ranked),
                 examples=ranked[:8],clean_file_diagnostic=clean_subset(ranked[:8]),parts=tickets,csv_sha256=checksum(csv_file),archive_sha256=checksum(archive),
                 base_sha256=checksum(incumbent),code_sha256=checksum(Path(__file__)),
                 arms={k:v['arms'] for k,v in reports.items()})
    write_json(DEST/'selection.json',summary)
    print('options',len(ranked),'qualified',summary['qualified'],candidates,flush=True)
    print('examples',ranked[:4],flush=True)


if __name__=='__main__':
    select()
