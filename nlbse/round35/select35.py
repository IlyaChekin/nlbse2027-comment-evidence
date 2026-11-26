"""Sign the readout comparison before any live run on the public split."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=('test_a','test_b','official')
FILES={'comparison':'eval35.py','sparse':'sparse35.py','projection':'project35.py'}


def sha(p):
    with p.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text())


def stage(name):
    complete={}
    for part in PARTS:
        root=HERE/name/part
        t=root/'results.json'
        table=read(t)
        assert table['rotation']==part
        code=FILES[name]
        choices={sha(HERE/code)}
        snapshot=HERE/'source_snapshots'/code
        if snapshot.is_file():
            choices.add(sha(snapshot))
        assert table['code_sha256'] in choices
        assert sha(root/'readouts.joblib')==table['weights_sha256']
        assert sha(root/'predictions.npz')==table['predictions_sha256']
        r34=ROOT/'round34/selection_approx.json'
        assert table['baseline_sha256']==read(r34)['receipts']['approx_comparison'][part]
        teacher=HERE/'teachers'/part
        if name in ('projection','comparison'):
            record=read(teacher/'receipt.json')
            assert record['code_sha256']==sha(HERE/'teacher35.py')
            assert record['predictions_sha256']==sha(teacher/'scores.npz')
            stamp=sha(teacher/'receipt.json') if name=='projection' else record['predictions_sha256']
            assert stamp==table['teacher_sha256']
            for fold,spec in record['heads'].items():
                for mode,info in spec.items():
                    if (teacher/f'{mode}_{fold}.pt').is_file():
                        assert sha(teacher/f'{mode}_{fold}.pt')==info['sha256']
            if name=='projection':
                assert table['projection_scores_sha256']==sha(root/'scores.npz')
                for arm,state in table['heads'].items():
                    assert sha(root/f'{arm}.pt')==state['sha256']
        complete[part]=(table,sha(t))
    return complete


def enumerate_options(name,d):
    a,b,c=[d[k][0]['options'] for k in PARTS]
    rows=[]
    for key,option in c.items():
        mode=option['mode']
        scope=key.split('_')[0]
        control=scope+'_control_threshold_0'
        gain_tp=[]
        gain_fp=[]
        project_tp=[]
        project_fp=[]
        for f in (a,b):
            stats=f[key]['matched']
            ref=f[control]['matched']
            new=stats['fp']['change']
            drop=stats['tp']['change']
            assert new['delta_fp']==0 and drop['delta_tp']==0
            project_tp.append(new['delta_tp'])
            project_fp.append(-drop['delta_fp'])
            gain_tp.append(new['delta_tp']-ref['fp']['change']['delta_tp'])
            gain_fp.append(ref['tp']['change']['delta_fp']-drop['delta_fp'])
        for sub in ('fixed','fp','tp'):
            x=option[sub]
            internal,review=x['cal'],x['held']
            rows.append(dict(stage=name,key=key,mode=mode,source=option['source'],width=option['width'],
                             strength=option['strength'],rule=sub,cut=x['cut'],cal_rows=option['cal_rows'],
                             held_rows=option['held_rows'],a_tp=project_tp[0],b_tp=project_tp[1],
                             a_fp=project_fp[0],b_fp=project_fp[1],a_tp_over=gain_tp[0],b_tp_over=gain_tp[1],
                             a_fp_over=gain_fp[0],b_fp_over=gain_fp[1],
                             cal_f1=internal['f1_after']-internal['f1_before'],
                             review_f1=review['f1_after']-review['f1_before'],
                             cal_vs_rethreshold=internal['f1_after']-c[control][sub]['cal']['f1_after'],
                             cal_tp=internal['delta_tp'],cal_fp=internal['delta_fp'],
                             review_tp=review['delta_tp'],review_fp=review['delta_fp']))
    return rows


def supported(r):
    t=[r[k+'_tp_over'] for k in ('a','b')]
    f=[r[k+'_fp_over'] for k in ('a','b')]
    across=(min(t)>=0 and sum(t)>0) or (min(f)>=0 and sum(f)>0)
    return r['mode']!='threshold' and across and r['cal_f1']>0 and r['cal_vs_rethreshold']>0 and r['review_f1']>=0


def main():
    records={k:stage(k) for k in FILES}
    rows=[item for k,v in records.items() for item in enumerate_options(k,v)]
    available={k:[e for e in rows if e['stage']==k and supported(e)] for k in records}
    priority=lambda r:(r['cal_f1'],r['review_f1'],-r['cal_rows'])
    branches={k:max(v,key=priority,default=None) for k,v in available.items()}
    portable=[branches[k] for k in ('projection','sparse') if branches[k]]
    chosen=max(portable,key=priority,default=None)
    cmp=HERE/'comparison.csv'
    with cmp.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=rows[0].keys())
        out.writeheader()
        out.writerows(rows)
    state=dict(selected=chosen,by_stage=branches,eligible={k:len(v) for k,v in available.items()},
               result_sha256={k:{p:x[1] for p,x in v.items()} for k,v in records.items()},
               code_sha256=sha(HERE/'select35.py'),csv_sha256=sha(cmp),tested=len(rows),
               released_archive_sha256=read(ROOT/'round34/release_summary.json')['archive_sha256'])
    (HERE/'selection.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:state[k] for k in ('selected','by_stage','eligible','tested')},indent=2))


if __name__=='__main__':
    main()
