"""Verify the measured arms and choose only among independent project and internal evidence."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=('test_a','test_b','official')
SOURCES={
    'comparison':('eval34.py','heads.joblib'),
    'compact_comparison':('compare34.py','calibrators.joblib'),
    'attention_comparison':('compare34.py','calibrators.joblib'),
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def verified(stage):
    files={}
    for part in PARTS:
        folder=HERE/stage/part
        result=read(folder/'results.json')
        code,weight=SOURCES[stage]
        assert result['rotation']==part
        source_versions={sha(HERE/code)}
        preserved=HERE/'selection_sources'/code
        if preserved.exists():
            source_versions.add(sha(preserved))
        assert result['code_sha256'] in source_versions
        assert result['weights_sha256']==sha(folder/weight)
        assert result['predictions_sha256']==sha(folder/'predictions.npz')
        view=HERE/'regions'/part
        policy=read(view/'receipt.json')
        assert policy['code_sha256']==sha(HERE/'sections34.py')
        assert policy['predictions_sha256']==sha(view/'scores.npz')
        if stage=='comparison':
            assert result['views_sha256']==policy['predictions_sha256']
        else:
            source='distilled' if stage=='compact_comparison' else 'attention'
            family=HERE/source/part
            receipt=read(family/'receipt.json')
            assert result['heads_sha256']==sha(family/'receipt.json')
            assert receipt['predictions_sha256']==sha(family/'scores.npz')
            for mode,details in receipt['heads'].items():
                assert details['sha256']==sha(family/f'{mode}.pt')
        files[part]=result
    return files


def evaluate(stage,sources):
    a,b,o=[sources[k]['options'] for k in PARTS]
    output=[]
    for k,c in o.items():
        width=k.split('_',1)[0]
        ctrl=width+('_control_threshold_0' if stage=='comparison' else '_threshold_shared_0')
        full=[o[ctrl]['fp']['cal']['f1_after'],o[ctrl]['tp']['cal']['f1_after'],o[ctrl]['fixed']['cal']['f1_after']]
        matched_tp=[t[k]['matched']['fp']['change']['delta_tp'] for t in (a,b)]
        matched_fp=[-t[k]['matched']['tp']['change']['delta_fp'] for t in (a,b)]
        pure_tp=[t[ctrl]['matched']['fp']['change']['delta_tp'] for t in (a,b)]
        pure_fp=[-t[ctrl]['matched']['tp']['change']['delta_fp'] for t in (a,b)]
        for protocol,cut_control in zip(('fp','tp','fixed'),full,strict=True):
            entry=c[protocol]
            u,v=entry['cal'],entry['held']
            output.append(dict(stage=stage,key=k,protocol=protocol,
                               mode=c.get('mode',c.get('model')),source=c['source'],width=c['width'],
                               strength=c['strength'],cut=entry['cut'],
                               cal_f1=u['f1_after']-u['f1_before'],cal_over_recut=u['f1_after']-cut_control,
                               review_f1=v['f1_after']-v['f1_before'],cal_tp=u['delta_tp'],cal_fp=u['delta_fp'],
                               review_tp=v['delta_tp'],review_fp=v['delta_fp'],
                               a_tp=matched_tp[0],b_tp=matched_tp[1],a_fp_saved=matched_fp[0],b_fp_saved=matched_fp[1],
                               a_tp_paired=matched_tp[0]-pure_tp[0],b_tp_paired=matched_tp[1]-pure_tp[1],
                               a_fp_paired=matched_fp[0]-pure_fp[0],b_fp_paired=matched_fp[1]-pure_fp[1],
                               cal_calls=c.get('cal_rows',c.get('cal_calls')),
                               review_calls=c.get('held_rows',c.get('calls'))))
    return output


def supported(item):
    t=[item[x+'_tp_paired'] for x in ('a','b')]
    p=[item[x+'_fp_paired'] for x in ('a','b')]
    project_ok=(min(t)>=0 and sum(t)>0) or (min(p)>=0 and sum(p)>0)
    return project_ok and item['cal_f1']>0 and item['cal_over_recut']>0 and item['review_f1']>=0


def main():
    signatures={}
    table=[]
    for stage in SOURCES:
        dataset=verified(stage)
        signatures[stage]={p:sha(HERE/stage/p/'results.json') for p in PARTS}
        table.extend(evaluate(stage,dataset))
    case=[row for row in table if row['stage']!='comparison' and supported(row)]
    chosen=max(case,key=lambda x:(x['cal_f1'],x['review_f1'],-x['cal_calls']),default=None)
    fn=HERE/'summary.csv'
    with fn.open('w',newline='') as handle:
        o=csv.DictWriter(handle,fieldnames=table[0].keys())
        o.writeheader()
        o.writerows(table)
    cost=read(HERE/'teacher_cost.json')
    assert cost['table_sha256']==signatures['comparison']['official']
    assert cost['code_sha256']==sha(HERE/'teacher_cost34.py')
    outcomes=dict(rows=len(table),eligible=len(case),selected=chosen,table_sha256=sha(fn),receipts=signatures,
                  teacher_cost_sha256=sha(HERE/'teacher_cost.json'))
    (HERE/'selection.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    print(json.dumps({k:outcomes[k] for k in ('rows','eligible','selected')},indent=2))


if __name__=='__main__':
    main()
