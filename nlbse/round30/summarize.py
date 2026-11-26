"""Recount the saved comparisons and check the original or replacement release."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARTS=('test_a','test_b','official')


def sha(path):
    with path.open('rb') as fd:
        return hashlib.file_digest(fd,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def comparison():
    records=[]
    modes={}
    for directory in (HERE/'comparisons/all',HERE/'comparisons/delta',HERE/'direct'):
        if not (directory/'official/results.json').is_file():
            continue
        a,b,c=[read(directory/p/'results.json') for p in PARTS]
        group=directory.name
        for part,v in zip(PARTS,(a,b,c),strict=True):
            folder=directory/part
            assert sha(folder/'predictions.npz')==v['predictions_sha256']
            key='readout_sha256' if group!='direct' else 'head_sha256'
            weights=folder/('readouts.joblib' if group!='direct' else 'readout.joblib')
            assert sha(weights)==v[key]
        for k,v in c['options'].items():
            pair=[x['options'][k] for x in (a,b)]
            for old in pair:
                detail=old.get('per',old.get('projects'))
                assert all(rec['delta_fp']==0 for rec in detail.values())
                assert sum(rec['delta_tp'] for rec in detail.values())==old['matched']['delta_tp']
            project=[m['matched']['delta_tp'] for m in pair]
            for arm,n in v['cuts'].items():
                cal,review=n['cal'],n['frozen']
                row=dict(study=group,mode=v.get('mode','direct'),key=k,cut=arm,a_tp=project[0],b_tp=project[1],
                         delta_cal_tp=cal['delta_tp'],delta_cal_fp=cal['delta_fp'],review_tp=review['delta_tp'],
                         review_fp=review['delta_fp'],proxy=n['proxy'])
                records.append(row)
                if min(project)>0:
                    label=(group,row['mode'])
                    direction=(row['proxy'],row['review_tp']-row['review_fp'])
                    if label not in modes or direction>modes[label][0]:
                        modes[label]=(direction,row)
    with (HERE/'comparison.csv').open('w',newline='') as fd:
        w=csv.DictWriter(fd,fieldnames=records[0])
        w.writeheader()
        w.writerows(records)
    return {f'{p[0]}/{p[1]}':v[1] for p,v in modes.items()}


def teacher():
    findings={}
    for name in ('teachers','tuned_teacher'):
        parts={}
        for side in PARTS:
            d=HERE/name/side
            v=read(d/'fit.json')
            assert sha(d/'oof.npz')==v['predictions_sha256']
            m=v['scope']
            parts[side]=dict(rows=v['rows'],scope=m)
        findings[name]=parts
    return findings


def final():
    stable=ROOT/'round28/recover_public/release_verification'
    archived=ROOT/'submission/NLBSE2027_round29_recall_tool.zip'
    checked=read(stable/'receipt.json')
    assert sha(archived)==checked['archive_sha256'] and checked['identical_binary_decisions']==61310
    result=dict(archived=archived.name,score=read(ROOT/'round28/summary.json')['score'])
    source=HERE/'public'
    policy=read(source/'policy.json')
    result['selected_for_public']=policy['selected']
    candidate=policy['selected']
    if candidate=='reference':
        return result
    results=read(source/'results.json')
    assert sha(source/'policy.json')==results['policy_sha256']
    for name,row in results['results'].items():
        file=source/('reference.csv' if name=='reference' else 'candidate.csv')
        assert sha(file)==row['predictions_sha256']
        assert all(median(v[key] for v in row['trials'])==value for key,value in row['timing'].items())
    chosen=read(source/'release_selection.json')
    if chosen['selected']==candidate:
        z=source/'release_verification/receipt.json'
        x=ROOT/'submission/NLBSE2027_round30_distill_tool.zip'
        actual=read(z)
        assert sha(x)==actual['archive_sha256'] and actual['identical_binary_decisions']==61310
        result.update(archived=x.name,score=results['results'][candidate]['score'])
    result.update(benchmark=results['results'], paired=results['pairs'][candidate])
    return result


if __name__=='__main__':
    summary=dict(teachers=teacher(),both_heldout=comparison(),release=final())
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(release=summary['release'],both_heldout=summary['both_heldout']),indent=2))
