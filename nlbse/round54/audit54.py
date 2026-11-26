"""Recount the held-out, final-classifier, and span-preparation evidence."""
from pathlib import Path

import numpy as np

from causal54 import DEST as CAUSAL, HERE, PARTS, ROOT, checksum, f, json_at, load, write_json


def artifact(folder, card, code, assets):
    p=HERE/folder/card
    data=json_at(p)
    assert data['code_sha256']==checksum(HERE/code)
    for label,file in assets.items():
        assert data[label]==checksum(p.with_name(file))
    return data


def main():
    pair=artifact('causal','pairs.json','causal54.py',{'raw_sha256':'pairs.jsonl.gz'})
    causal=artifact('causal','oracle.json','causal54.py',{'oracle_sha256':'oracle.npz'})
    spaces=artifact('spaces','results.json','space54.py',{'predictions_sha256':'predictions.npz',
                                                       'comparison_sha256':'comparison.csv'})
    selection=artifact('comparison','selection.json','select54.py',{'comparison_sha256':'comparison.csv'})
    endpoint=artifact('endpoint','receipt.json','endpoint54.py',{'predictions_sha256':'predictions.npz',
                                                                'comparison_sha256':'comparison.csv'})
    locations=artifact('spans','receipt.json','spans54.py',{'file_sha256':'targets.npz'})
    live=artifact('live','selection.json','live54.py',{})
    assert live['offline_sha256']==checksum(HERE/'comparison/selection.json')
    w_report={}
    for folder,record,report in [('causal',causal,'panels'),('spaces',spaces,'phases')]:
        name='oracle.npz' if folder=='causal' else 'predictions.npz'
        with np.load(HERE/folder/name) as z:
            for part, arms in record[report].items():
                m=np.isin(z['ids'],pair['cohorts'][part])
                before,truth=z['baseline'][m,0],z['gold'][m]
                results={}
                for arm,r in arms.items():
                    q=z[arm][m,0]
                    valid=(q>=0)&(before>=0)
                    fixed=int((valid & (before!=truth)&(q==truth)).sum())
                    harmed=int((valid & (before==truth)&(q!=truth)).sum())
                    assert fixed==r['repaired'] and harmed==r['hurt'] and len(q)==r['rows']
                    results[arm]={'fixed':fixed,'harmed':harmed}
                w_report[folder+'/'+part]=results
    _,labels,project,paths,_=f.inputs()
    with np.load(HERE/'endpoint/predictions.npz') as z:
        for p,r in endpoint['profiles'].items():
            y,base,m=(z[p+'_'+k] for k in ('gold','base','file_held'))
            assert np.array_equal(y,labels[z[p+'_ids'],0])
            for key,v in r.items():
                b=z[p+'_'+key]
                assert np.array_equal(b[:,1],base[:,1])
                assert v['change']==f.comparison(y,base[:,0],b[:,0])
                assert v['new_files']==f.comparison(y[m],base[m,0],b[m,0])
    with np.load(HERE/'spans/targets.npz') as z:
        ids=z['ids']
        for phase in PARTS:
            m=z[phase+'_fit']
            src,_,stamp=load(phase)
            excluded=np.unique(np.r_[src['dev']['ids'],src.get('test',src['dev'])['ids']])
            own={(project[i],paths[i]) for i in excluded if paths[i]}
            assert not np.intersect1d(ids[m],excluded).size
            assert all(paths[i] and (project[i],paths[i]) not in own for i in ids[m])
            hidden=m & (z['exact_source'][:,0]==1) & (z['visible_scanner'][:,0]==0)
            located=z['spans'][:,0,0]>=0
            assert int((hidden&located).sum())==locations['parts'][phase]['hidden_located']
    release=json_at(ROOT/'round38/release_summary.json')
    name=ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    assert release['archive_sha256']==checksum(name)
    done=dict(known_score=release['score'],gap=release['target_gap'],learned_tested=selection['candidates'],
              learned_selected=len(selection['selected']),live_choice=live['selected'],scanner_recounts=w_report,
              end_to_end=endpoint['profiles'],source_spans=locations['parts'],
              code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',done)
    print('release',done['known_score'],'active gap',done['gap'],'tested',done['learned_tested'],
          'passed',done['learned_selected'],flush=True)


if __name__=='__main__':
    main()
