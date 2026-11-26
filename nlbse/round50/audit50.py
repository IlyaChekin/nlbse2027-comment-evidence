"""Recount annotation, correction and restoration outcomes from stored predictions."""
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from study50 import HERE, ROOT, DEST as MODELS, PARTS, checksum, json_at, load, f, per_project, diagnostic_order, write_json
from source41 import parts
from bounds50 import DEST as PUBLIC
from restore50 import DEST as RESTORE
from gate50 import DEST as GATE


def change(y,base,pred):
    return f.comparison(y.astype(bool),base.astype(bool),pred.astype(bool))


def model_evaluation():
    selection=HERE/'comparison/selection.json'
    plan=json_at(selection)
    assert plan['code_sha256']==checksum(HERE/'select50.py')
    assert plan['comparison_sha256']==checksum(selection.with_name('comparison.csv'))
    trials={}
    for part in PARTS:
        source,_,stamp=load(part)
        local=MODELS/part/'results.json'
        stats=json_at(local)
        assert plan['sources'][part]==checksum(local) and stamp==stats['proof']['matrix_sha256']
        cols=local.with_name('annotations.npz')
        assert checksum(cols)==stats['annotation_sha256']
        phases={}
        with np.load(cols) as z:
            for tag,frame in zip(('cal','held'),parts(source),strict=True):
                active=frame['old']&(frame['flag']>=0)
                assert np.array_equal(z[tag+'_ids'],frame['ids'][active])
                valid=z[tag+'_known']
                ids=frame['ids'][active][valid]
                baseline=z[tag+'_scanner'][valid]
                y=z[tag+'_w'][valid]
                assert np.array_equal(baseline,frame['flag'][active][valid])
                quality={}
                for arm,r in stats['report'][tag]['heads'].items():
                    p=z[tag+'_'+arm][valid]
                    assert change(y,baseline,p>.5)==r
                    signed,_=diagnostic_order(p,frame['signed'][active][valid],ids)
                    fixed,_=per_project(y,frame['project'][active][valid],baseline,np.ones(len(y),bool),
                                       signed,('fp','tp'))
                    # For this diagnostic only, calibrate a rank per held project against W.
                    query=change(y,baseline,p>.5)
                    pr=average_precision_score(y,p)
                    quality[arm]=dict(w_tp=query['delta_tp'],w_fp=query['delta_fp'],
                                      equal_fp_tp=fixed['fp']['change']['delta_tp'],
                                      equal_tp_removed_fp=-fixed['tp']['change']['delta_fp'],
                                      average_precision=float(pr))
                phases[tag]=quality
        trials[part]=phases
    return dict(table_sha256=plan['comparison_sha256'],settings=plan['tested'],qualified=plan['qualified'],
                annotation_comparison=trials,selection_sha256=checksum(selection))


def restoration():
    src=RESTORE/'receipt.json'
    r=json_at(src)
    assert r['code_sha256']==checksum(HERE/'restore50.py')
    assert r['comparison_sha256']==checksum(RESTORE/'comparison.csv')
    changes={}
    for rotation,table in r['versions'].items():
        p=RESTORE/rotation/'results.json'
        assert table['receipt_sha256']==checksum(p)
        z=p.with_name('predictions.npz')
        assert checksum(z)==table['predictions_sha256']
        assert checksum(p.with_name('counts.joblib'))==table['weights_sha256']
        notes={}
        with np.load(z) as q:
            for phase,x in table['summary'].items():
                y,known,prior=(q[f'{phase}_{a}'] for a in ('gold','known','scanner'))
                for enum,item in x.items():
                    saw=q[f'{phase}_restored_{enum}']
                    symbols=q[f'{phase}_scanned_{enum}']
                    assert np.array_equal(symbols[:,0],saw)
                    assert item['scans_if_replacing']==len(y)
                    assert item['truth_rows']==int(known.sum())
                    assert item['coverage']==change(y[known],prior[known]==1,saw[known]==1)
                    notes[f'{phase}_{enum}']=[item['coverage']['delta_tp'],item['coverage']['delta_fp']]
        changes[rotation]=notes
    return dict(changes=changes,table_sha256=r['comparison_sha256'])


def selective():
    src=GATE/'selection.json'
    r=json_at(src)
    assert r['code_sha256']==checksum(HERE/'gate50.py')
    assert r['table_sha256']==checksum(GATE/'comparison.csv')
    for name,g in r['evidence'].items():
        folder=GATE/name/'predictions.npz'
        assert checksum(folder)==g['predictions_sha256']
        assert checksum(MODELS/name/'results.json')==g['model_sha256']
        data,_,stamp=load(name)
        assert stamp==g['matrix_sha256']
        _,h=parts(data)
        with np.load(folder) as r0:
            assert np.array_equal(r0['ids'],h['ids'])
            for k,v in g['cases'].items():
                b=r0[k]
                d=change(r0['truth'],r0['base'],b)
                assert all(v['held'][u]==s for u,s in d.items())
                assert v['held']['changes']==int((r0['base']!=b).sum())
    return dict(comparisons=r['comparisons'],qualified=r['qualified'],table_sha256=r['table_sha256'])


def bounds():
    src=PUBLIC/'results.json'
    p=json_at(src)
    z=PUBLIC/'attributions.npz'
    assert p['code_sha256']==checksum(HERE/'bounds50.py') and p['predictions_sha256']==checksum(z)
    base=ROOT/'round38/release_summary.json'
    ref=json_at(base)
    assert p['archive_sha256']==ref['archive_sha256']==checksum(ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip')
    with np.load(z) as t:
        y=t['gold'][:,0].astype(bool)
        known=t['observed']
        flagged=t['disagree']&known
        original=t['before']
        fixed=change(y,original,np.where(flagged,y,original))
        baseline=change(y,original,original)['before']
        assert baseline=={key:ref['vulnerability'][key] for key in ('tp','fp','fn','tn')}
        assert p['scenarios']['archived_W_disagrees_observed']['rows']==int(flagged.sum())
        weak=(t['component']&1)>0
        external=t['consistent']&~weak&((t['component']&6)>0)
        true_w=change(y,original,np.where(flagged,weak,original))
        together=change(y,original,np.where(flagged|external,y,original))
        both_count=int((flagged&external).sum())
    def score(tp, fp):
        positive=baseline['tp']+tp
        errors=baseline['fn']-tp+baseline['fp']-fp
        f1=2.*positive/(2*positive+errors)
        return f.competition_formula((f1+ref['quality']['MAT']['f1'])/2.,
                                     ref['cost']['forward_seconds'],ref['cost']['analytic_gflops'])
    dt,df=fixed['delta_tp'],-fixed['delta_fp']
    candidate=score(dt,df)
    assert candidate==p['scenarios']['archived_W_disagrees_observed']['unchanged_cost_score']
    r_dt,r_df=true_w['delta_tp'],-true_w['delta_fp']
    required=dict(tp_if_all_fp=next(i for i in range(dt+1) if score(i,df)>=.97),
                  fp_if_all_tp=next(i for i in range(df+1) if score(dt,i)>=.97))
    return dict(score=ref['score'],observed_formatting_oracle=candidate,
                observed_formatting_extra_tp=dt,observed_formatting_removed_fp=df,
                historical_w_replacement=dict(extra_tp=r_dt,removed_fp=r_df,score=score(r_dt,r_df)),
                replacements_needed_at_release_cost=required,
                combined_oracle=dict(extra_tp=together['delta_tp'],removed_fp=-together['delta_fp'],
                                     overlap=both_count,score=score(together['delta_tp'],-together['delta_fp'])),
                evidence=checksum(src))


def main():
    data=dict(models=model_evaluation(),selective=selective(),restoration=restoration(),bound=bounds(),
              code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',data)
    print({k:v for k,v in data.items() if k!='models'},flush=True)


if __name__=='__main__':
    main()
