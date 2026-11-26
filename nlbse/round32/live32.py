"""Measure the frozen rule-conditioned candidate and the released tool on the same inputs."""
import csv
from statistics import median
import time


import joblib
import numpy as np

from context32 import HERE, ROOT, R28, f, checksum, json_at, write_json, rule_order
from audit29 import packages, canonical_mat
from first_hit32 import scan, SCRIPT
from length_eval import recipe
from pipeline import PUBLIC, load
from positive_runtime import apply as correct_positive, scope
from rank30 import rerank
from subgroups32 import features
from eval32 import cut_for

OUT=HERE/'public'


def load_choice():
    p=HERE/'selection.json'
    receipt=json_at(p)
    choice=receipt['benchmark_candidate']
    if choice is None:
        raise ValueError('No eligible runtime candidate')
    home=HERE/choice['stage']/'official'
    if checksum(home/'results.json')!=receipt['option_tables'][choice['stage']]['official']:
        raise ValueError('Internal review receipt changed')
    stats=json_at(home/'results.json')
    if stats['code_sha256']!=checksum(HERE/'subgroups32.py'):
        raise ValueError('Rule calibration source changed')
    if stats['rule_order']!=rule_order()[0]:
        raise ValueError('Rule order mismatch')
    model_file=home/'models.joblib'
    if stats['weights_sha256']!=checksum(model_file):
        raise ValueError('Selected rule weights changed')
    ref=json_at(HERE/'first_hit_public.json')
    assert ref['program_sha256']==checksum(SCRIPT) and ref['same_ternary']==ref['rows']
    return choice,joblib.load(model_file)[choice['style']],dict(
        selection_sha256=checksum(p),weights_sha256=checksum(model_file),
        scanner_sha256=checksum(SCRIPT),runtime_sha256=checksum(HERE/'subgroups32.py'))


def run(text, active_candidate, model, config, policy, adjustment):
    core, gate, regression=recipe()
    start=time.perf_counter()
    raw,aux,cost=f.infer_student(model,text,f.pinned()[0],columns=22,return_all=True)
    stage=time.perf_counter()
    values=np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
    est=f.estimate(regression,values)
    p,no,yes,asked=f.choose(values,est,gate,core['gate'])
    known=np.zeros(len(text),bool)
    flag=np.zeros_like(known)
    response,seconds=scan([s for s,k in zip(text,asked,strict=True) if k],active_candidate)
    known[asked]=response[:,0]>=0
    flag[asked]=response[:,0]==1
    p[known]=np.where(flag[known],yes[known],no[known])
    state=dict(p=p.astype(np.float32),w=est['w'],no=no,yes=yes,known=known,flag=flag)
    bits,_,_,work=rerank(text,values,state,core['threshold'],config['variant'],config['weights'],
                         config['section'],config['section_head'],config['negative'],config['recall'])
    old,changed,score,additional=correct_positive(text,values,state,core['threshold'],bits,
                                                  config['correction'],config['positive'])
    eligible,margin,_=scope(text,state,core['threshold'],None)
    margin[changed]=score-config['correction']['cut']
    assert np.array_equal(old[eligible],margin[eligible]>0.)
    use=eligible & (abs(margin)<policy['width']) if policy['width'] is not None else eligible
    result=old.copy()
    extra=0
    if active_candidate and use.any():
        code=[len(s) for s,k in zip(text,use,strict=True) if k]
        counts=np.zeros((len(text),2),dtype=np.int32)
        counts[asked]=response[:,1:]
        x=features(margin[use],counts[use],policy['style'])
        margin[use]+=policy['strength']*(x@adjustment)
        result[use]=margin[use]>policy['cut']
        extra=2*x.size+len(code)
    tag=canonical_mat(text)
    cost['forward_seconds']+=time.perf_counter()-stage
    cost['wall_seconds']=time.perf_counter()-start
    cost['analytic_gflops']+=(work['base_ops']+work['extra_ops']+additional['nnz']+extra)/1e9
    cost['scan_seconds']=seconds
    cost['new_rows']=int(use.sum()) if active_candidate else 0
    return np.column_stack((result,tag)),use,margin,cost


def main():
    policy,adjustment,signature=load_choice()
    signed_dir=R28/'recover_public'
    selected=json_at(signed_dir/'release_selection.json')['selected']
    r29=json_at(signed_dir/'results.json')
    reported=r29['results'][selected]
    original=signed_dir/reported['csv']
    assert checksum(original)==reported['predictions_sha256'] and checksum(PUBLIC)==r29['public_sha256']
    with original.open() as handle:
        expected=np.array([[int(r[k]) for k in ('pred_vulnerability','pred_MAT')]
                           for r in csv.DictReader(handle)],bool)
    texts,truth,_=load(PUBLIC)
    gold=truth[:,0].astype(bool)
    tok=f.small.tokenizer()
    model=f.pipe('official',tok)
    config=packages()
    for active in (False,True):
        run(texts[:4096],active,model,config,policy,adjustment)
    trials={'reference':[],'candidate':[]}
    outcomes={}
    for turn in range(5):
        for active in ((False,True) if turn%2==0 else (True,False)):
            arm='candidate' if active else 'reference'
            pred,affected,margin,cost=run(texts,active,model,config,policy,adjustment)
            if arm in outcomes and not np.array_equal(outcomes[arm][0],pred):
                raise ValueError('Public decisions vary across repeated trials')
            if not active:
                assert np.array_equal(expected,pred)
            else:
                assert np.array_equal(expected[~affected],pred[~affected]) and np.array_equal(expected[:,1],pred[:,1])
            outcomes[arm]=(pred,affected,margin)
            quality=f.metric(truth,pred)
            measured=f.competition_formula(quality['mean']['f1'],cost['forward_seconds'],cost['analytic_gflops'])
            trials[arm].append(cost|{'score':measured})
            print(turn,arm,measured,cost['forward_seconds'],flush=True)
    OUT.mkdir(exist_ok=True)
    results={}
    tie={}
    for arm in trials:
        pred,touched,margin=outcomes[arm]
        quality=f.metric(truth,pred)
        amount={k:median(row[k] for row in trials[arm]) for k in trials[arm][0] if k!='score'}
        result=f.competition_formula(quality['mean']['f1'],amount['forward_seconds'],amount['analytic_gflops'])
        npz=OUT/f'{arm}.npz'
        np.savez_compressed(npz,pred=pred,margin=margin,scope=touched,truth=truth)
        results[arm]=dict(score=result,cost=amount,quality=quality,trials=trials[arm],predictions_sha256=checksum(npz))
        if arm=='candidate':
            for mode in ('tp','fp'):
                mask=touched
                same=expected[:,0].copy()
                t=cut_for(gold[mask],same[mask],margin[mask],mode)
                same[mask]=margin[mask]>t
                tie[mode]=f.comparison(gold,expected[:,0],same)
            tie['frozen']=f.comparison(gold,expected[:,0],pred[:,0])
    differences=[right['score']-left['score'] for left,right in zip(trials['reference'],trials['candidate'],strict=True)]
    write_json(OUT/'results.json',dict(policy=policy,receipt=signature,baseline_reference_sha256=checksum(original),
               public_sha256=checksum(PUBLIC),results=results,paired=differences,
               quality=tie,code_sha256=checksum(HERE/'live32.py')))
    print(tie,differences,flush=True)


if __name__=='__main__':
    main()
