"""Recalculate per-project budgets and measured scores from saved predictions."""
import csv
from pathlib import Path
from statistics import median
import sys

import numpy as np

from study44 import HERE, ROOT, PARTS, f, checksum, json_at, load, parts, write_json, cut
from structure44 import DEST as FEATURE
from pipeline import PUBLIC
sys.path.insert(0,str(ROOT/'round43'))
from audit43 import verify


def internal(folder,fit_code,pick_code,frames,certificate):
    selection = folder/'selection.json'
    plans = json_at(selection)
    assert plans['csv_sha256']==checksum(folder/'comparison.csv')
    assert plans['code_sha256']==checksum(HERE/pick_code)
    with (folder/'comparison.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows)==plans.get('tested',plans.get('checked'))
    assert (plans['chosen'] is not None)==any(x['eligible']=='True' for x in rows)
    results = {}
    for key,(held,matrix_proof) in frames.items():
        where = folder/key
        file = where/'results.json'
        stage = json_at(file)
        assert checksum(file)==plans['sources'][key]
        assert stage['code_sha256']==checksum(HERE/fit_code)
        source = stage['training']
        assert source.get('matrix_sha256',source.get('source_sha256'))==matrix_proof
        assert source['features_sha256']==certificate
        assert source['pairs_sha256']==checksum(ROOT/'round37/matched'/f'{key}.npz')
        results[key] = verify(where,stage,held,key)
    return dict(roles=results,chosen=plans['chosen'],selection_sha256=checksum(selection))


def profile(archive,baseline):
    folder = HERE/'public'
    p = folder/'results.json'
    if not p.exists():
        return None
    result = json_at(p)
    record = result['receipt']
    prior = ROOT/'round38/public_primary/candidate.npz'
    measured = {'archive':archive,'public':PUBLIC,'anchor':prior,
                'selection':HERE/'comparison/selection.json','model':HERE/'comparison/official/weights.joblib',
                'code':HERE/'live44.py','extractor':HERE/'structure44.py'}
    for tag,path in measured.items():
        assert checksum(path)==record[tag+'_sha256']
    old = np.load(prior)
    r,y = old['pred'],old['truth']
    assert r.shape==y.shape==(30655,2)
    runs = {}
    estimates = {}
    for mode,stats in result['comparison'].items():
        file = folder/(mode+'.npz')
        assert checksum(file)==stats['predictions_sha256']
        values = np.load(file)
        assert np.array_equal(values['truth'],y)
        pred,scope,margin = values['pred'],values['scope'],values['margin']
        runs[mode] = pred,scope,margin
        assert stats['quality']==f.metric(y,pred)
        runs_count = len(stats['trials'])
        assert runs_count==6
        for k in stats['cost']:
            assert median([x[k] for x in stats['trials']])==stats['cost'][k]
        q = stats['quality']['mean']['f1']
        scores = [f.competition_formula(q,x['forward_seconds'],x['analytic_gflops']) for x in stats['trials']]
        assert scores==[s['score'] for s in stats['trials']]
        assert abs(median(scores)-stats['score'])<1e-14
        estimates[mode] = scores
    assert np.array_equal(runs['reference'][0],r)
    v,scope,margin = runs['candidate']
    assert np.array_equal(v[:,1],r[:,1]) and np.array_equal(v[~scope],r[~scope])
    gold = y[:,0].astype(bool)
    original = r[:,0]
    assert result['changes']['frozen']==f.comparison(gold,original,v[:,0])
    for rule in ('fp','tp'):
        prediction = original.copy()
        prediction[scope] = margin[scope]>cut(gold[scope],original[scope],margin[scope],rule)
        assert result['changes'][rule]==f.comparison(gold,original,prediction)
    differences = [b-a for a,b in zip(estimates['reference'],estimates['candidate'],strict=True)]
    assert differences==result['paired']
    c,b = (result['comparison'][name] for name in ('candidate','reference'))
    wins = sum(x>0 for x in differences)
    better = (wins>=5 and c['score']>max(baseline,b['score'])
              and c['quality']['mean']['f1']>b['quality']['mean']['f1'])
    assert better==result['promoted']
    return dict(scope=int(scope.sum()),wins=wins,pairs=len(differences),candidate=c['score'],reference=b['score'],
                result_sha256=checksum(p),promoted=better,prediction_bits=r.size)


def diversity():
    """Use development to choose a ranking per feature view, then count complementary errors."""
    definitions = {}
    for stage in ('comparison','special'):
        folder = HERE/stage
        with (folder/'comparison.csv').open() as src:
            rows = list(csv.DictReader(src))
        for view in sorted({r['view'] for r in rows}):
            subset = [r for r in rows if r['view']==view]
            r = max(subset,key=lambda q:(float(q['cal_f1']),-int(q['review_rows']),q['key'],q['rule']))
            definitions[stage,view] = (r['key'],r['rule'])
    quality = {}
    for part in PARTS:
        archives = {k:np.load(HERE/k/part/'predictions.npz') for k in ('comparison','special')}
        same = archives['comparison']
        base,y = same['base'],same['truth']
        states = {k:[base.copy(),base.copy()] for k in ('control','syntax')}
        for (stage,view),(key,rule) in definitions.items():
            store = archives[stage]
            assert np.array_equal(store['ids'],same['ids'])
            q = store[key+'_'+rule]
            separate = ('control','syntax') if view in ('numeric','lexical') else ('syntax',)
            for item in separate:
                selected,missed = states[item]
                selected |= q
                missed &= q
        result = {}
        for name,(up,down) in states.items():
            result[name] = dict(fn_covered=int((y & ~base & up).sum()),fp_covered=int((~y & base & ~down).sum()))
        quality[part] = result
    return dict(by_role=quality,policies={a+'/'+b:list(value) for (a,b),value in definitions.items()})

def main():
    cert = FEATURE/'receipt.json'
    rows = json_at(cert)
    assert rows['code_sha256']==checksum(HERE/'structure44.py')
    assert rows['roster_sha256']==checksum(FEATURE/'rows.npz')
    assert rows['parser_sha256']==checksum(ROOT/'round43/parser.sha256')
    assert rows['previous_extractor_sha256']==checksum(ROOT/'round43/features43.py')
    for view,metadata in rows['features'].items():
        assert metadata['sha256']==checksum(FEATURE/(view+'.npz'))
    bank = {}
    for role in PARTS:
        matrices,_,matrix_stamp = load(role)
        assert matrix_stamp==rows['sources'][role]
        bank[role] = parts(matrices)[1],matrix_stamp
    timing = json_at(HERE/'micro.json')
    assert timing['roster_sha256']==checksum(HERE/'micro_rows.npz')
    assert timing['readout_sha256']==checksum(HERE/'comparison/official/weights.joblib')
    assert timing['code_sha256']==checksum(HERE/'live44.py')
    assert timing['medians']=={k:median(v) for k,v in timing['samples'].items()}
    assert all(len(x)==5 for x in timing['samples'].values())
    p = ROOT/'submission/NLBSE2027_round38_paired_rule_tool.zip'
    score = json_at(ROOT/'round38/release_summary.json')
    assert score['archive_sha256']==checksum(p)
    outputs = {}
    for key,fit,pick in (('comparison','study44.py','select44.py'),('special','special44.py','special44.py')):
        outputs[key] = internal(HERE/key,fit,pick,bank,checksum(cert))
    receipt = dict(decisions=outputs,public=profile(p,score['score']),micro_sha256=checksum(HERE/'micro.json'),
                   confirmed_score=score['score'],gap=score['target_gap'],diversity=diversity(),
                   code_sha256=checksum(Path(__file__)))
    write_json(HERE/'audit.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
