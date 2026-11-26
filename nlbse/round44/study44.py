"""Compare region syntax and lexical controls on observed negative scanner results."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import HistGradientBoostingClassifier

from structure44 import HERE,ROOT,DEST as FEATURES,ARMS,PARTS,f,checksum,json_at,load,write_json
from source41 import crop, parts
from project_split import fingerprint
from points33 import OUT
from policy33 import matrix
from rank38 import fit, predict
from study43 import sparse_first
from stack40 import diagnostic_order, cut, adjusted, per_project
from matched_veto33 import evaluation

DEST = HERE/'comparison'
R37 = ROOT/'round37'
VIEWS = dict(numeric=(), lexical=('lexical',), signature=('signature',), body=('body',),
             middle=('middle',), usage=('usage',), syntax=('signature','body','usage'),
             syntax_text=('lexical','signature','body','usage'))
MODELS = ('linear','tree')
PANELS = ('all','consensus','veto')
WIDTHS = (.8,1.8,3.8,None)
MIXES = (.35,.7)
ROLES = {'negative_consensus':False,'negative_veto':False,'negative_retained':True}


def source_data(stamp,rotation):
    snapshot = FEATURES/'receipt.json'
    metadata = json_at(snapshot)
    assert metadata['sources'][rotation]==stamp
    assert metadata['code_sha256']==checksum(HERE/'structure44.py')
    assert metadata['previous_extractor_sha256']==checksum(ROOT/'round43/features43.py')
    assert metadata['parser_sha256']==checksum(ROOT/'round43/parser.sha256')
    assert metadata['roster_sha256']==checksum(FEATURES/'rows.npz')
    with np.load(FEATURES/'rows.npz') as v:
        ids = v['ids']
    bank = {}
    for key in ARMS:
        file = FEATURES/(key+'.npz')
        assert checksum(file)==metadata['features'][key]['sha256']
        bank[key] = load_npz(file)
        assert bank[key].shape[0]==len(ids)
    return ids,bank,checksum(snapshot)


def pairs(rotation,frame,keep,corpus):
    folder = R37/'matched'
    digest = json_at(folder/'results.json')
    assert digest['code_sha256']==checksum(R37/'pairs37.py')
    check = digest['analysis'][rotation]
    assert check['points_sha256']==checksum(OUT/rotation/'receipt.json')
    file = folder/(rotation+'.npz')
    assert check['predictions_sha256']==checksum(file)
    lookup = {row:i for i,row in enumerate(frame['ids'])}
    parts = {}
    with np.load(file) as values:
        for name,swap in ROLES.items():
            p,n = [np.asarray([lookup[i] for i in values[name+'_'+role]],dtype=np.int64)
                   for role in ('anchor','control')]
            for role,x in (('anchor',p),('control',n)):
                assert np.array_equal(frame['fold'][x],values[name+'_fold'])
                assert np.array_equal(frame['duplicates'][x],values[name+'_'+role+'_dup_mask'])
            if swap:
                p,n = n,p
            assert frame['y'][p].all() and not frame['y'][n].any()
            good = keep[p]&keep[n]
            a,b = p[good],n[good]
            ids = frame['ids']
            assert all((corpus[2][i],corpus[3][i])==(corpus[2][j],corpus[3][j])
                       for i,j in zip(ids[a],ids[b],strict=True))
            parts[name] = a,b
    return parts,checksum(file)


def route(frame):
    # Neither training nor inference uses outcomes from unperformed scanner queries.
    known_negative = frame['old'] & (frame['flag']==0)
    if 'observed_regime' in frame:
        assert np.array_equal(known_negative,frame['observed_regime']==0)
    return known_negative


def scope(frame,panel,width):
    mask = route(frame)
    if panel=='veto':
        mask &= frame['unqueried']>0.
    if panel=='consensus':
        mask &= frame['unqueried']<=0.
    if width is not None:
        mask &= abs(frame['signed'])<width
    return mask


def design(frame,t,extras):
    x = matrix(frame['pre'],frame['length'],t)
    signed = frame['signed'].clip(-10.,10.)
    prior = frame['unqueried'].clip(-10.,10.)
    decided = signed>0.
    unqueried = prior>0.
    flags = (2*unqueried+decided)[:,None]==np.arange(4)
    return np.column_stack((signed,prior,x[:,:14],flags,flags*signed[:,None],flags*prior[:,None],*extras))


def metrics(frame,roster,channels,files,corpus):
    answers = {}
    for family,(p,n) in roster.items():
        if not len(p):
            continue
        ids = frame['ids'][p]
        source = [(corpus[2][i],files[i]) for i in ids]
        margin = frame['signed']
        results = {'baseline':evaluation(margin,p,n,source)}
        for key,values in channels.items():
            results[key] = evaluation(margin+.45*values[0],p,n,source)
        answers[family] = results
    return answers


def fit_models(rotation):
    start = perf_counter()
    frames,t,stamp = load(rotation)
    original = (frames['oof'],*parts(frames))
    masks = [route(frame) for frame in original]
    fields = [crop(d,mask) for d,mask in zip(original,masks,strict=True)]
    train = fields[0]
    corpus = f.inputs()
    check = frames.get('test',frames['dev'])
    excluded = np.unique(np.r_[frames['dev']['ids'],check['ids']])
    held_words = {fingerprint(corpus[0][r]) for r in excluded}
    allowed = np.fromiter((fingerprint(corpus[0][i]) not in held_words for i in train['ids']),bool)
    assert not np.isin(train['ids'],excluded).any()
    if 'test' in frames:
        assert not set(corpus[2][train['ids']])&set(corpus[2][check['ids']])
    roster,pair_proof = pairs(rotation,train,allowed,corpus)
    joined = np.unique(np.column_stack(tuple(np.concatenate(x) for x in zip(*roster.values(),strict=True))),axis=0)
    p,n = joined.T
    ids,banks,features_proof = source_data(stamp,rotation)
    keys = [np.searchsorted(ids,frame['ids']) for frame in fields]
    assert all(np.array_equal(ids[k],r['ids']) for k,r in zip(keys,fields,strict=True))
    values,sparse = sparse_first(fields,banks,keys,p,n,allowed)
    findings = metrics(train,roster,values,corpus[3],corpus)
    print(rotation,'pairs',{k:len(x[0]) for k,x in roster.items()},
          {k:{m:round(w['pair_win'],4) for m,w in q.items()} for k,q in findings.items()},flush=True)
    models,result = {},{}
    keep = np.flatnonzero(allowed)
    pair = np.searchsorted(keep,p),np.searchsorted(keep,n)
    assert np.array_equal(keep[pair[0]],p) and np.array_equal(keep[pair[1]],n)
    truth = train['y'][allowed]
    margin = train['signed'][allowed]
    rate = .4+1.75*np.exp(-.52*abs(margin))
    for v,names in VIEWS.items():
        a,*pending = [design(frame,t,[values[key][j] for key in names]) for j,frame in enumerate(fields)]
        a = a[allowed]
        for kind in MODELS:
            key = v+'_'+kind
            if kind=='linear':
                model = fit(a,margin,truth,*pair)
            else:
                model = HistGradientBoostingClassifier(learning_rate=.075,max_iter=110,max_leaf_nodes=19,
                            min_samples_leaf=85,l2_regularization=9.,random_state=44,
                            monotonic_cst=[1]+[0]*(a.shape[1]-1),early_stopping=False)
                model.fit(a,truth,sample_weight=rate)
            estimates = []
            for phase,scope_bits,x in zip(fields[1:],masks[1:],pending,strict=True):
                prediction = np.full(len(scope_bits),np.nan)
                prediction[scope_bits] = (predict(model,x,phase['signed']) if kind=='linear'
                                          else model.decision_function(x))
                estimates.append(prediction)
            models[key],result[key] = model,estimates
    info = dict(source_sha256=stamp,features_sha256=features_proof,pairs_sha256=pair_proof,
                original_pairs={k:len(a) for k,(a,_) in roster.items()},deduplicated_pairs=len(p),
                support=[int(m.sum()) for m in masks],removed_copies=int((~allowed).sum()),
                ranking=findings,fit_seconds=perf_counter()-start)
    return original[1:],result,dict(readout=models,sparse=sparse),info


def assess(rotation,frames,variants):
    report = {}
    margins = [frame['signed'] for frame in frames]
    base = [z>0. for z in margins]
    y = [r['y'] for r in frames]
    saved = dict(ids=frames[1]['ids'],truth=y[1],base=base[1])
    for panel,width in product(PANELS,WIDTHS):
        pieces = [scope(r,panel,width) for r in frames]
        if not all(m.any() for m in pieces):
            continue
        for arm,mix in [('threshold',0.),*product(variants,MIXES)]:
            k = f'{panel}/{width}/{arm}/{mix:g}'
            z = [(1.-mix)*p[m]+mix*e[m] for p,m,e in
                 zip(margins,pieces,variants.get(arm,margins),strict=True)]
            point = dict(arm=arm,panel=panel,width=width,mix=mix,
                         cal_rows=int(pieces[0].sum()),review_rows=int(pieces[1].sum()))
            if rotation!='official':
                r,_ = diagnostic_order(z[1],margins[1][pieces[1]],frames[1]['ids'][pieces[1]])
                point['matched'],rows = per_project(y[1],frames[1]['project'],base[1],pieces[1],r,('fp','tp'))
                for rule in ('fp','tp'):
                    saved[k+'_same_'+rule] = rows[rule]
            for rule in ('fixed','fp','tp'):
                score = cut(y[0][pieces[0]],base[0][pieces[0]],z[0],rule)
                d,_ = adjusted(y[0],base[0],pieces[0],z[0],score)
                q,bits = adjusted(y[1],base[1],pieces[1],z[1],score)
                point[rule] = dict(cal=d,held=q,cut=score)
                saved[k+'_'+rule] = bits
            report[k] = point
    return report,saved


def main(rotation):
    frames,values,models,report = fit_models(rotation)
    results,bits = assess(rotation,frames,values)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'predictions.npz',**bits)
    joblib.dump(models,out/'weights.joblib',compress=3)
    write_json(out/'results.json',dict(options=results,training=report,
                   predictions_sha256=checksum(out/'predictions.npz'),weights_sha256=checksum(out/'weights.joblib'),
                   code_sha256=checksum(Path(__file__))))
    print(rotation,'variants',len(results),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
