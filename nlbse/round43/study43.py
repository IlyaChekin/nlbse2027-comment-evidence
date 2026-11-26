"""Compare first-hit syntax paths with matched text windows and numeric controls."""
import argparse
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import load_npz

from features43 import HERE, ROOT, R39, DEST as FEATURES, OUTPUT, f, checksum, json_at, write_json
from data39 import load_ready, archived, oof, pairs, crop, routes
from location39 import PARTS
from study39 import smooth_sparse
from rank38 import positive_features, fit, predict, cut, per_project, adjusted
from project_split import fingerprint
sys.path.insert(0,str(ROOT/'round40'))
from stack40 import diagnostic_order
from matched_veto33 import evaluation

DEST = HERE/'comparison'
READOUTS = {'geometry':(),'text':('text',),'masked':('masked',),'ast':('ast',),
            'ast_text':('text','ast'),'ast_masked':('masked','ast'),'center':('center',)}
VIEWS = ('text','masked','ast','center')
APERTURES = ('pooled','dominant','strong')
WIDTHS = (1.,2.5,4.,None)
MIX = (.35,.65,1.)
DOMINANT = 2


def available(frame):
    return routes(frame)['positive'] & (frame['location'][:,3]>0)


def eligible_scope(frame,panel,width):
    use = available(frame)
    if panel!='pooled':
        use &= frame['location'][:,1]==DOMINANT
    if panel=='strong':
        use &= frame['signed']>=1.8
    if width is not None:
        use &= abs(frame['signed'])<width
    return use


def design(frame,threshold,geometry,extra):
    base = positive_features(frame,threshold,frame['aux_positive'],frame['first'],True)
    return np.column_stack((base,geometry,*extra))


def sources(geometry_proof):
    card = json_at(FEATURES/'receipt.json')
    assert card['context_sha256']==checksum(HERE/'features43.py')
    assert card['previous_context_sha256']==checksum(R39/'context39.py')
    assert card['scanner_receipt_sha256']==geometry_proof==checksum(OUTPUT/'receipt.json')
    assert card['rows_sha256']==checksum(FEATURES/'rows.npz')
    assert card['parser_wheels_manifest_sha256']==checksum(HERE/'parser.sha256')
    cache = {}
    for name in VIEWS:
        file = FEATURES/(name+'.npz')
        assert checksum(file)==card['features'][name]['sha256']
        cache[name] = load_npz(file)
    with np.load(FEATURES/'rows.npz') as z:
        rows,geo = z['ids'],z['shape']
    for v in cache.values():
        assert v.shape[0]==len(rows)
    return rows,geo,cache,checksum(FEATURES/'receipt.json')


def sparse_first(rows,views,parts,positive,negative,permitted):
    out,heads = {},{}
    all_data = rows[0]
    for k,matrix in views.items():
        fit_mat = matrix[parts[0]]
        values = np.full(len(parts[0]),np.nan)
        for fold in np.unique(all_data['fold']):
            hold = all_data['fold']==fold
            use = permitted & ((all_data['duplicates']&(1<<int(fold)))==0)
            assert not (use&hold).any()
            same = use[positive]&use[negative]
            model = smooth_sparse(fit_mat,positive[same],negative[same])
            values[hold] = model.decision_function(fit_mat[hold])
        assert np.isfinite(values).all()
        all_good = permitted[positive]&permitted[negative]
        head = smooth_sparse(fit_mat,positive[all_good],negative[all_good])
        heads[k] = head
        out[k] = [values,*(head.decision_function(matrix[row]) for row in parts[1:])]
    return out,heads


def assemble(rotation):
    start = perf_counter()
    frames,ref,stamps,geometry_proof = load_ready(rotation)
    choice,baseline_hash = archived(rotation,frames,ref,stamps)
    folds,digest = oof(rotation,frames,ref['cutoff'],choice)
    source,geo,cache,receipt = sources(geometry_proof)
    prep = frames['dev']
    eval_rows = frames.get('test',prep)
    data = (frames['oof'],crop(prep,prep['cal']),crop(eval_rows,eval_rows['held']))
    scope = [available(z) for z in data]
    selected = [crop(z,active) for z,active in zip(data,scope,strict=True)]
    corpus,_,family,files,_ = f.inputs()
    absent = np.unique(np.r_[prep['ids'],eval_rows['ids']])
    copies = {fingerprint(corpus[i]) for i in absent}
    train = selected[0]
    permitted = np.fromiter((fingerprint(corpus[i]) not in copies for i in train['ids']),bool)
    assert not np.isin(train['ids'],absent).any()
    if 'test' in frames:
        assert not set(family[train['ids']])&set(family[eval_rows['ids']])
    positive,negative,digest0 = pairs(rotation,data[0],'positive_retained')
    assert digest0==digest
    same = scope[0][positive]&scope[0][negative]
    retained = np.flatnonzero(scope[0])
    a,b = (np.searchsorted(retained,ix[same]) for ix in (positive,negative))
    assert np.array_equal(retained[a],positive[same]) and np.array_equal(retained[b],negative[same])
    kept = permitted[a]&permitted[b]
    a,b = a[kept],b[kept]
    assert all((family[x],files[x])==(family[y],files[y])
               for x,y in zip(train['ids'][a],train['ids'][b],strict=True))
    indices = [np.searchsorted(source,r['ids']) for r in selected]
    assert all(np.array_equal(source[ix],r['ids']) for ix,r in zip(indices,selected,strict=True))
    raw,head = sparse_first(selected,cache,indices,a,b,permitted)
    pts = np.flatnonzero(permitted)
    pair = np.searchsorted(pts,a),np.searchsorted(pts,b)
    assert np.array_equal(pts[pair[0]],a) and np.array_equal(pts[pair[1]],b)
    heads,values = {},{}
    for name,extras in READOUTS.items():
        design_mats = [design(frame,ref['cutoff'],geo[ix],[raw[k][j] for k in extras])
                       for j,(frame,ix) in enumerate(zip(selected,indices,strict=True))]
        for loss in ('point','pair'):
            mode = name+'_'+loss
            p,n = pair if loss=='pair' else (a[:0],b[:0])
            model = fit(design_mats[0][permitted],train['signed'][permitted],train['y'][permitted],p,n)
            heads[mode] = model
            variants = []
            for matrix,frame,allowed in zip(design_mats[1:],selected[1:],scope[1:],strict=True):
                score = np.full(len(allowed),np.nan)
                score[allowed] = predict(model,matrix,frame['signed'])
                variants.append(score)
            values[mode] = variants
    files_used = [(family[i],files[i]) for i in train['ids'][b]]
    comparison = {'baseline':evaluation(-train['signed'],b,a,files_used)}
    for k,q in raw.items():
        comparison[k] = evaluation(-(train['signed']+.30*q[0]),b,a,files_used)
    detail = dict(pairs=len(a),row_counts=[len(z['ids']) for z in selected],folds=folds,
                  removed_token_copies=int((~permitted).sum()),pair_ranks=comparison,
                  pairs_sha256=digest,baseline_sha256=baseline_hash,cache_sha256=receipt,
                  geometry_sha256=geometry_proof,fit_seconds=perf_counter()-start)
    print(rotation,len(a),detail['row_counts'],
          {k:round(x['pair_win'],5) for k,x in comparison.items()},flush=True)
    return data[1:],values,(heads,head),detail


def compare(rotation,frames,variants):
    report = {}
    base = [q['signed']>0. for q in frames]
    truth = [q['y'] for q in frames]
    pred = dict(ids=frames[1]['ids'],base=base[1],truth=truth[1])
    for panel,width in product(APERTURES,WIDTHS):
        if panel=='strong' and width in (1.,2.5):
            continue
        scope = [eligible_scope(r,panel,width) for r in frames]
        if not all(m.any() for m in scope):
            continue
        for name,weight in [('threshold',0.),*product(variants,MIX)]:
            key = f'{panel}/{width}/{name}/{weight:g}'
            score = []
            for j,(q,m) in enumerate(zip(frames,scope,strict=True)):
                z = q['signed'][m]
                score.append(z if name=='threshold' else (1.-weight)*z+weight*variants[name][j][m])
            row = dict(panel=panel,width=width,arm=name,mix=weight,
                       cal_rows=int(scope[0].sum()),review_rows=int(scope[1].sum()))
            if rotation!='official':
                order,_ = diagnostic_order(score[1],frames[1]['signed'][scope[1]],frames[1]['ids'][scope[1]])
                row['matched'],d = per_project(truth[1],frames[1]['project'],base[1],scope[1],order,('fp','tp'))
                pred[key+'_same_fp'] = d['fp']
                pred[key+'_same_tp'] = d['tp']
            for mode in ('fixed','tp','fp'):
                c = cut(truth[0][scope[0]],base[0][scope[0]],score[0],mode)
                r,_ = adjusted(truth[0],base[0],scope[0],score[0],c)
                h,bits = adjusted(truth[1],base[1],scope[1],score[1],c)
                row[mode] = dict(cal=r,held=h,cut=c)
                pred[key+'_'+mode] = bits
            report[key] = row
    return report,pred


def main(rotation):
    frames,variants,models,details = assemble(rotation)
    results,preds = compare(rotation,frames,variants)
    folder = DEST/rotation
    folder.mkdir(exist_ok=True,parents=True)
    path = folder/'predictions.npz'
    model = folder/'weights.joblib'
    np.savez_compressed(path,**preds)
    joblib.dump(dict(readout=models[0],sparse=models[1]),model,compress=3)
    write_json(folder/'results.json',dict(rotation=rotation,options=results,training=details,
                   predictions_sha256=checksum(path),weights_sha256=checksum(model),code_sha256=checksum(Path(__file__))))
    print(rotation,len(results),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    main(parser.parse_args().rotation)
