"""Compare encoder continuations with the signed Round 29 route and false-positive budgets."""
import argparse
from itertools import product
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from common31 import HERE, ROOT, OUT, PARTS, R28, f, checksum, json_at, write_json, reference, match, strict_cut
from continue_fit import MODES
from disagreement import clip_logit, trained_logistic
from evaluate_rules import infer
from fuse_sparse import calibrate_sparse
from project_eval import OP
from length_eval import REFERENCE, R24, base_context, calibrate, exact_state, fit_heads, load_sparse, recipe, sparse_calibration
from normalize_code import batch as canonicalize
from positive_runtime import scope, signals, proposed, ranking
from section_runtime import compute
from student_runtime import head, predict

R25 = ROOT/'round25/views'
R27 = ROOT/'round27'
WIDTHS = (.4, .9, 1.8, None)


def prepare(rotation, dev, test, fit, cal):
    folder = R24/rotation
    rec = json_at(folder/'fit.json')['models']['token']
    path = folder/'token.joblib'
    if checksum(path) != rec['weights_sha256']:
        raise ValueError('The signed token expert has changed')
    expert = joblib.load(path)
    raw = [load_sparse(folder, 'token', phase, ctx.frame, rec)
           for phase, ctx in zip(('dev', 'test'), (dev,) if dev is test else (dev, test))]
    if rotation == 'official':
        primary, routing, heads = recipe()
        gate, cutoff = primary['gate'], primary['threshold']
        path = folder/'tuned/models.joblib'
        cfg = json_at(folder/'tuned/policy.json')
        if checksum(path) != cfg['models_sha256']:
            raise ValueError('The frozen token calibration has changed')
        package = joblib.load(path)['token']
        oof = package['oof']
        combiner = package['combiner']
        variant = cfg['candidates'][cfg['selected']]
    else:
        budget = json_at(f.HERE/'factor_numeric'/rotation/'study.json')['dev_fp']
        heads, gate, cutoff, _, _ = calibrate({'dev':dev.frame, 'test':test.frame}, REFERENCE, fit, cal, budget)
        routing = OP
        oof = sparse_calibration(folder, 'token', rec)
        combiner = fit_heads(dev.frame, dev.state, calibrate_sparse(oof, raw[0]), fit, cutoff)
        variant = json_at(folder/'conditioned/results.json')['options']['token:no:blend:0.5']
    w25 = json_at(R25/'public/policy.json')['selected']
    r25 = json_at(R25/rotation/'results.json')
    h25 = R25/rotation/'calibrator.joblib'
    if checksum(h25) != r25['model_sha256']:
        raise ValueError('The frozen regional readout has changed')
    section = r25['results'][w25]
    section_head = joblib.load(h25)[section['region']]
    p27 = json_at(R27/'public/policy.json')
    selected = p27['selected']
    name, criterion = selected.rsplit(':', 1)
    r27 = R27/'joint'/rotation
    if checksum(r27/'results.json') != p27['table_sha256'][rotation]:
        raise ValueError('The frozen positive correction has changed')
    r27_record = json_at(r27/'results.json')
    pos = p27['supported'][selected]
    pos_cfg = pos['online'] | {'cut':r27_record['options'][name]['cuts'][criterion]['cut']}
    if checksum(r27/'weights.joblib') != r27_record['weights_sha256']:
        raise ValueError('The frozen positive correction weights have changed')
    pos_head = joblib.load(r27/'weights.joblib')[pos['weights_key']]['model']
    cfg = json_at(R28/'recover_public/policy.json')
    chosen = json_at(R28/'recover_public/release_selection.json')['selected']
    name, criterion = chosen.rsplit(':', 1)
    src = R28/'recover_calibrated'/rotation
    result = json_at(src/'results.json')
    if checksum(src/'results.json') != cfg['tables_sha256'][rotation]:
        raise ValueError('The frozen correction has changed')
    if checksum(src/'heads.joblib') != result['weights_sha256']:
        raise ValueError('The frozen correction weights have changed')
    rec_cfg = cfg['variants'][chosen]['setting']['recover'] | {
        'cut':result['options'][name]['cuts'][criterion]['cut']}
    rec_head = joblib.load(src/'heads.joblib')['compact']
    _, name, criterion = json_at(R28/'public/release_selection.json')['selected'].split(':')
    neg = json_at(R28/'studies/sparse'/rotation/'results.json')['options'][name]
    return [dict(heads=heads, gate=gate, routing=routing, cutoff=cutoff, variant=variant, combiner=combiner,
                 sparse=calibrate_sparse(oof, scores), raw=scores, expert=expert[0],
                 section=section, section_head=section_head, pos_cfg=pos_cfg, pos_head=pos_head,
                 rec_cfg=rec_cfg, rec_head=rec_head, neg=neg, neg_criterion=criterion) for scores in raw]


def state_for(ctx, values, p):
    frame = dict(ctx.frame, **{REFERENCE:values})
    route = exact_state(frame, p['heads'], p['gate'], p['routing'])
    return base_context(frame,route,p['sparse'],p['combiner'],p['cutoff'],p['variant']['threshold'],ctx.texts)


def section_margin(ctx, p, cache):
    section = p['section']
    relevant = ctx.active & (abs(ctx.current-ctx.cutoff)<section['distance'])
    seen = ctx.active & ~cache['available']
    if seen.any():
        extra, _ = compute([s for s,b in zip(ctx.texts,seen,strict=True) if b],section['region'],p['expert'])
        cache['value'][seen] = extra
        cache['available'] |= seen
    margin = clip_logit(ctx.current.astype(np.float64))-clip_logit(np.asarray(np.float32(ctx.cutoff),dtype=np.float64))
    if relevant.any():
        x = np.column_stack((ctx.inputs[ctx.active],cache['value'][ctx.active]))
        q = p['section_head'].predict_proba(x)[:,1]
        r = relevant[ctx.active]
        w = section['damping']
        blended = ((1.-w)*ctx.current[ctx.active]+w*q)[r]
        margin[relevant] = clip_logit(blended) - clip_logit(np.asarray(section['threshold']))
    return margin


def neg_cut(ctx, margin, p, cal):
    opt = p['neg']
    use = ctx.active & cal & (abs(margin)<opt['width'])
    y = ctx.frame['y'][use]
    values = np.sort(margin[use][y>0])[::-1]
    positives = int((y & (margin[use]>0.)).sum())+opt['cuts'][p['neg_criterion']]['cal']['delta_tp']
    return float(np.nextafter(values[positives-1],-np.inf)) if positives else float(max(margin[use]))


def replay(ctx, raw_margin, p, shift):
    state = ctx.state
    rank = clip_logit(state['p'].astype(np.float64))-clip_logit(np.asarray(np.float32(p['cutoff']), dtype=np.float64))
    opt = p['pos_cfg']
    flag, view, anchor = scope(ctx.texts,state,p['cutoff'],opt['width'])
    if flag.any():
        texts = [s for s,on in zip(ctx.texts,flag,strict=True) if on]
        x = signals(ctx.frame[REFERENCE][flag],texts,p['raw'][flag,1])
        v = proposed(x,texts,view[flag],anchor,opt['mode'],p['pos_head'])
        rank[flag] = ranking(view[flag],v,opt['mix'],opt['direction'])-opt['cut']
    rank[ctx.active] = raw_margin[ctx.active]-shift
    rec = p['rec_cfg']
    active = ctx.active & (abs(rank)<rec['width'])
    if active.any():
        old = rank[active]
        matrix = np.column_stack((raw_margin[active],old,ctx.inputs[active,:3]))
        w = rec['mix']
        delta = (1.-w)*old + w*p['rec_head'].decision_function(matrix)
        if rec['direction'] == 'recover':
            delta = np.where(old>0.,old,np.maximum(old,delta))
        rank[active] = delta-rec['cut']
    return rank


def arm_forward(rotation, mode, receipt, contexts):
    torch_path = OUT/rotation/f'{mode}.pt'
    if checksum(torch_path) != receipt['arms'][mode]['sha256']:
        raise ValueError('Continuation weights have changed')
    tok = f.small.tokenizer()
    model = f.pipe(rotation, tok).encoder
    learned = torch.load(torch_path,map_location='cpu',weights_only=True)
    model.load_state_dict(learned['backbone'],strict=False)
    add = head(model.width).cuda().eval()
    add.load_state_dict(learned['head'])
    canonical = SimpleNamespace(encoder=model.encoder,tok=tok,mode='control')
    route = f.pinned()[0]
    outcomes = []
    for ctx in contexts:
        raw, aux, ext, cost = predict(model,tok,ctx.texts,(add,))
        view = (raw[:,0]>route['lower']) & (raw[:,0]<route['upper'])
        if view.any():
            text = canonicalize([s for s,b in zip(ctx.texts,view,strict=True) if b])
            second, stats = infer(canonical,text,encoding=f.tokenize)
            raw[view,0] = .5*(raw[view,0]+second[:,0])
            for key in ('forward_seconds','wall_seconds','analytic_gflops'):
                cost[key] += stats[key]
        values = np.column_stack((raw[:,0],aux[:,0],aux[:,-2:]))
        outcomes.append((values,ext[:,0].astype(np.float64),cost | {'canonical_rows':int(view.sum())}))
    return outcomes


def fits_and_scores(contexts, core, add, widths, fit):
    train = contexts[0]
    sx = []
    for ctx, rank, vals in zip(contexts,core,add,strict=True):
        values, extra, _ = vals
        delta = extra-clip_logit(values[:,2].astype(np.float64))
        sx.append(np.column_stack((rank,extra,delta)))
    selected = fit & train.active
    readout = trained_logistic(sx[0][selected],train.frame['y'][selected],c=.7)
    scored = [('core', core, [ctx.active for ctx in contexts])]
    for width, weight in product(widths,(.35,.7)):
        rank, scope = [], []
        for ctx, current, features in zip(contexts,core,sx,strict=True):
            active = ctx.active.copy()
            if width is not None:
                active &= abs(current)<width
            moved = current.copy()
            moved[active] = (1.-weight)*current[active]+weight*readout.decision_function(features[active])
            rank.append(moved)
            scope.append(active)
        name = f'{width:g}_{weight:g}' if width is not None else f'all_{weight:g}'
        scored.append((name,rank,scope))
    return scored, readout


def matched_budget(y, old, rank, scope, projects):
    # Drift outside W0 is part of the candidate. Use a single per-project cutoff in W0
    # to restore the signed total FP count. Report whenever W0 cannot meet that budget.
    pred = rank>0.
    status = {}
    for project in np.unique(projects):
        at = projects == project
        scoped = at & scope
        budget = int((at & old & ~y).sum()) - int((at & ~scoped & pred & ~y).sum())
        if budget < 0 or budget > int((scoped & ~y).sum()):
            status[project] = dict(unattainable_fp_budget=budget)
            continue
        if scoped.any():
            pred[scoped], _ = f.exact_budget(rank[scoped],y[scoped],budget)
        result = f.comparison(y[at],old[at],pred[at])
        if result['delta_fp'] != 0:
            raise ValueError('The project match changed the total FP count')
        status[project] = result
    return pred,status


def compare(rotation):
    dev,test,fit,cal,review,original0,original1,m0,m1 = reference(rotation)
    frames = (dev,test) if review is None else (dev,)
    held = np.ones(len(test.texts),dtype=bool) if review is None else review
    recipes = prepare(rotation,dev,test,fit,cal)
    caches = [dict(available=np.zeros(len(ctx.texts),bool),value=np.empty((len(ctx.texts),4))) for ctx in frames]
    base_margin = [section_margin(c,p,cache) for c,p,cache in zip(frames,recipes,caches,strict=True)]
    shift = neg_cut(dev,base_margin[0],recipes[0],cal)
    for i, (ctx, orig, margins) in enumerate(zip(frames,(original0,original1)[:len(frames)],(m0,m1)[:len(frames)],strict=True)):
        restored = replay(ctx,base_margin[i],recipes[i],shift)
        if not np.array_equal(restored>0.,orig) or not np.allclose(restored[ctx.active],margins[ctx.active],atol=1e-10):
            raise ValueError(f'Independent baseline playback differs on frame {i}')
    receipt = json_at(OUT/rotation/'fit.json')
    y0 = dev.frame["y"].astype(bool)
    y1 = test.frame["y"].astype(bool)
    dest = HERE/'continuation'/rotation
    dest.mkdir(parents=True,exist_ok=True)
    arrays = dict(ids=test.frame['ids'][held],truth=y1[held],base=original1[held],base_negative=test.active[held])
    arms = {}
    models = {}
    for mode in MODES:
        scored = arm_forward(rotation,mode,receipt,frames)
        updated = [state_for(ctx,row[0],rec) for ctx,row,rec in zip(frames,scored,recipes,strict=True)]
        margin = [section_margin(c,p,cache) for c,p,cache in zip(updated,recipes,caches,strict=True)]
        core = [replay(c,m,p,shift) for c,m,p in zip(updated,margin,recipes,strict=True)]
        diagnostics = {}
        for phase, old, new, values, subset in (
            ('cal',dev,updated[0],scored[0][0],cal),('held',test,updated[-1],scored[-1][0],held),
        ):
            drift = abs(old.frame[REFERENCE][subset]-values[subset]).astype(np.float64)
            a = old.state['asked'][subset]
            b = new.state['asked'][subset]
            diagnostics[phase] = dict(old_scan=int(a.sum()),new_scan=int(b.sum()),scan_changes=int((a!=b).sum()),
                                      negative_changes=int((old.active[subset]!=new.active[subset]).sum()),
                                      channel_mae=drift.mean(0).tolist(),channel_delta_99=np.quantile(drift,.99,axis=0).tolist())
        choices, fit_readout = fits_and_scores(updated,core,scored,WIDTHS,fit)
        models[mode] = fit_readout
        for name, values, touched in choices:
            a,b = values[0],values[-1]
            sa,sb = touched[0],touched[-1]
            if name == 'core':
                sa = np.zeros_like(sa)
            options = {}
            if review is None:
                bits, projects = matched_budget(y1,original1,b,updated[-1].active,test.frame['project'])
                options['matched'] = f.comparison(y1,original1,bits)
                options['matched_complete'] = all('delta_fp' in item for item in projects.values())
                options['projects'] = projects
                arrays[mode+'_'+name+'_matched'] = bits[held]
            has_scope = sa.any()
            for key in ('fp','fixed'):
                if key=='fixed' or not has_scope:
                    cutoff = 0.
                else:
                    allowed = int((cal & original0 & ~y0).sum()) - int((cal & ~sa & (a>0.) & ~y0).sum())
                    if allowed < 0:
                        continue
                    local = cal & sa
                    _, cutoff = f.strict_budget(a[local],y0[local],allowed)
                pa,pb = a>0.,b>0.
                if name!='core':
                    pa[sa],pb[sb] = a[sa]>cutoff,b[sb]>cutoff
                options[key] = dict(cut=cutoff,cal=f.comparison(y0[cal],original0[cal],pa[cal]),
                                    frozen=f.comparison(y1[held],original1[held],pb[held]))
                arrays[mode+'_'+name+'_'+key] = pb[held]
            diagnostics[name] = options
        arms[mode] = diagnostics
        arrays[mode+'_scope'] = updated[-1].active[held]
        arrays[mode+'_baseline_rank'] = core[-1][held]
        arrays[mode+'_head_logits'] = scored[-1][1][held]
        print(rotation,mode,diagnostics['held'],diagnostics['core'],flush=True)
    path = dest/'predictions.npz'
    np.savez_compressed(path,**arrays)
    joblib.dump(models,dest/'readouts.joblib',compress=3)
    write_json(dest/'results.json',dict(arms=arms,rotation=rotation,reference_replayed=True,reference_shift=shift,
               fit_receipt_sha256=checksum(OUT/rotation/'fit.json'),predictions_sha256=checksum(path),
               readout_sha256=checksum(dest/'readouts.joblib'),code_sha256=checksum(HERE/'continue_eval.py')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    compare(parser.parse_args().rotation)
