"""Learn directional corrections and the expected benefit of reading an extra expert."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'round41'))
from source41 import PARTS, checksum, f, json_at, load, parts, write_json
from stack40 import model
from policy33 import matrix
from project_split import fingerprint

DEST = HERE/'fitted'
MODES = ('lex','dense')
DENSE = ('positive_point','positive_pair','negative','graph','unix')
SCALE = 6.
OPTIONS = ('point','numeric','lex','dense')
PHASES = ('oof','cal','held')


def common(frame,t):
    current = frame['signed'].clip(-12.,12.)
    prior = frame['unqueried'].clip(-12.,12.)
    values = matrix(frame['pre'],frame['length'],t)
    known = frame['old'] & (frame['flag']>=0)
    group = np.where(known,np.where(frame['flag']==0,0,1),2)
    if 'observed_regime' in frame:
        assert np.array_equal(group,frame['observed_regime'])
    hot = group[:,None]==np.arange(3)
    margin = np.column_stack((current,prior,values[:,0],values[:,5],values[:,6],values[:,7]))
    side = np.column_stack((hot,hot*current[:,None],hot*prior[:,None]))
    return np.column_stack((margin,values[:,:14],side)),group


def lexical(frame):
    a,b = frame['sparse'].T.clip(-12.,12.)
    c = frame['new'].clip(-12.,12.)
    before = frame['signed'].clip(-12.,12.)
    return np.column_stack((a,b,c,a-b,a-c,b-c,a-before,b-before,c-before))


def experts(frame,t,add=None):
    base,groups = common(frame,t)
    if add is None:
        return base
    x = lexical(frame)
    features = (base,x)
    if not isinstance(add,str):
        present = np.isfinite(add)
        g = np.where(present,add,0.).clip(-10.,10.)
        distance = np.where(present,g-frame['signed'][:,None],0.).clip(-12.,12.)
        features += (g,present,distance)
    return np.column_stack(features)


def aligned(selected,source,keys):
    ix,where,other = np.intersect1d(selected,source,return_indices=True)
    assert np.array_equal(selected[where],source[other]) and len(ix)==len(where)
    result = []
    for value in keys:
        empty = np.full(len(selected),np.nan,np.float32)
        empty[where] = value[other]
        result.append(empty)
    return result,where,other


def teacher_bank(rotation,data,stamp):
    choices = [data['oof'],*parts(data)]
    roles = ('oof','dev','test' if 'test' in data else 'dev')
    used = [list() for _ in choices]
    sources = {}
    for family in ('positive','negative'):
        folder = ROOT/'round38'/f'{family}_teacher'/rotation
        receipt = folder/'receipt.json'
        proof = json_at(receipt)
        expected = 'teacher38.py' if family=='positive' else 'negative38.py'
        assert proof['code_sha256']==checksum(ROOT/'round38'/expected)
        assert proof['rotation']==rotation
        assert proof['scores_sha256']==checksum(folder/'scores.npz')
        with np.load(folder/'scores.npz') as z:
            for k,(frame,phase) in enumerate(zip(choices,roles,strict=True)):
                options = ((z[phase+'_point'],z[phase+'_pair']) if family=='positive' else (z[phase],))
                dest,match,old = aligned(frame['ids'],z[phase+'_ids'],options)
                signed = frame['observed_regime']
                assert (signed[match]==(1 if family=='positive' else 0)).all()
                if phase=='oof':
                    assert np.array_equal(frame['fold'][match],z['fold'][old])
                used[k].extend(dest)
        sources[family]=checksum(receipt)
    folder = ROOT/'round41/heads'/rotation
    receipt = folder/'receipt.json'
    proof = json_at(receipt)
    assert proof['matrix_sha256']==stamp
    assert proof['predictions_sha256']==checksum(folder/'scores.npz')
    assert proof['code_sha256']==checksum(ROOT/'round41/heads41.py')
    with np.load(folder/'scores.npz') as z:
        for k,(frame,phase) in enumerate(zip(choices,PHASES,strict=True)):
            dest,match,_ = aligned(frame['ids'],z[phase+'_ids'],[z[phase+'_graph'],z[phase+'_unix']])
            assert np.array_equal(sorted(match),np.flatnonzero(frame['observed_regime']!=1))
            used[k].extend(dest)
    sources['strong']=checksum(receipt)
    return choices,[np.column_stack(d) for d in used],sources


def learn(x,y,rank,available,directional):
    focus = .38+1.62*np.exp(-.45*abs(rank))
    old = rank>0.
    if not directional:
        return model(x[available],y[available],focus[available],C=.30)
    group = {}
    for reverse in (False,True):
        use = available&(old==reverse)
        wrong = y[use]!=reverse
        group[reverse] = model(x[use],wrong,focus[use],C=.30)
    return group


def score(head,x,rank,directional=True):
    if not directional:
        return head.decision_function(x)
    z = np.empty(len(rank))
    bit = rank>0.
    for reverse,learner in head.items():
        active = bit==reverse
        if active.any():
            z[active] = (1-2*reverse)*learner.decision_function(x[active])
    return z


def query_model(x,old,good,has,fit):
    target = np.zeros(len(has),np.int8)
    target[has & good] = 1
    target[has & ~good] = 2
    result = {}
    for positive in (False,True):
        keep = fit & (old==positive)
        assert len(np.unique(target[keep]))==3
        result[positive] = model(x[keep],target[keep],np.ones(keep.sum()),C=.30)
    return result


def priority(models,x,rank,benefit,cost):
    old = rank>0.
    value = np.empty(len(rank))
    for p,clf in models.items():
        side = old==p
        if not side.any():
            continue
        z = clf.predict_proba(x[side])
        # Estimate the F1 gain in units of the harm of accepting one false positive.
        if p:
            value[side] = z[:,1]-benefit*z[:,2]-cost
        else:
            value[side] = benefit*z[:,1]-z[:,2]-cost
    return value


def query_costs():
    receipt = ROOT/'round41/graph/receipt.json'
    p = json_at(receipt)
    time = p['cost']['forward_seconds']/p['rows']
    flops = p['cost']['analytic_gflops']/p['rows']
    # Sparse / dense costs are illustrative screening prices.
    # Dense uses double the measured GraphCodeBERT rate to approximate a two-encoder read.
    part = ROOT/'round40/public_combined/results.json'
    candidate = json_at(part)['comparison']['candidate']['cost']
    score = json_at(ROOT/'round38/release_summary.json')
    v = score['vulnerability']
    loss = .3*2*v['tp']/((2*v['tp']+v['fn']+v['fp'])*(2*v['tp']+v['fn']+v['fp']+1))
    # No unmeasured cost is used as evidence of an improvement.
    return dict(lex=0.,dense=(2*time*.2/1000.+2*flops*.2/3_000_000)/loss),{
        'dense_receipt':checksum(receipt),'sparse_profile':checksum(part),
        'sparse_profile_new_rows':candidate['new_rows'],'fp_score_units':loss}


def main(rotation):
    start = perf_counter()
    data,t,stamp = load(rotation)
    frames,dense,proof = teacher_bank(rotation,data,stamp)
    training = frames[0]
    old = training['signed']>0.
    direction = np.where(old,-1.,1.)
    known = training['y']
    corp = f.inputs()
    dest = data.get('test',data['dev'])
    held = np.unique(np.r_[data['dev']['ids'],dest['ids']])
    word = {fingerprint(corp[0][i]) for i in held}
    eligible = np.fromiter((fingerprint(corp[0][i]) not in word for i in training['ids']),bool)
    assert not np.isin(training['ids'],held).any()
    if 'test' in data:
        assert not set(corp[2][training['ids']])&set(corp[2][dest['ids']])
    screens = [experts(q,t) for q in frames]
    attrs = {'numeric':screens,'point':screens,
             'lex':[experts(q,t,'lex') for q in frames],
             'dense':[experts(q,t,b) for q,b in zip(frames,dense,strict=True)]}
    # Local competition-score exchange rate for returning a TP versus adding an FP.
    tp = (old & known & eligible).sum()
    trade = ((known & eligible).sum()+(old & ~known & eligible).sum())/max(1,tp)
    fees,cost_proof = query_costs()
    cross = {k:np.full(len(old),np.nan) for k in ('numeric',*MODES)}
    observed = {}
    evidence = {}
    folds = training['fold']
    for fold in np.unique(folds):
        permitted = eligible & ((training['duplicates']&(1<<int(fold)))==0)
        active = folds==fold
        assert not (permitted&active).any()
        learned = {}
        for arm in cross:
            x = attrs[arm][0]
            head = learn(x,known,training['signed'],permitted,True)
            cross[arm][active] = score(head,x[active],training['signed'][active])
            learned[arm] = score(head,x[permitted],training['signed'][permitted])
        for arm in MODES:
            use = direction[permitted]
            offset = (learned[arm]-learned['numeric'])*use
            limits = [max(0.,float(np.quantile(offset[use==v],.74))) for v in (-1.,1.)]
            amount = direction[active]*(cross[arm][active]-cross['numeric'][active])
            gate = amount>np.where(direction[active]<0.,limits[0],limits[1])
            evidence.setdefault(arm,np.zeros(len(old),bool))[active] = gate
        observed[int(fold)] = dict(train=int(permitted.sum()),hold=int(active.sum()))
    assert all(np.isfinite(z).all() for z in cross.values())
    estimates,modelset = {},{}
    saved = {}
    receipts = {}
    good = known!=old
    for name,x in attrs.items():
        head = learn(x[0],known,training['signed'],eligible,name!='point')
        modelset[name] = head
        estimates[name] = [score(head,a,q['signed'],name!='point')
                           for a,q in zip(x[1:],frames[1:],strict=True)]
    for name in MODES:
        gate = query_model(screens[0],old,good,evidence[name],eligible)
        modelset['gate_'+name] = gate
        receipts[name] = dict(proposals=int((evidence[name]&eligible).sum()),
                              useful=int((evidence[name]&good&eligible).sum()),
                              harmful=int((evidence[name]&~good&eligible).sum()),cost=fees[name])
        estimates['gate_'+name] = [priority(gate,x,q['signed'],trade,fees[name])
                                  for x,q in zip(screens[1:],frames[1:],strict=True)]
    for phase,frame in zip(('cal','held'),frames[1:],strict=True):
        i = int(phase=='held')
        saved[phase+'_ids'] = frame['ids']
        saved.update({phase+'_'+key:z[i] for key,z in estimates.items()})
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'scores.npz',**saved)
    np.savez_compressed(out/'crossfit.npz',ids=training['ids'],fold=folds,eligible=eligible,old=old,
                        y=known,**cross,**{k+'_proposal':v for k,v in evidence.items()})
    joblib.dump(modelset,out/'weights.joblib',compress=3)
    write_json(out/'receipt.json',dict(matrix_sha256=stamp,teachers=proof,price_note=cost_proof,trade=trade,
                   queries=receipts,folds=observed,excluded=int((~eligible).sum()),seconds=perf_counter()-start,
                   files={key:checksum(out/key) for key in ('weights.joblib','scores.npz','crossfit.npz')},
                   code_sha256=checksum(Path(__file__))))
    print(rotation,round(perf_counter()-start,2),receipts,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
