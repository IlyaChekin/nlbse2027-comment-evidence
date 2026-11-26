"""Transfer held-file teacher order into an existing compact representation."""
import argparse
from pathlib import Path
from time import perf_counter
import sys

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'round41'))
from source41 import PARTS, checksum, crop, f, json_at, load, parts, supported, write_json
from data40 import features
from project_split import fingerprint

DEST = HERE/'fitted'
R41 = ROOT/'round41'
R37 = ROOT/'round37'
CATALOG = {'negative_consensus':False,'negative_veto':False,'negative_retained':True,
           'unqueried_miss':False,'unqueried_fp':True}
ARMS = {'label':(False,None,0),'soft':(True,None,0),
        'pair_gold':(False,'gold',0),'pair_order':(False,'order',0),
        'pair_joint':(True,'order',0),'pair_residual':(True,'residual',0),
        'soft_mlp':(True,None,24),'pair_mlp':(True,'order',24)}
DEVICES = 'cuda'
MIX = .70
TEMPERATURE = 1.7
EPOCHS = 16
BATCH = 3072
PAIR_MULT = .32


def positions(index,values):
    by_id = {v:i for i,v in enumerate(index)}
    return np.fromiter((by_id[v] for v in values),np.int64)


def matched(rotation,frame,eligible,teacher,meta,corpus):
    base = R37/'matched'
    receipt = json_at(base/'results.json')
    proof = receipt['analysis'][rotation]
    file = base/f'{rotation}.npz'
    assert receipt['code_sha256']==checksum(R37/'pairs37.py')
    assert proof['predictions_sha256']==checksum(file)
    assert proof['points_sha256']==checksum(ROOT/'round33/points'/rotation/'receipt.json')
    archive = np.load(file)
    groups = {}
    all_pos,all_neg = [],[]
    for k,swapped in CATALOG.items():
        a,b = (positions(frame['ids'],archive[k+'_'+name]) for name in ('anchor','control'))
        for name,rows in (('anchor',a),('control',b)):
            assert np.array_equal(frame['fold'][rows],archive[k+'_fold'])
            assert np.array_equal(frame['duplicates'][rows],archive[f'{k}_{name}_dup_mask'])
        if swapped:
            a,b = b,a
        assert frame['y'][a].all() and not frame['y'][b].any()
        lane = frame['observed_regime']
        assert (lane[a]==lane[b]).all() and not (lane[a]==1).any()
        keep = eligible[a]&eligible[b]
        a,b = a[keep],b[keep]
        reference = frame['signed'][a]-frame['signed'][b]
        g,u = (teacher[name][a]-teacher[name][b] for name in ('graph','unix'))
        consensus = (g>0.) & (u>0.) & (reference<=0.)
        groups[k] = dict(available=len(a),selected=int(consensus.sum()),anchors=len(np.unique(a[consensus])),
                         baseline_rate=float((reference>0.).mean()) if len(a) else None,
                         graph_rate=float((g>0.).mean()) if len(a) else None,
                         unix_rate=float((u>0.).mean()) if len(a) else None)
        all_pos.extend(a[consensus])
        all_neg.extend(b[consensus])
    roster = np.unique(np.column_stack((all_pos,all_neg)).astype(np.int64),axis=0)
    a,b = roster.T
    assert all((corpus[2][i],corpus[3][i])==(corpus[2][j],corpus[3][j])
               for i,j in zip(frame['ids'][a],frame['ids'][b],strict=True))
    assert np.isin(np.r_[frame['ids'][a],frame['ids'][b]],meta).all()
    return roster,groups,checksum(file)


def train(design, rank, y, soft, active, roster, mode, salt):
    point,kind,hidden = ARMS[mode]
    torch.manual_seed(4200+salt)
    rng = torch.Generator(device=DEVICES).manual_seed(4200+salt)
    ix = torch.as_tensor(np.flatnonzero(active),device=DEVICES)
    x = design[ix]
    mean = x.mean(0)
    scale = x.std(0).clamp_min(.045)
    x = ((x-mean)/scale).clamp(-8.,8.)
    rank = rank[ix].clamp(-12.,12.)
    labels = y[ix]
    targets = (soft[ix]/TEMPERATURE).sigmoid()
    weight = .45+1.7*(-.26*rank.abs()).exp()
    layer = nn.Linear(x.shape[1],1).cuda() if not hidden else nn.Sequential(
        nn.Linear(x.shape[1],hidden),nn.Tanh(),nn.Linear(hidden,1)).cuda()
    opt = torch.optim.AdamW(layer.parameters(),lr=.004,weight_decay=.06,fused=True)
    local = roster[active[roster].all(1)]
    a,b = (torch.as_tensor(np.searchsorted(ix.cpu().numpy(),q),device=DEVICES) for q in local.T)
    teacher_diff = soft[ix][a]-soft[ix][b]
    base_diff = MIX*(rank[a]-rank[b])
    if kind=='residual':
        desired = (teacher_diff-base_diff).clamp(-5.,5.)
    elif kind=='gold':
        desired = torch.ones_like(a,dtype=torch.float32)
    else:
        desired = (teacher_diff/TEMPERATURE).sigmoid()
    _,reverse,counts = np.unique(local[:,0],return_inverse=True,return_counts=True)
    pair_weight = torch.as_tensor(1./counts[reverse],device=DEVICES,dtype=torch.float32)
    if kind and not len(a):
        raise ValueError(f'No audited pair for {mode}')
    losses = []
    for step in range(EPOCHS):
        opt.param_groups[0]['lr'] = .0045*(1.-.77*step/(EPOCHS-1))
        batches = torch.randperm(len(x),generator=rng,device=DEVICES).split(BATCH)
        perm = torch.randperm(len(a),generator=rng,device=DEVICES)
        total = torch.zeros((),device=DEVICES)
        for j,ids in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            q = MIX*rank[ids]+layer(x[ids])[:,0]
            objective = F.binary_cross_entropy_with_logits(q,labels[ids],reduction='none')
            if point:
                kd = F.binary_cross_entropy_with_logits(q/TEMPERATURE,targets[ids],reduction='none')
                objective = .48*objective+.52*TEMPERATURE**2*kd
            loss = (weight[ids]*objective).mean()
            if kind:
                sub = perm[j::len(batches)]
                if len(sub):
                    delta = (layer(x[a[sub]])-layer(x[b[sub]]))[:,0]
                    pair_loss = (F.smooth_l1_loss(delta,desired[sub],reduction='none') if kind=='residual'
                                 else F.binary_cross_entropy_with_logits(delta+base_diff[sub],desired[sub],reduction='none'))
                    loss += PAIR_MULT*(pair_loss*pair_weight[sub]).sum()/pair_weight[sub].sum()
            loss.backward()
            opt.step()
            total += loss.detach()
        losses.append(float(total/len(batches)))
    w = list(layer.parameters())
    return dict(center=mean.cpu().numpy(),scale=scale.cpu().numpy(),
                params=[x.detach().cpu().numpy() for x in w],pairs=len(a),loss=losses)


def infer(x,rank,head):
    z = ((x-head['center'])/head['scale']).clip(-8.,8.)
    w,b,*layers = head['params']
    result = z@w.T+b
    if layers:
        w,b = layers
        result = np.tanh(result)@w.T+b
    return MIX*rank.clip(-12.,12.)+result[:,0]


def design(frame,threshold,states):
    # The trained head uses the pre-existing first-pass states and score features.
    own,_,_,_,regime = features(frame,threshold)
    assert np.array_equal(regime,frame['observed_regime'])
    return np.column_stack((own,states)).astype(np.float32)


def main(rotation):
    begin = perf_counter()
    matrices,threshold,data_hash = load(rotation)
    teacher_dir = R41/'heads'/rotation
    teacher = json_at(teacher_dir/'receipt.json')
    states_dir = R41/'compact'/rotation
    states = json_at(states_dir/'receipt.json')
    assert teacher['matrix_sha256']==states['matrix_sha256']==data_hash
    assert checksum(teacher_dir/'scores.npz')==teacher['predictions_sha256']
    assert checksum(states_dir/'states.npz')==states['state_sha256']==teacher['compact_sha256']
    teacher_scores = np.load(teacher_dir/'scores.npz')
    states_raw = np.load(states_dir/'states.npz')
    rows = [crop(m,supported(m)) for m in (matrices['oof'],*parts(matrices))]
    fit = rows[0]
    corpus = f.inputs()
    excluded = np.unique(np.r_[matrices['dev']['ids'],matrices.get('test',matrices['dev'])['ids']])
    duplicate = {fingerprint(corpus[0][i]) for i in excluded}
    eligible = np.fromiter((fingerprint(corpus[0][i]) not in duplicate for i in fit['ids']),bool)
    assert int((~eligible).sum())==teacher['removed_validation_copies']
    assert not np.intersect1d(fit['ids'],excluded).size
    for role,m in zip(('oof','cal','held'),rows,strict=True):
        assert np.array_equal(teacher_scores[role+'_ids'],m['ids'])
    assert np.array_equal(states_raw['fold'],fit['fold'])
    t = {name:teacher_scores['oof_'+name] for name in ('unix','graph','average')}
    roster,notes,source_pairs = matched(rotation,fit,eligible,t,teacher_scores['oof_ids'],corpus)
    mat = []
    for m,role in zip(rows,('oof','dev','test' if 'test' in matrices else 'dev'),strict=True):
        order = positions(states_raw[role+'_ids'],m['ids'])
        mat.append(design(m,threshold,states_raw[role][order]))
    del states_raw
    x = torch.as_tensor(mat[0],device=DEVICES)
    gold,score,target = (torch.as_tensor(a,device=DEVICES,dtype=torch.float32)
                         for a in (fit['y'],fit['signed'],t['average']))
    result,models = {},{}
    print(rotation,'pairs',len(roster),notes,'seconds',round(perf_counter()-begin,2),flush=True)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'pair_roster.npz',pair_ids=fit['ids'][roster],
                        fold=fit['fold'][roster[:,0]],teacher=np.column_stack((t['graph'][roster].T,t['unix'][roster].T)).T,
                        rank=fit['signed'][roster])
    for mode in ARMS:
        destination = [np.full(len(m['ids']),np.nan,np.float32) for m in rows[1:]]
        models[mode] = {}
        for side in (0,2):
            selected = (fit['observed_regime']==side)&eligible
            head = train(x,score,gold,target,selected,roster,mode,side)
            models[mode][side] = head
            for frame,matrix,pred in zip(rows[1:],mat[1:],destination,strict=True):
                mask = frame['observed_regime']==side
                pred[mask] = infer(matrix[mask],frame['signed'][mask],head)
        for name,pred in zip(('cal','held'),destination,strict=True):
            assert np.isfinite(pred).all()
            result[name+'_'+mode] = pred
        print(rotation,mode,round(perf_counter()-begin,2),flush=True)
    np.savez_compressed(out/'scores.npz',cal_ids=rows[1]['ids'],held_ids=rows[2]['ids'],**result)
    joblib.dump(models,out/'weights.joblib',compress=3)
    write_json(out/'receipt.json',dict(matrix_sha256=data_hash,exclude_copies=int((~eligible).sum()),
                teacher_sha256=checksum(teacher_dir/'receipt.json'),compact_sha256=checksum(states_dir/'receipt.json'),
                pair_source_sha256=source_pairs,pair_roster_sha256=checksum(out/'pair_roster.npz'),
                pair_count=len(roster),cohorts=notes,arms=ARMS,inputs=mat[0].shape[1],epochs=EPOCHS,
                scores_sha256=checksum(out/'scores.npz'),weights_sha256=checksum(out/'weights.joblib'),
                code_sha256=checksum(Path(__file__)),seconds=perf_counter()-begin))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
