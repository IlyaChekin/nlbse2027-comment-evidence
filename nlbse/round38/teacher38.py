"""Train and score a file-disjoint teacher for positive scanner responses."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data38 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, phases, routes, pairs
from collect_dense import DEST as DENSE
from matched_veto33 import evaluation

DEST = HERE/'positive_teacher'
HIDDEN = 96
EPOCHS = 8
BATCH = 128
PAIR = 64
SEED = 3831


class Readout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,HIDDEN),nn.LayerNorm(HIDDEN),
                                 nn.GELU(),nn.Dropout(.15),nn.Linear(HIDDEN,1))

    def forward(self,x):
        return self.net(x).squeeze(1)


def prepare_data(source, data, ids, y, scope):
    rows = data['ids'][scope]
    selected = np.searchsorted(ids,rows)
    assert np.array_equal(ids[selected],rows)
    assert np.array_equal(y[selected],data['y'][scope])
    return torch.as_tensor(source[selected],device='cuda',dtype=torch.float32)


def schedule(allowed,p,n,counts,seed):
    rng = np.random.default_rng(seed)
    plans = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(p))
        for batch in range(0,len(order),PAIR):
            choice = order[batch:batch+PAIR]
            various = rng.choice(allowed,size=BATCH,replace=False)
            plans.append((np.r_[p[choice],n[choice],various],counts[choice],len(choice)))
    return plans


def fit(x,y,margin,plans,scheme,seed):
    torch.manual_seed(seed)
    model = Readout(x.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(),lr=.0014,weight_decay=.025,fused=True)
    warm = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=len(plans),eta_min=.00018)
    elapsed = perf_counter()
    sums = y.new_zeros(2)
    for row,importance,count in plans:
        weight = torch.as_tensor(importance,device='cuda',dtype=torch.float32)
        imp = torch.cat((weight,weight,y.new_ones(len(row)-2*count)))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            residual = model(x[row]).float()
            out = .65*margin[row]+residual
            ce = F.binary_cross_entropy_with_logits(out,y[row],reduction='none')
            point = (ce*imp).sum()/imp.sum()
            difference = out[:count]-out[count:2*count]
            contrast = (F.softplus(-difference)*weight).sum()/weight.sum()
            loss = point+.65*contrast if scheme=='pair' else point
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        warm.step()
        sums+=torch.stack((point.detach(),contrast.detach()))
    return model.eval(),dict(seconds=perf_counter()-elapsed,updates=len(plans),loss=(sums/len(plans)).tolist())


@torch.inference_mode()
def predict(model,x):
    return torch.cat([model(b) for b in x.split(1024)]).cpu().numpy()


def main(rotation):
    phases_,ref,base = phases(rotation)
    report = json_at(DENSE/'receipt.json')
    assert checksum(DENSE/'features.npy')==report['archive_sha256']
    assert checksum(DENSE/'rows.npz')==report['rows_sha256']
    x = np.load(DENSE/'features.npy',mmap_mode='r')
    with np.load(DENSE/'rows.npz') as a:
        ids,lab = a['ids'],a['y']
    assert (ids[1:]>ids[:-1]).all()
    training = phases_['oof']
    picked = routes(training)['positive']
    ix = np.flatnonzero(picked)
    signals = prepare_data(x,training,ids,lab,picked)
    fold = training['fold'][picked]
    dup = training['duplicates'][picked]
    y = torch.as_tensor(training['y'][picked],device='cuda',dtype=torch.float32)
    margin = torch.as_tensor(training['latest'][picked],device='cuda',dtype=torch.float32).clamp(-7.,7.)
    a,b,signed = pairs(rotation,training,'positive_retained')
    left,right = np.searchsorted(ix,a),np.searchsorted(ix,b)
    assert np.array_equal(ix[left],a) and np.array_equal(ix[right],b)
    forecasts = {'oof_ids':training['ids'][picked],'fold':fold}
    summaries = {}
    held_x = {}
    for part,data in phases_.items():
        if part=='oof':
            continue
        mask = routes(data)['positive']
        held_x[part] = prepare_data(x,data,ids,lab,mask)
        forecasts[part+'_ids'] = data['ids'][mask]
    out = DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    for method in ('point','pair'):
        forecasts['oof_'+method] = np.empty(len(ix),np.float32)
    for k in np.unique(fold):
        allowed = (dup&(1<<int(k)))==0
        held = fold==k
        assert not (allowed&held).any()
        keep = allowed[left]&allowed[right]
        controls,anchors = left[keep],right[keep]
        _,owner,total = np.unique(anchors,return_inverse=True,return_counts=True)
        weights = 1./total[owner]
        weights /= weights.mean()
        plans = schedule(np.flatnonzero(allowed),controls,anchors,weights,SEED+int(k))
        mean = signals[allowed].mean(0)
        scale = signals[allowed].std(0).clamp_min(.015)
        z = ((signals-mean)/scale).clamp(-8.,8.)
        summaries[int(k)] = {}
        for name in ('point','pair'):
            model,details = fit(z,y,margin,plans,name,SEED+int(k))
            forecasts['oof_'+name][held] = predict(model,z[held])
            if k==0:
                for part,state in held_x.items():
                    forecasts[part+'_'+name] = predict(model,((state-mean)/scale).clamp(-8.,8.))
            path = out/f'{name}_{k}.pt'
            torch.save(dict(weights=model.cpu().state_dict(),mean=mean.cpu(),scale=scale.cpu()),path)
            summaries[int(k)][name] = details | {'weights_sha256':checksum(path),'fit':int(allowed.sum()),
                                                  'held':int(held.sum()),'pairs':int(keep.sum())}
            print(rotation,int(k),name,summaries[int(k)][name],flush=True)
    _,_,project,file,_ = f.inputs()
    groups = [(project[i],file[i]) for i in training['ids'][b]]
    # Anchor the averaging on the baseline false positive in each pair.
    base_margin = training['latest'][picked]
    pair_audit = {'reference':evaluation(-base_margin,right,left,groups)}
    for name in ('point','pair'):
        z = .65*np.clip(base_margin,-7.,7.)+forecasts['oof_'+name]
        pair_audit[name] = evaluation(-z,right,left,groups)
    scored = out/'scores.npz'
    np.savez_compressed(scored,**forecasts)
    write_json(out/'receipt.json',dict(rotation=rotation,rows=len(ix),base_sha256=base,folds=summaries,
                  embedding_sha256=report['archive_sha256'],pairs_sha256=signed,epochs=EPOCHS,
                  pair_audit=pair_audit,scores_sha256=checksum(scored),code_sha256=checksum(Path(__file__))))
    print(rotation,pair_audit,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
