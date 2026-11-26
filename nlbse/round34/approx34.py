"""Approximate the token attention scores with the compact encoder's existing pooled vector."""
import argparse

import numpy as np
import torch

from sections34 import HERE, PARTS, f, checksum, json_at, write_json
from eval34 import loaded
from attention34 import DEST as SOURCE
from compact34 import cache

DEST=HERE/'approximation'
MODES={'ridge_lo':.04,'ridge_med':.18,'ridge_hi':.60}


def ridge(x, y, w, reg):
    mean=(w@x)/w.sum()
    center=(w@y)/w.sum()
    h=x-mean
    base=h.T@(w[:,None]*h)
    base.diagonal().add_(reg*w.sum())
    target=h.T@(w[:,None]*(y-center))
    weight=torch.linalg.solve(base,target)
    return weight,center-(mean@weight)


def main(rotation):
    data,points,teacher=loaded(rotation)
    scope=SOURCE/rotation
    info=json_at(scope/'receipt.json')
    file=scope/'scores.npz'
    assert checksum(file)==info['predictions_sha256']
    texts=f.inputs()[0]
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    pooled={}
    forecasts={}
    with np.load(file) as z:
        for name,record in data.items():
            index=record['ids'][record['support']]
            assert np.array_equal(index,z[name+'_ids'])
            local=cache(model,tok,[texts[i] for i in index])[:,:2*model.width].float()
            pooled[name]=local
            forecasts[name+'_ids']=index
        labels=torch.as_tensor(z['oof_distilled_pairs'],device='cuda',dtype=torch.float32).clamp(-8.,8.)
    dw=data['oof']['support']
    q=torch.as_tensor(data['oof']['queried'][dw],device='cuda',dtype=torch.float32)
    weight=.30+1.8*(-abs(q)).exp()
    out=DEST/rotation
    out.mkdir(parents=True,exist_ok=True)
    arms={}
    for name,reg in MODES.items():
        coef,bias=ridge(pooled['oof'],labels,weight,reg)
        path=out/f'{name}.pt'
        torch.save(dict(weight=coef.T.contiguous().cpu(),bias=bias.cpu()),path)
        for slot,state in pooled.items():
            forecasts[slot+'_'+name]=(state@coef+bias).cpu().numpy()
        error=np.abs(forecasts['oof_'+name]-labels.cpu().numpy())
        arms[name]=dict(sha256=checksum(path),reg=reg,errors=error.mean(0).tolist())
        print(rotation,name,arms[name],flush=True)
    target=out/'scores.npz'
    np.savez_compressed(target,**forecasts)
    write_json(out/'receipt.json',dict(points=points['phases'],teacher_sha256=teacher['predictions_sha256'],
                    source_sha256=info['predictions_sha256'],code_sha256=checksum(HERE/'approx34.py'),
                    predictions_sha256=checksum(target),heads=arms))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
