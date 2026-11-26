"""Train matched pointwise and paired heads on the same frozen representations."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from pairs import HERE, OUT, PARTS, f, checksum, json_at, write_json
from head_runtime import head, collect

MODE = ('point', 'pair')
DEST = HERE/'heads'
EPOCHS = 3
BATCH = 384


def fit(where):
    signed = OUT/where/'manifest.json'
    source = OUT/where/'pairs.npz'
    if checksum(source) != json_at(signed)['predictions_sha256']:
        raise ValueError('Pair selection changed')
    with np.load(source) as z:
        rows, y, pairs = z['rows'], z['label'], z['pairs']
    values, checks, _, _, _ = f.inputs()
    assert np.array_equal(checks[rows,0], y)
    texts = [values[i] for i in rows]
    tokenizer = f.small.tokenizer()
    model = f.pipe(where, tokenizer).encoder
    start = time.perf_counter()
    x, intercept = collect(model, tokenizer, texts)
    if not torch.isfinite(x).all() or not torch.isfinite(intercept).all():
        raise ValueError('Compact representation contains nonfinite values')
    duration = time.perf_counter()-start
    YY = torch.as_tensor(y,dtype=torch.float32,device='cuda')
    PP = torch.from_numpy(pairs).cuda().long()
    balance = min(5., ((len(y)-y.sum())/y.sum())**.5)
    dst = DEST/where
    dst.mkdir(parents=True, exist_ok=True)
    logs = {}
    for name in MODE:
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(173)
        residual = head(model.width).cuda()
        optimizer = torch.optim.AdamW(residual.parameters(), lr=.0015, weight_decay=.025)
        ticks = EPOCHS*((len(x)+BATCH-1)//BATCH)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,ticks,eta_min=.00015)
        stats = []
        time0 = time.perf_counter()
        for epoch in range(EPOCHS):
            order = torch.randperm(len(x),generator=generator).split(BATCH)
            batches = torch.randint(len(PP),(len(order),128),generator=generator)
            sums = torch.zeros(3,device='cuda')
            residual.train()
            for step, idx in enumerate(order):
                p = PP[batches[step]]
                flat = torch.cat((idx.cuda(),p.reshape(-1)))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    correction = residual(x[flat]).float()[:,0]
                    z = intercept[flat]+correction
                    point = F.binary_cross_entropy_with_logits(z[:len(idx)],YY[flat[:len(idx)]],
                                                               pos_weight=YY.new_tensor(balance))
                    a = z[len(idx):].reshape(-1,2)
                    support = .5*(F.softplus(-a[:,0]).mean()+F.softplus(a[:,1]).mean())
                    pair_loss = F.softplus(.25+a[:,1]-a[:,0]).mean()
                    loss = point+.3*support+.025*correction.square().mean()
                    if name == 'pair':
                        loss = loss+.6*pair_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual.parameters(),.75)
                optimizer.step()
                schedule.step()
                sums += torch.stack((point.detach(),support.detach(),pair_loss.detach()))
            stats.append((sums/len(order)).tolist())
        path = dst/f'{name}.pt'
        torch.save(residual.cpu().state_dict(),path)
        logs[name] = dict(sha256=checksum(path),seconds=time.perf_counter()-time0,loss=stats)
        print(where,name,logs[name],flush=True)
    write_json(dst/'fit.json',dict(rows=len(y),positive=int(y.sum()),pairs=len(pairs),
                pair_source_sha256=checksum(source),model_width=model.width,representation_seconds=duration,
                modes=logs,code_sha256=checksum(Path(__file__))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    fit(p.parse_args().rotation)
