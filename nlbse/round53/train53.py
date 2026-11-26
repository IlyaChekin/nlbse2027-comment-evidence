"""Mask absent source labels throughout matched compact-encoder training."""
import argparse
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'round51'))
from train51 import (BATCH, COMPONENT_WEIGHTS, EPOCHS, KEEP, PARTS, ROOT, SEED, added,
                     batch, batch_plan, cache_base, checksum, encode, extra_head, f, full_context,
                     json_at, json_at_dict, late, lexical_features, prepare, release_start, write_json)
from support51 import DEST as SUPPORT

DEST = HERE / 'training'
ARMS = ('union', 'all_components', 'measured_components')


def fit_arm(rotation, arm, tok, seq, side, targets, valid, weight, anchor):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    net, _ = release_start(rotation, tok)
    late(net, arm)
    readout = extra_head(net.width).cuda()
    updated = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=readout.parameters(), lr=.0011),
                                  dict(params=updated, lr=.00018)], fused=True, weight_decay=.02)
    ids = list(range(len(seq)))
    total = EPOCHS * len(list(batch_plan(ids, seq, BATCH)))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.04 * total), total)
    auxiliary = targets.new_tensor(COMPONENT_WEIGHTS if arm != 'union' else (0., 0.))
    started = perf_counter()
    history = []
    step = 0
    for epoch in range(EPOCHS):
        rng.shuffle(ids)
        groups = list(batch_plan(ids, seq, BATCH))
        rng.shuffle(groups)
        sums = torch.zeros(4, device='cuda')
        for active in groups:
            step += 1
            token, mask = batch(active, seq, tok.pad_token_id)
            available = valid[active]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                legacy, pooled = encode(net, token, mask, side[active])
                logits = added(readout, legacy, pooled)
                error = F.binary_cross_entropy_with_logits(logits, targets[active],
                                                            pos_weight=weight, reduction='none')
                usual = error[:, 0].mean()
                source = (error[:, 1:] * available).sum(0) / available.sum(0).clamp_min(1.)
                drift = F.smooth_l1_loss(legacy.clamp(-16., 16.), anchor[active].float(),
                                         beta=.5, reduction='none')
                preserve = drift[:, [0, 2, 22, 23]].mean() + .18 * drift.mean()
                loss = usual + source @ auxiliary + KEEP * preserve
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*readout.parameters(), *updated], .85)
            optimizer.step()
            scheduler.step()
            sums += torch.cat((usual.detach()[None], source.detach(), preserve.detach()[None]))
            if step % 600 == 0:
                print(rotation, arm, epoch, step, total, round(perf_counter() - started, 2), flush=True)
        history.append((sums / len(groups)).tolist())
    assert step == total
    backbone = {k: v.detach().cpu() for k, v in net.named_parameters() if v.requires_grad}
    return dict(backbone=backbone, head=readout.cpu().state_dict()), dict(
        seconds=perf_counter() - started, steps=step, loss=history, backbone_tensors=len(backbone))


def main(rotation):
    ids, texts, target, matched, source = prepare(rotation)
    support = SUPPORT / 'receipt.json'
    mask_path = SUPPORT / 'mask.npz'
    assert json_at(support)['mask_sha256'] == checksum(mask_path)
    with np.load(mask_path) as z:
        assert np.array_equal(z['available'][ids], matched)
        assert np.array_equal(((z['component'][ids] & 6) > 0) & matched, target[:, 2].astype(bool))
        measured = z['external_available'][ids]
    assert np.all(measured <= matched)
    assert np.all(target[:, 2].astype(bool) <= measured)
    all_labels = np.column_stack((matched, matched))
    explicit = np.column_stack((matched, measured))
    tok = f.small.tokenizer()
    model, base = release_start(rotation, tok)
    folder = DEST / rotation
    folder.mkdir(exist_ok=True, parents=True)
    roster = folder / 'rows.npz'
    if roster.exists():
        with np.load(roster) as z:
            for name, value in dict(ids=ids, target=target, matched=matched, external_measured=measured).items():
                assert np.array_equal(z[name], value)
    else:
        np.savez_compressed(roster, ids=ids, target=target, matched=matched, external_measured=measured)
    # Use identical class weights in all arms, estimated where each label is defined.
    positive = target.sum(0)
    rates = np.array([len(ids), matched.sum(), measured.sum()])
    weights = np.sqrt((rates - positive) / np.maximum(1., positive)).clip(1., [2., 3., 12.])
    receipt = folder / 'fit.json'
    config = json_at_dict(dict(seed=SEED, epochs=EPOCHS, batch=BATCH, samples=len(ids), labels=rates.tolist(),
                 weights=weights.tolist(), positive=positive.astype(int).tolist(), anchor_weight=KEEP,
                 base_checkpoint_sha256=base, source_mask_sha256=checksum(mask_path),
                 roster_sha256=checksum(roster), code_sha256=checksum(Path(__file__)), sources=source))
    stamp = json_at(receipt) if receipt.exists() else dict(config=config, arms={})
    assert stamp['config'] == config
    todo = [arm for arm in ARMS if arm not in stamp['arms'] or not (folder / (arm + '.pt')).is_file() or
            checksum(folder / (arm + '.pt')) != stamp['arms'][arm]['sha256']]
    if not todo:
        return
    seq = full_context(texts, tok)
    side = torch.as_tensor(lexical_features(texts), device='cuda')
    late(model, 'frozen_union')
    anchor = cache_base(model, tok, seq, side)
    del model
    y = torch.as_tensor(target, device='cuda')
    weight = torch.as_tensor(weights, device='cuda', dtype=torch.float32)
    for arm in todo:
        validity = torch.as_tensor(explicit if arm == 'measured_components' else all_labels, device='cuda')
        state, profile = fit_arm(rotation, arm, tok, seq, side, y, validity, weight, anchor)
        path = folder / (arm + '.pt')
        partial = path.with_suffix('.partial')
        torch.save(state, partial)
        partial.replace(path)
        stamp['arms'][arm] = profile | dict(sha256=checksum(path))
        write_json(receipt, stamp)
        print(rotation, arm, stamp['arms'][arm], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation', choices=PARTS)
    main(p.parse_args().rotation)
