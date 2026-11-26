"""Matched continuation training with fixed and variable input budgets."""
from collections import Counter
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from common import HERE, PORTABLE, SEED, TRAIN, load, load_checkpoint, prepare, select_indices, stratified, export, write_json
from neural import batch_plan
from windows import pack

OUT = HERE / "training"
BATCH = 32
CAPS = (256, 384, 512)
WEIGHTS = (0.2, 0.6, 0.2)
LEARNING_RATE = 1e-5
ARMS = ("fixed384", "mixed")


def select_rows(texts, y):
    fit, _ = select_indices(pilot=False)
    long = [i for i in fit if len(texts[i]) >= 950]
    short = [i for i in fit if len(texts[i]) < 950]
    return np.sort(np.concatenate((stratified(long, y, 14000), stratified(short, y, 26000))))


def train(model, data, y, arm):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    order = list(range(len(y)))
    rng.shuffle(order)
    plan = list(batch_plan(order, data[384].views["head"], BATCH))
    rng.shuffle(plan)
    options = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, fused=True)
    scheduler = get_linear_schedule_with_warmup(options, int(0.06 * len(plan)), len(plan))
    positives = y.sum(axis=0)
    weight = torch.as_tensor(np.clip((len(y)-positives)/positives, 1, 5),
                             dtype=torch.float32, device="cuda")
    gold = torch.as_tensor(y, dtype=torch.float32, device="cuda")
    losses = torch.zeros((), device="cuda")
    used = Counter()
    extra = 0
    padded_tokens = 0
    model.train()
    started = time.perf_counter()
    for step, rows in enumerate(plan, 1):
        cap = 384 if arm == "fixed384" else rng.choices(CAPS, weights=WEIGHTS)[0]
        used[cap] += len(rows)
        inputs = data[cap]
        first = pack(inputs.views["head"], rows, inputs.pad, "cuda")
        extra_positions = [position for position, row in enumerate(rows) if inputs.omitted[row]]
        extra_rows = [rows[position] for position in extra_positions]
        second = pack(inputs.views["cover64"], extra_rows, inputs.pad, "cuda") if extra_rows else None
        targets = gold[rows]
        options.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            head = model(*first).float()
            joint = head
            if second is not None:
                tail = model(*second).float()
                mat = head[:,1].clone()
                mat[extra_positions] = torch.maximum(mat[extra_positions], tail[:,1])
                joint = torch.stack((head[:,0],mat), dim=1)
            loss = (F.binary_cross_entropy_with_logits(joint, targets, pos_weight=weight)
                    + 0.25*F.binary_cross_entropy_with_logits(head, targets, pos_weight=weight))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        options.step()
        scheduler.step()
        losses += loss.detach()
        extra += len(extra_rows)
        padded_tokens += first[0].numel() + (second[0].numel() if second is not None else 0)
        if step % 200 == 0 or step == len(plan):
            print(arm, step, "of", len(plan), "loss", round(losses.item()/step, 6),
                  "seconds", round(time.perf_counter()-started, 2), flush=True)
    return dict(seconds=time.perf_counter()-started, steps=len(plan), rows_at_each_cap=dict(used),
                extra_windows=extra, padded_tokens=padded_tokens, positive_weights=weight.cpu().tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    texts, labels, _ = load(TRAIN)
    rows = select_rows(texts, labels)
    np.save(OUT/"fit_rows.npy", rows)
    sample = [texts[i] for i in rows]
    y = labels[rows]
    del texts, labels
    _, tokenizer = load_checkpoint(PORTABLE, "cpu")
    started = time.perf_counter()
    data = {cap: prepare(sample, tokenizer, ("head", "cover64"), head_limit=cap) for cap in CAPS}
    print("Prepared", len(rows), "matched fit rows", round(time.perf_counter()-started, 2), flush=True)
    for arm in ARMS:
        model, _ = load_checkpoint(PORTABLE, "cuda")
        config = train(model, data, y, arm)
        checksum = export(model, tokenizer, OUT/arm)
        write_json(OUT/arm/"training.json", dict(config, model_sha256=checksum, fit_rows=len(rows),
                                               learning_rate=LEARNING_RATE, seed=SEED, batch=BATCH, label_positives=y.sum(axis=0).tolist()))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
