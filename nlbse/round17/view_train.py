"""Matched context and identifier-view continuation for one compact encoder."""

import argparse
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "round16"))
from experiment import batch, batch_plan, checkpoint, checksum, fit_data, inputs, json_at, lexical_features, load_original, small
from context import tokenize
from normalize_code import parallel
from pipeline import select_indices, write_json


MODES = ("crop", "full", "paired", "consistent")
REPR_MODES = ("full", "latent", "relational")
SEEDS = (42, 101)


def cohort(scope, labels, teacher):
    path = ROOT / "round15/mining/scores.npz"
    sha = checksum(path)
    if sha != json_at(path.parent / "manifest.json")["sha256"]:
        raise ValueError("Out-of-fold scores changed")
    if scope == "projects":
        rows = inputs()[4]["fit"]
        with np.load(path) as f:
            if not np.array_equal(f["ids"], rows) or not np.array_equal(f["y"], labels):
                raise ValueError("Training examples differ from the out-of-fold scores")
            p = f["compact"][:, 0]
    else:
        rows, _ = select_indices(False)
        p = teacher[:, 0].copy()
        with np.load(path) as f:
            _, local, cached = np.intersect1d(rows, f["ids"], return_indices=True)
            if not np.array_equal(labels[local], f["y"][cached]):
                raise ValueError("Labels changed in the official overlap")
            p[local] = f["compact"][cached, 0]
    return (p >= .35) & (p <= .85), sha


def crop(sequence):
    return [v if len(v) <= 512 else v[:384] + v[-128:] for v in sequence]


def bce_and_distill(logits, gold, soft, balance):
    ce = F.binary_cross_entropy_with_logits(logits, gold, pos_weight=balance)
    kd = F.binary_cross_entropy_with_logits(logits / 1.7, soft)
    return .7 * ce + .3 * 1.7 ** 2 * kd


def agreement(a, b):
    a, b = a.float() / 1.5, b.float() / 1.5
    target = .5 * (a.detach().sigmoid() + b.detach().sigmoid())
    return .5 * (F.binary_cross_entropy_with_logits(a, target) + F.binary_cross_entropy_with_logits(b, target))


def fit(scope, seed, modes, representation=False, output_root=None, override=None):
    out = (output_root or (HERE/"representation" if representation else HERE/scope))/str(seed)
    out.mkdir(parents=True, exist_ok=True)
    texts, labels, teacher, teacher_sha = fit_data(scope)
    eligible, selection_sha = cohort(scope, labels, teacher)
    tok = small.tokenizer()
    # Finish CPU process work before creating a CUDA context for the models.
    normalized = parallel(texts)
    a, b = tokenize(texts, tok), tokenize(normalized, tok)
    eligible &= np.fromiter((x != y for x, y in zip(a, b, strict=True)), dtype=bool, count=len(a))
    del normalized
    rng = random.Random(seed)
    swap = [rng.random() < .45 for _ in texts]
    raw = [y if changed else x for x, y, changed in zip(a, b, swap, strict=True)]
    opposite = [x if changed else y for x, y, changed in zip(a, b, swap, strict=True)]
    cropped = crop(raw)
    order = list(range(len(raw)))
    rng.shuffle(order)
    batches = list(batch_plan(order, raw, small.BATCH))
    rng.shuffle(batches)
    source = checkpoint(scope)
    side = torch.as_tensor(lexical_features(texts), device="cuda")
    gold = torch.as_tensor(labels, device="cuda", dtype=torch.float32)
    balance = torch.as_tensor(np.clip((len(labels)-labels.sum(0))/labels.sum(0), 1, 5), device="cuda", dtype=torch.float32)
    soft = (torch.as_tensor(teacher, device="cuda").clamp(1e-5, 1-1e-5).logit()/1.7).sigmoid()
    special,teacher_override_meta = ({},None) if override is None else override(labels,teacher)
    digest = None
    if representation:
        folder = HERE/"representation"
        source_repr = folder/"teacher.npz"
        digest = checksum(source_repr)
        if digest != json_at(folder/"teacher.json")["sha256"]:
            raise ValueError("Compressed teacher representation changed")
        with np.load(source_repr) as data:
            if not np.array_equal(data["ids"], inputs()[4]["fit"]) or not np.array_equal(data["y"], labels):
                raise ValueError("Teacher representation rows do not match the fit")
            latent = F.normalize(torch.as_tensor(data["features"], device="cuda", dtype=torch.float32),dim=1)
    result = {}
    for mode in modes:
        torch.manual_seed(seed)
        model = load_original(tok, "control", source)
        model.train()
        target = soft if mode not in special else (torch.as_tensor(special[mode],device="cuda").clamp(1e-5,1-1e-5).logit()/1.7).sigmoid()
        groups = [
            {"params": model.embed.parameters(), "lr": .00012},
            {"params": [v for k, v in model.named_parameters() if not k.startswith("embed.")], "lr": .00025},
        ]
        probe = None
        if mode in REPR_MODES[1:]:
            probe = torch.nn.Sequential(torch.nn.LayerNorm(model.width),torch.nn.Linear(model.width,latent.shape[1])).cuda()
            groups.append({"params":probe.parameters(),"lr":.0008})
        opt = torch.optim.AdamW(groups, weight_decay=.025, fused=True)
        schedule = get_linear_schedule_with_warmup(opt, int(.04*len(batches)), len(batches))
        torch.manual_seed(seed)
        started = time.perf_counter()
        num_pairs = 0
        for step, ids in enumerate(batches, 1):
            x, mask = batch(ids, cropped if mode == "crop" else raw, tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if probe is None:
                    current = model(x, mask, side[ids])
                else:
                    states=model.token_states(x,mask)
                    current=model.readout(states,mask,side[ids])
                    pooled=(states.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                    v=F.normalize(probe(pooled).float(),dim=1)
                loss = bce_and_distill(current, gold[ids], target[ids], balance)
                if probe is not None:
                    target=latent[ids]
                    direct=1-(v*target).sum(1).mean()
                    if mode=="relational":
                        direct=.25*direct+F.mse_loss(v@v.T,target@target.T)
                    loss=loss+.12*direct
                if mode in {"paired", "consistent"}:
                    positions = [i for i, row in enumerate(ids) if eligible[row]]
                    paired = [ids[i] for i in positions]
                    if paired:
                        alt, mask_alt = batch(paired, opposite, tok.pad_token_id)
                        other = model(alt, mask_alt, side[paired])
                        loss = loss + .35*len(paired)/len(ids)*bce_and_distill(other, gold[paired], target[paired], balance)
                        if mode == "consistent":
                            loss = loss + .35*len(paired)/len(ids)*agreement(current[positions, 0], other[:, 0])
                        num_pairs += len(paired)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            schedule.step()
            if step % 400 == 0:
                print(scope, seed, mode, step, len(batches), round(time.perf_counter()-started, 1), flush=True)
        path = out/f"{mode}.pt"
        temp = path.with_suffix(".part")
        torch.save(model.state_dict(), temp)
        temp.replace(path)
        result[mode] = dict(path=path.name, sha256=checksum(path), pairs=num_pairs, steps=len(batches),
                            seconds=time.perf_counter()-started)
        write_json(out/"fit.json", dict(scope=scope, seed=seed, rows=len(labels), kept_long=sum(len(v)>512 for v in raw),
                   pairable=int(eligible.sum()), teacher_sha256=teacher_sha, selection_sha256=selection_sha,
                   initial_sha256=source[1], representation_sha256=digest,teacher_override=teacher_override_meta,modes=result))
        del model, opt, schedule, probe


def open_view(scope, seed, mode, tok, experiment=None):
    if mode == "reference":
        source = checkpoint(scope)
    else:
        folder = (experiment or HERE/scope)/str(seed)
        cfg = json_at(folder/"fit.json")["modes"][mode]
        source = folder/cfg["path"], cfg["sha256"]
    model = load_original(tok, "control", source)
    for layer in model.local:
        layer.use_shifted = True
    model.eval()
    return SimpleNamespace(encoder=model, tok=tok, mode="control")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scope", choices=("projects", "official"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    ap.add_argument("--representation", action="store_true")
    args = ap.parse_args()
    if args.representation and args.scope!="projects":
        ap.error("The representation cache was built on project-disjoint training rows")
    fit(args.scope, args.seed, REPR_MODES if args.representation else args.modes,args.representation)
