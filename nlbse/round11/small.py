"""Train a narrow code encoder with dilated convolutions and pooled global attention."""
import argparse
import json
from pathlib import Path
import random
from statistics import median
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round10"))
from students import (ROOT, batch, batch_plan, calibrated, dataset, get_fit, lexical_features,
                      metric, ordered_batches, save, sha256, write_json)
from score_runtime import competition_formula

WIDTH, CAP, BATCH = 128, 512, 128
OUT = HERE / "small"
DILATIONS = (1, 2, 4, 8)


def run_directory(width, continuation=None):
    name = "small" if width == WIDTH else f"small{width}"
    return HERE / (f"{name}_{continuation}" if continuation else name)


class LocalBlock(nn.Module):
    def __init__(self, dilation, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 5, dilation=dilation, padding=2*dilation, groups=width)
        self.mix = nn.Linear(width, 2*width)
        self.out = nn.Linear(width, width)
        self.use_shifted = False

    def shifted_conv(self, y):
        dilation = self.conv.dilation[0]
        inputs = y.half()
        weights = self.conv.weight[:, 0].half().float()
        bias = self.conv.bias.half().float()
        padded = F.pad(inputs, (0, 0, 2*dilation, 2*dilation))
        width = inputs.shape[1]
        mixed = torch.zeros_like(inputs, dtype=torch.float32)
        for tap in range(5):
            start = tap*dilation
            mixed.addcmul_(padded[:, start:start+width, :], weights[:, tap])
        return (mixed+bias).half()

    def forward(self, x, mask):
        y = self.norm(x) * mask
        if self.use_shifted:
            y = self.shifted_conv(y)
        else:
            y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        return x + self.out(F.glu(self.mix(y), dim=-1))


class Encoder(nn.Module):
    def __init__(self, vocab, width=WIDTH):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(vocab, width)
        self.local = nn.ModuleList(LocalBlock(d, width) for d in DILATIONS)
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2*width)
        self.global_out = nn.Linear(width, width)
        self.head = nn.Sequential(nn.LayerNorm(2*width+5), nn.Dropout(.12), nn.Linear(2*width+5, 2))

    def token_states(self, tokens, valid):
        mask = valid.unsqueeze(-1)
        x = self.embed(tokens)
        for layer in self.local:
            x = layer(x, mask)
        b, length, width = x.shape
        y = self.norm(x)
        summaries, count = self.attention_memory(y, valid)
        key, value = self.kv(summaries).chunk(2, dim=-1)
        q = self.query(y).view(b, length, 4, -1).transpose(1, 2)
        k = key.reshape(b, -1, 4, width//4).transpose(1, 2)
        v = value.reshape(b, -1, 4, width//4).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=count[:, None, None, :] > 0)
        return x + self.global_out(attention.transpose(1, 2).reshape(b, length, width))

    def attention_memory(self, y, valid):
        b, length, width = y.shape
        mask = valid.unsqueeze(-1)
        pad = (-length) % 4
        summaries = F.pad(y * mask, (0, 0, 0, pad)).view(b, -1, 4, width).sum(dim=2)
        count = F.pad(valid, (0, pad)).view(b, -1, 4).sum(-1)
        summaries = summaries / count.clamp_min(1).unsqueeze(-1)
        return summaries, count

    def readout(self, x, valid, side):
        mask = valid.unsqueeze(-1)
        average = (x * mask).sum(dim=1) / valid.sum(1, keepdim=True)
        maximum = x.masked_fill(~mask.bool(), -torch.inf).amax(dim=1)
        return self.head(torch.cat((average, maximum, side), dim=-1))

    def forward(self, tokens, valid, side):
        return self.readout(self.token_states(tokens, valid), valid, side)


def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / "models/codet5_small", local_files_only=True)


def encode(texts, tok):
    result = []
    for i in range(0, len(texts), 1024):
        # Reserve the closing quarter of long functions while keeping special tokens.
        data = tok(texts[i:i+1024], truncation=False, verbose=False)["input_ids"]
        result.extend(s if len(s) <= CAP else s[:384] + s[-128:] for s in data)
    return result


def operations(n, length, width):
    pool = (length+3)//4
    local = len(DILATIONS)*2*length*(5*width+3*width**2)
    attention = 4*(length+pool)*width**2 + 4*length*pool*width
    return n*(local+attention+4*(2*width+5))/1e9


def fit(epochs, width, continuation=None):
    out = run_directory(width, continuation)
    torch.manual_seed(42)
    rng = random.Random(42)
    tok = tokenizer()
    model = Encoder(len(tok), width).cuda()
    if continuation:
        previous = run_directory(width)
        selection = json.loads((previous/"selected.json").read_text())
        initial = previous/selection["checkpoint"]
        if sha256(initial) != selection["sha256"]:
            raise ValueError("Unexpected base encoder for continuation")
        model.load_state_dict(torch.load(initial, weights_only=True, map_location="cpu"))
    else:
        recipe = json.loads((ROOT / "round10/runs/t5_small/selected.json").read_text())
        initial = ROOT / "round10/runs/t5_small" / recipe["checkpoint"]
        if sha256(initial) != recipe["settings"]["sha256"]:
            raise ValueError("Unexpected CodeT5-small initial weights")
        weights = torch.load(initial, weights_only=True, map_location="cpu")["encoder.shared.weight"]
        with torch.no_grad():
            seed = weights[:len(tok)].cuda()
            seed = seed - seed.mean(0)
            _, _, basis = torch.pca_lowrank(seed, q=width+16, niter=3, center=False)
            projected = seed @ basis[:, :width]
            model.embed.weight.copy_(projected / projected.std(dim=0).clamp_min(.005))
    texts, y, teacher = get_fit()
    sequences = encode(texts, tok)
    side = torch.as_tensor(lexical_features(texts), device="cuda")
    gold = torch.as_tensor(y, device="cuda", dtype=torch.float32)
    soft = torch.sigmoid(torch.logit(torch.as_tensor(teacher, device="cuda").clamp(1e-5,1-1e-5))/1.7)
    pos = torch.as_tensor(np.clip((len(y)-y.sum(0))/y.sum(0),1,5), dtype=torch.float32, device="cuda")
    rates = (.00016,.00035) if continuation else (.0007,.0012)
    mismatch = (soft[:,0] > .5) != (gold[:,0] > .5)
    opt = torch.optim.AdamW([{"params": model.embed.parameters(), "lr": rates[0]},
                             {"params": [p for name, p in model.named_parameters() if not name.startswith("embed.")], "lr": rates[1]}],
                            weight_decay=.025, fused=True)
    order = list(range(len(y)))
    steps = len(list(batch_plan(order, sequences, BATCH)))*epochs
    schedule = get_linear_schedule_with_warmup(opt, int(.04*steps), steps)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", dict(width=width, cap=CAP, batch=BATCH, dilations=DILATIONS, fit=len(y),
              epochs=epochs, seed=42, initial_sha256=sha256(initial), teacher_weight=.30, temperature=1.7,
              continuation=continuation, rates=rates, fit_disagreements=int(mismatch.sum().item())))
    step = 0
    began = time.perf_counter()
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        groups = list(batch_plan(order, sequences, BATCH))
        rng.shuffle(groups)
        model.train()
        for rows in groups:
            x, mask = batch(rows, sequences, tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, mask, side[rows])
                hard = F.binary_cross_entropy_with_logits(logits, gold[rows], pos_weight=pos, reduction="none")
                distill = F.binary_cross_entropy_with_logits(logits/1.7, soft[rows], reduction="none")
                if continuation == "gold_disagreement":
                    mismatch_rows = mismatch[rows]
                    # Keep both objectives for agreement rows and emphasize gold when they conflict.
                    loss_v = torch.where(mismatch_rows, 1.25*hard[:,0], .70*hard[:,0]+.30*1.7**2*distill[:,0])
                    loss = (loss_v.mean()+(.70*hard[:,1]+.30*1.7**2*distill[:,1]).mean())/2
                else:
                    loss = (.70*hard + .30*1.7**2*distill).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            schedule.step()
            step += 1
            if step % 500 == 0:
                print(step, steps, float(loss.detach()), round(time.perf_counter()-began,1), flush=True)
        temporary = out / f"epoch{epoch}.part"
        torch.save(model.state_dict(), temporary)
        temporary.replace(temporary.with_suffix(".pt"))
        print("epoch", epoch, flush=True)


@torch.inference_mode()
def scores(model, tok, texts):
    p = np.empty((len(texts), 2), dtype=np.float32)
    cost = np.empty(len(texts), dtype=np.float64)
    forward = 0.
    model.eval()
    start = time.perf_counter()

    def prepare(i):
        part = texts[i:i+2048]
        return encode(part, tok), lexical_features(part)

    with torch.autocast("cuda", dtype=torch.float16):
        for offset, (seq, side) in ordered_batches(range(0, len(texts), 2048), prepare, True):
            side = torch.as_tensor(side, device="cuda")
            for rows in batch_plan(list(range(len(seq))), seq, BATCH):
                x, mask = batch(rows, seq, tok.pad_token_id)
                ix = offset + np.asarray(rows)
                t = time.perf_counter()
                p[ix] = model(x, mask, side[rows]).float().sigmoid().cpu().numpy()
                forward += time.perf_counter() - t
                cost[ix] = operations(1, x.shape[1], model.width)
    return p, cost, dict(wall_seconds=time.perf_counter()-start, forward_seconds=forward,
                         analytic_gflops=float(cost.sum()))


def evaluate(public=False, width=WIDTH, continuation=None):
    out = run_directory(width, continuation)
    tok = tokenizer()
    model = Encoder(len(tok), width).cuda()
    _, _, texts, y, ids, cal = dataset(full=True, public=public)
    if public:
        candidate = json.loads((out / "selected.json").read_text())
        model.load_state_dict(torch.load(out/candidate["checkpoint"], weights_only=True, map_location="cpu"))
        scores(model, tok, texts[:2048])
        trials = []
        for _ in range(3):
            p, costs, profile = scores(model, tok, texts)
            trials.append(profile)
        cost = {k: median(r[k] for r in trials) for k in trials[0]}
        np.savez_compressed(out/"public.npz", ids=ids, y=y, p=p, gflops=costs)
        save(out/"public", y, ids, cal, p, cost, candidate["thresholds"])
        metrics = metric(y, p > candidate["thresholds"])
        write_json(out/"public/analysis.json", dict(metrics=metrics, cost=cost, repeats=trials,
                   score=competition_formula(metrics["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])))
        return
    with np.load(ROOT / "round8/frozen/holdout.npz") as splits:
        if ids != splits["rows"].tolist():
            raise ValueError("Holdout order changed")
        folds = splits["folds"]
    cal, review = folds == 3, folds == 4
    all_runs = []
    for checkpoint in sorted(out.glob("epoch*.pt")):
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        p, costs, profile = scores(model, tok, texts)
        t = calibrated(y[cal], p[cal])
        record = dict(checkpoint=checkpoint.name, sha256=sha256(checkpoint), thresholds=t,
                      calibration=metric(y[cal], p[cal]>t), review=metric(y[review], p[review]>t), profile=profile)
        np.savez_compressed(out / f"{checkpoint.stem}_holdout.npz", ids=ids, y=y, p=p, gflops=costs, folds=folds)
        all_runs.append(record)
        print(checkpoint.name, record["review"], flush=True)
    selected = max(all_runs,key=lambda row: row["calibration"]["vulnerability"]["f1"])
    write_json(out/"selected.json", dict(selected, candidates=all_runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "tune", "public"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--width", type=int, choices=(128, 256), default=WIDTH)
    parser.add_argument("--continuation", choices=("matched", "gold_disagreement"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.epochs,args.width,args.continuation)
    else:
        evaluate(public=args.stage == "public", width=args.width, continuation=args.continuation)
