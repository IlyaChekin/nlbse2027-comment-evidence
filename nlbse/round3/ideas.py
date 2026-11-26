"""Packed, two-label GPU pilots with four fixed round-3 variants."""
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from protocol import OUT, SIX, TWELVE, dataset, length_thresholds, save, threshold
from pipeline import metric, seed_all, write_json
from comment_spans import LEXEME, comment_bits

DEVICE = "cuda"
BATCH = 32
WINDOW = 2048
CAP = 512



@dataclass
class Encodings:
    head: list
    tail: list
    comment: list
    lengths: np.ndarray
    pad: int


def encode(texts, tokenizer):
    head, tail, comment, lengths = [], [], [], []
    for start in range(0, len(texts), 1024):
        chunk = texts[start:start+1024]
        enc = tokenizer(chunk, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        for text, tokens, offsets in zip(chunk, enc["input_ids"], enc["offset_mapping"], strict=True):
            keep = CAP - 2
            head.append([tokenizer.bos_token_id, *tokens[:keep], tokenizer.eos_token_id])
            tail.append([tokenizer.bos_token_id, *tokens[-keep:], tokenizer.eos_token_id] if len(tokens) > keep else [])
            comment.append([False, *comment_bits(text, offsets[:keep]), False])
            lengths.append(len(tokens) + 2)
    check = tokenizer(texts[:9], truncation=True, max_length=CAP)["input_ids"]
    assert head[:9] == check
    return Encodings(head, tail, comment, np.array(lengths), tokenizer.pad_token_id)


def batch_plan(data, ids):
    for start in range(0, len(ids), WINDOW):
        window = sorted(ids[start:start+WINDOW], key=lambda i: len(data.head[i]))
        yield [window[begin:begin+BATCH] for begin in range(0, len(window), BATCH)]


def batch(data, ids, tail=False, roles=False):
    sequences = data.tail if tail else data.head
    width = max(len(sequences[i]) for i in ids)
    values = np.full((len(ids), width), data.pad, dtype=np.int64)
    keep = np.zeros_like(values)
    comments = np.zeros_like(values, dtype=bool) if roles else None
    for j, i in enumerate(ids):
        tokens = sequences[i]
        values[j, :len(tokens)] = tokens
        keep[j, :len(tokens)] = 1
        if roles:
            comments[j, :len(tokens)] = data.comment[i]
    payload = {k: torch.from_numpy(v).to(DEVICE) for k, v in (("input_ids", values), ("attention_mask", keep))}
    return payload, torch.from_numpy(comments).to(DEVICE) if roles else None


def load_model(path):
    return RobertaForSequenceClassification.from_pretrained(
        path, local_files_only=True, attn_implementation="eager").eval().to(DEVICE)


def probabilities(logits):
    return logits.float().sigmoid()


def wide(model, data, ids):
    inputs, _ = batch(data, ids)
    first = probabilities(model(**inputs).logits)
    tail_rows = [j for j, i in enumerate(ids) if data.tail[i]]
    last = first.clone()
    if tail_rows:
        other, _ = batch(data, [ids[j] for j in tail_rows], tail=True)
        last[tail_rows] = probabilities(model(**other).logits)
    return first, last


def split_features(model, data, ids):
    inputs, comments = batch(data, ids, roles=True)
    states = model.roberta(**inputs).last_hidden_state
    valid = inputs["attention_mask"].bool()
    valid[:, 0] = False
    valid.scatter_(1, valid.sum(1, keepdim=True), False)
    cls = states[:, 0]
    vectors = []
    for mask in (valid & ~comments, valid & comments):
        weights = mask[:, :, None]
        pooled = (states * weights).sum(1) / weights.sum(1).clamp_min(1)
        vectors.append(torch.cat((cls, pooled), dim=-1))
    return torch.stack(vectors, dim=1), model.classifier(states)


class SplitHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(),
                          nn.Dropout(.1), nn.Linear(128, 1)) for _ in range(2)
        ])
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, vectors):
        return torch.cat([branch(vectors[:, j]) for j, branch in enumerate(self.branches)], dim=-1)


def blocks(model, hidden, mask, start, stop):
    extended = model.roberta.get_extended_attention_mask(mask, mask.shape)
    for layer in islice(model.roberta.encoder.layer, start, stop):
        hidden = layer(hidden, attention_mask=extended)[0]
    return hidden


def prefix(model, data, ids):
    inputs, _ = batch(data, ids)
    mask = inputs["attention_mask"]
    state = model.roberta.embeddings(input_ids=inputs["input_ids"])
    return blocks(model, state, mask, 0, 6), mask


def exit_features(model, data, ids):
    front, mask = prefix(model, data, ids)
    final = model.classifier(blocks(model, front, mask, 6, 12))
    return front[:, 0], final


def make_early_head():
    return nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                         nn.Dropout(.1), nn.Linear(256, 2))


def confidence(p, thresholds):
    return torch.where(p >= thresholds, (p-thresholds)/(1-thresholds), (thresholds-p)/thresholds).amin(dim=1)


def routed(model, head, data, ids, early_th, final_th, cutoff):
    front, mask = prefix(model, data, ids)
    fast = probabilities(head(front[:, 0]))
    active = torch.where(confidence(fast, early_th) < cutoff)[0]
    used = fast.clone()
    bits = fast > early_th
    if active.numel():
        longer = mask[active]
        width = int(longer.sum(1).max())
        slow = probabilities(model.classifier(blocks(model, front[active, :width], longer[:, :width], 6, 12)))
        used[active], bits[active] = slow, slow > final_th
    return used, bits, len(ids) - active.numel()


@torch.no_grad()
def collect(data, ids, compute):
    results = None
    pos = {i: j for j, i in enumerate(ids)}
    for batches in batch_plan(data, ids):
        for chunk in batches:
            where = [pos[i] for i in chunk]
            with torch.autocast(DEVICE, dtype=torch.float16):
                output = compute(chunk)
            if results is None:
                results = [np.empty((len(ids), *x.shape[1:]), dtype=np.float32) for x in output]
            for values, x in zip(results, output, strict=True):
                values[where] = x.float().cpu().numpy()
    return results


def train_head(name, module, features, old, labels, residual):
    start = time.perf_counter()
    module.to(DEVICE).train()
    x, base, gold = [torch.as_tensor(a, device=DEVICE) for a in (features, old, labels)]
    positive = torch.tensor([2.05, 5.], device=DEVICE)
    objective = nn.BCEWithLogitsLoss(pos_weight=positive)
    optimizer = torch.optim.AdamW(module.parameters(), lr=5e-4, weight_decay=.02)
    total = 6
    for _ in range(total):
        for ids in torch.randperm(len(gold), device=DEVICE).split(256):
            logits = module(x[ids])
            if residual:
                logits = logits + base[ids]
            supervised = objective(logits, gold[ids].float())
            soft = nn.functional.binary_cross_entropy_with_logits(logits / 2, (base[ids] / 2).sigmoid())
            loss = .5 * supervised + 2 * soft
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), out / "trained_head.pt")
    write_json(out / "fit.json", dict(rows=len(gold), epochs=total, wall_seconds=time.perf_counter()-start))
    return module.eval()


@torch.inference_mode()
def measure(name, data, ids, compute, model_parameters, *, profile_operations=True):
    positions = {i: j for j, i in enumerate(ids)}
    answers, bits = np.empty((len(ids),2), dtype=np.float32), np.empty((len(ids),2), dtype=bool)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    gflops, early = 0., 0
    for group, batches in enumerate(batch_plan(data, ids)):
        context = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True)
                   if profile_operations else nullcontext())
        with context as profiler:
            for chunk in batches:
                with torch.autocast(DEVICE, dtype=torch.float16):
                    p, b, exited = compute(chunk)
                where = [positions[i] for i in chunk]
                answers[where], bits[where] = p.cpu().numpy(), b.cpu().numpy()
                early += exited
            torch.cuda.synchronize()
        if profiler is not None:
            gflops += sum(x.flops for x in profiler.key_averages()) / 1e9
        if group % 3 == 0:
            print(f"{name} windows={group+1}", flush=True)
    return answers, bits, dict(rows=len(ids), wall_seconds=time.perf_counter()-start,
                              profiler_counted_gflops=gflops if profile_operations else None,
                              early_rows=early, parameters=model_parameters,
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def compare_exit(y, early, full, early_th, final_th,
                 cutoffs=(0.35, 0.5, 0.65, 0.75, 0.85, 1.01)):
    local = confidence(torch.tensor(early), torch.tensor(early_th)).numpy()
    reference = metric(y, full > final_th)["mean"]["f1"]
    checks = []
    for cutoff in cutoffs:
        exits = local >= cutoff
        bits = np.where(exits[:,None], early > early_th, full > final_th)
        f1 = metric(y, bits)["mean"]["f1"]
        checks.append(dict(cutoff=cutoff, rate=float(exits.mean()), f1=f1))
    best = max((r for r in checks if r["f1"] >= reference - .005), key=lambda r:r["rate"])
    return best, checks


def select_wide(y, start, end, eligible):
    blend = np.zeros(2)
    for column in range(2):
        best_score = -1
        for fraction in (.5, .25, .75):
            mixed = start.copy()
            mixed[eligible, column] = ((1-fraction)*start[eligible,column] + fraction*end[eligible,column])
            t = threshold(y, mixed)
            current = list(metric(y, mixed > t).values())[column]["f1"]
            if current > best_score:
                best_score, blend[column] = current, fraction
    return blend, threshold(y, start + (end-start)*blend)


def main():
    seed_all(42)
    texts, labels, row_ids, nfit, mask = dataset()
    train = list(range(nfit))
    cal = (np.flatnonzero(mask) + nfit).tolist()
    test = (np.flatnonzero(~mask) + nfit).tolist()
    tokenizer = RobertaTokenizerFast.from_pretrained(SIX, local_files_only=True)
    data = encode(texts, tokenizer)
    print(f"encoded fit={len(train)} calibration={len(cal)} selection={len(test)}", flush=True)
    y = labels[test]
    params = lambda model: sum(p.numel() for p in model.parameters())
    to_gpu = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    six = load_model(SIX)
    twelve = load_model(TWELVE)

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.float16):
        ids = train[:8]
        inputs, _ = batch(data, ids)
        _, manual = exit_features(twelve, data, ids)
        expected = twelve(**inputs).logits
        torch.testing.assert_close(manual, expected, atol=.001, rtol=.001)
    print("Layer-by-layer inference matches the native twelve-block forward.", flush=True)

    print("Fitting role-specific residual heads.", flush=True)
    vectors, six_logits = collect(data, train, lambda ids: split_features(six, data, ids))
    split_model = train_head("split_roles", SplitHead(vectors.shape[-1]), vectors, six_logits, labels[train], True)
    del vectors, six_logits
    print("Fitting the early-exit head.", flush=True)
    early_vectors, teacher = collect(data, train, lambda ids: exit_features(twelve, data, ids))
    early = train_head("early_exit", make_early_head(),
                       early_vectors, teacher, labels[train], False)
    del early_vectors, teacher
    print("Calibrating on reserved training rows.", flush=True)
    def plain(model, ids):
        x, _ = batch(data, ids)
        return probabilities(model(**x).logits)

    def split_prob(ids):
        feat, base = split_features(six, data, ids)
        return probabilities(base + split_model(feat.to(base.dtype)))

    def early_and_full(ids):
        feat, last = exit_features(twelve, data, ids)
        return probabilities(early(feat)), probabilities(last)

    six_cal, = collect(data, cal, lambda ids: (plain(six, ids),))
    simple_th = threshold(labels[cal], six_cal)
    length_th, edges = length_thresholds(labels[cal], six_cal, data.lengths[cal])
    split_cal, = collect(data, cal, lambda ids: (split_prob(ids),))
    split_th = threshold(labels[cal], split_cal)
    ecal, full_cal = collect(data, cal, early_and_full)
    early_th = threshold(labels[cal], ecal)
    full_th = threshold(labels[cal], full_cal)
    chosen, gates = compare_exit(labels[cal], ecal, full_cal, early_th, full_th)
    wide_first, wide_last = collect(data, cal, lambda ids: wide(twelve, data, ids))
    blend, wide_th = select_wide(labels[cal], wide_first, wide_last, data.lengths[cal] > CAP)
    calibration = dict(calibration_rows=len(cal), selection_rows=len(test), fit_rows=len(train),
                       early_exit=chosen, gate_search=gates, expensive_tail_fraction=blend.tolist(),
                       overlength_selection=int((data.lengths[test] > CAP).sum()),
                       functions_with_first_window_comments=sum(any(data.comment[i]) for i in test))
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "calibration.json", calibration)
    print(json.dumps(calibration), flush=True)

    def classify(fn, th):
        def predict(ids):
            p = fn(ids)
            return p, p > to_gpu(th), 0
        return predict

    def wide_prob(ids):
        a,b = wide(twelve, data, ids)
        return a + (b-a)*to_gpu(blend)

    def evaluate(name, fn, th, count, config=None):
        pred, bits, meta = measure(name, data, test, fn, count)
        save(name, row_ids[test], y, pred, bits, meta, {**(config or {}), "thresholds": th.tolist()})
        return pred, bits, meta

    p, _, cost = evaluate("six_control", classify(lambda ids: plain(six, ids), simple_th), simple_th, params(six))
    roles = np.digitize(data.lengths[test], edges, right=True)
    save("length_thresholds", row_ids[test], y, p, p > length_th[roles],
         {**cost, "cost_note": "same six-block forward, thresholds reuse token length"},
         dict(edges=edges, thresholds=length_th.tolist()))
    full_p, full_bits, _ = evaluate("twelve_control", classify(lambda ids: plain(twelve, ids), full_th), full_th, params(twelve))
    wide_p, wide_bits, _ = evaluate("extra_context", classify(wide_prob, wide_th), wide_th, params(twelve),
                                   dict(tail_fraction=blend.tolist(), windows="first and last 512 tokens for overlength input"))
    evaluate("split_roles", classify(split_prob, split_th), split_th, params(six)+params(split_model))
    evaluate("early_exit", lambda ids: routed(twelve, early, data, ids, to_gpu(early_th), to_gpu(full_th), chosen["cutoff"]),
             full_th, params(twelve)+params(early), dict(early_thresholds=early_th.tolist(), cutoff=chosen["cutoff"]))
    long = data.lengths[test] > CAP
    write_json(OUT / "long_context_diagnostic.json", dict(rows=int(long.sum()), positives=y[long].sum(0).tolist(),
               twelve=metric(y[long],full_bits[long]), extra_context=metric(y[long],wide_bits[long])))


if __name__ == "__main__":
    main()
