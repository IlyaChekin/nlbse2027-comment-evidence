"""Matched small-data pilots for a byte CNN and pretrained code encoders."""
import argparse
from dataclasses import asdict, dataclass
import json
import random
import time

import numpy as np
import torch
from torch import nn
from transformers import (AutoTokenizer, RobertaForSequenceClassification, T5EncoderModel,
                          get_linear_schedule_with_warmup)

from protocol import HERE, ROOT, dataset, save, write_json

BLOCK = 2048


@dataclass(frozen=True)
class Config:
    family: str
    cap: int
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int = 42


class ByteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 24, padding_idx=0)
        self.conv = nn.ModuleList((nn.Conv1d(24, 48, 5, stride=4, padding=2),
                                   nn.Conv1d(48, 64, 5, stride=2, padding=2),
                                   nn.Conv1d(64, 96, 5, stride=2, padding=2)))
        self.head = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 2))

    def forward(self, tokens, mask):
        hidden = self.embedding(tokens).transpose(1, 2)
        length = mask.sum(dim=1, keepdim=True)
        for layer in self.conv:
            hidden = nn.functional.gelu(layer(hidden))
            stride = layer.stride[0]
            length = (length + stride - 1) // stride
            active = torch.arange(hidden.shape[-1], device=tokens.device) < length
            hidden = hidden * active[:, None, :]
        mean = hidden.sum(dim=-1) / length
        maximum = hidden.masked_fill(~active[:, None, :], float("-inf")).amax(dim=-1)
        return self.head(torch.cat((mean, maximum), dim=-1))


class T5Joint(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.encoder = architecture
        width = self.encoder.config.d_model
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.1), nn.Linear(width, 2))

    def forward(self, tokens, mask):
        encoded = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.head(pooled)


def model_for(kind):
    if kind == "byte":
        return ByteCNN(), None, 0, []
    source = ROOT / "models" / kind
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if kind == "codet5p":
        layers = [0, 2, 4, 6, 8, 10]
        encoder = T5EncoderModel.from_pretrained(source, local_files_only=True)
        encoder.encoder.block = nn.ModuleList(encoder.encoder.block[i] for i in layers)
        encoder.config.num_layers = len(layers)
        return T5Joint(encoder), tokenizer, tokenizer.pad_token_id, layers
    layers = [1, 3, 5, 7, 9, 11]
    model = RobertaForSequenceClassification.from_pretrained(
        source, num_labels=2, local_files_only=True, attn_implementation="sdpa")
    model.roberta.encoder.layer = nn.ModuleList(model.roberta.encoder.layer[i] for i in layers)
    model.config.num_hidden_layers = len(layers)
    return model, tokenizer, tokenizer.pad_token_id, layers


def byte_view(text, cap):
    data = text.encode("utf-8")
    if len(data) > cap:
        half = cap // 2
        data = data[:cap-half] + data[-half:]
    return np.frombuffer(data, dtype=np.uint8).astype(np.int16) + 1


def encode(texts, tokenizer, cap):
    if tokenizer is None:
        return [byte_view(text, cap) for text in texts]
    tokenized = []
    for start in range(0, len(texts), 1024):
        content = tokenizer(texts[start:start+1024], max_length=cap, truncation=True)["input_ids"]
        tokenized.extend(np.asarray(row, dtype=np.int32) for row in content)
    return tokenized


def batch_plan(rows, sequence, size):
    for start in range(0, len(rows), BLOCK):
        order = sorted(rows[start:start+BLOCK], key=lambda i: len(sequence[i]))
        yield from (order[index:index+size] for index in range(0, len(order), size))


def batch(rows, sequence, pad):
    width = max(len(sequence[i]) for i in rows)
    tokens = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros_like(tokens)
    for row, i in enumerate(rows):
        length = len(sequence[i])
        tokens[row, :length] = sequence[i]
        mask[row, :length] = 1
    return torch.as_tensor(tokens, device="cuda"), torch.as_tensor(mask, device="cuda")


def score(model, kind, tokens, mask):
    if kind in ("codebert", "graphcodebert"):
        return model(input_ids=tokens, attention_mask=mask).logits
    return model(tokens, mask)


def train(model, config, tokens, pad, y):
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)
    initial_plan = list(batch_plan(list(range(len(y))), tokens, config.batch_size))
    total = config.epochs * len(initial_plan)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total), total)
    weights = np.clip((len(y) - y.sum(axis=0)) / y.sum(axis=0), 1, 5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(weights, device="cuda", dtype=torch.float32))
    y = torch.as_tensor(y, dtype=torch.float32, device="cuda")
    order = list(range(len(y)))
    start = time.perf_counter()
    step = 0
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.epochs):
        rng.shuffle(order)
        plan = list(batch_plan(order, tokens, config.batch_size))
        rng.shuffle(plan)
        losses = torch.zeros((), device="cuda")
        for ids in plan:
            inputs = batch(ids, tokens, pad)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = score(model, config.family, *inputs)
                loss = loss_fn(prediction, y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses += loss.detach()
            step += 1
            if step % 200 == 0:
                print(config.family, "step", step, "of", total,
                      "seconds", round(time.perf_counter()-start, 1), flush=True)
        print(config.family, "epoch", epoch+1, "loss", round(losses.item()/len(plan), 5), flush=True)
    return dict(seconds=time.perf_counter()-start, steps=step, peak_MiB=int(torch.cuda.max_memory_allocated()/2**20),
                positive_weights=weights.tolist())


def batch_gflops(kind, model, n, length):
    if kind == "byte":
        operations = 0
        for layer in model.conv:
            stride = layer.stride[0]
            length = (length+stride-1)//stride
            operations += 2 * length * layer.in_channels * layer.out_channels * layer.kernel_size[0]
        operations += 2 * (192*96 + 96*2)
    else:
        c = model.encoder.config if kind == "codet5p" else model.config
        width, ff, depth = ((c.d_model, c.d_ff, c.num_layers) if kind == "codet5p"
                            else (c.hidden_size, c.intermediate_size, c.num_hidden_layers))
        ff_gates = 3 if kind == "codet5p" and c.is_gated_act else 2
        operations = depth * (2*length*(4*width*width+ff_gates*width*ff)+4*length**2*width)
        operations += 2*width*(2 if kind == "codet5p" else width+2)
    return n * operations / 1e9


@torch.inference_mode()
def predict(model, kind, sequence, pad, batch_size):
    model.eval()
    probabilities = np.empty((len(sequence), 2), dtype=np.float32)
    gflops = 0.
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for ids in batch_plan(list(range(len(sequence))), sequence, batch_size):
            inputs = batch(ids, sequence, pad)
            probabilities[ids] = score(model, kind, *inputs).float().sigmoid().cpu().numpy()
            gflops += batch_gflops(kind, model, len(ids), inputs[0].shape[1])
    return probabilities, dict(forward_and_pack_seconds=time.perf_counter()-start, analytic_dense_gflops=gflops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("byte", "codet5p", "codebert", "graphcodebert"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if args.public and not args.full:
        parser.error("Public confirmation requires a frozen full-fit model")
    if args.family == "byte":
        config = Config(args.family, 2048, 128, 5, 0.001)
    elif args.family == "codet5p":
        config = Config(args.family, 512, 32, 1, 2e-5) if args.full else Config(args.family, 512, 32, 2, 3e-5)
    else:
        config = Config(args.family, 512, 32, 2, 2e-5)
    torch.manual_seed(config.seed)
    training, y, evaluation, labels, ids, calibration = dataset(args.full, args.public)
    model, tokenizer, pad, layers = model_for(config.family)
    model.cuda()
    out = HERE / "runs" / ("full" if args.full else "pilot") / config.family
    out.mkdir(parents=True, exist_ok=True)
    path = out / "weights.pt"
    threshold = None
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if args.public:
        threshold = json.loads((out / "metrics.json").read_text())["thresholds"]
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    else:
        warm_start = None
        if args.full and config.family == "codet5p":
            warm_start = HERE / "runs/pilot/codet5p/weights.pt"
            model.load_state_dict(torch.load(warm_start, weights_only=True, map_location="cpu"))
        write_json(out / "config.json", dict(**asdict(config), retained_source_layers=layers,
                                            fit_rows=len(y), parameters=parameters,
                                            warm_start=str(warm_start.relative_to(ROOT)) if warm_start else None))
        start = time.perf_counter()
        train_tokens = encode(training, tokenizer, config.cap)
        print(args.family, "encoded fit", len(y), "seconds", round(time.perf_counter()-start, 2), flush=True)
        result = train(model, config, train_tokens, pad, y)
        pending = path.with_suffix(".part")
        torch.save(model.state_dict(), pending)
        pending.replace(path)
        write_json(out / "training.json", result)
    start = time.perf_counter()
    eval_tokens = encode(evaluation, tokenizer, config.cap)
    tokenizing = time.perf_counter()-start
    probabilities, detail = predict(model, config.family, eval_tokens, pad, config.batch_size)
    cost = dict(platform=torch.cuda.get_device_name(), rows=len(ids), parameters=parameters,
                bytes=path.stat().st_size, tokenize_seconds=tokenizing, **detail)
    save(out / "public" if args.public else out, labels, ids, calibration, probabilities, cost, threshold)


if __name__ == "__main__":
    main()
