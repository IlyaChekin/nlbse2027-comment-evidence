"""Try CUDA's memory-efficient fused attention with the frozen T5 relative bias."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round8"))
from late_tokens import (EARLY, Policy, attention_tensor, reduced_forward, selection, take_tokens)
from frontier import (JointHead, PORTABLE, THRESHOLDS, as_common_scale, blocks, dataset, dimensions,
                      lexical_features, load_checkpoint, ordered_batches, pool, save, use_inputs)
from tokenizer_speed import DeferredOffsets
from analyze import predict_bits
from pipeline import write_json
from score_runtime import competition_formula
from windows import BLOCK, chunks, pack
from math import ceil


def padded_stride_bias(bias):
    aligned = (-bias.shape[-1]) % 8
    return F.pad(bias, (0, aligned))[..., :bias.shape[-1]] if aligned else bias.contiguous()


def fused_block(block, state, bias, indices=None):
    attn_layer = block.layer[0]
    attn = attn_layer.SelfAttention
    normalized = attn_layer.layer_norm(state)
    query_states = normalized if indices is None else take_tokens(normalized, indices)
    query = attention_tensor(attn.q, query_states, attn.n_heads, attn.key_value_proj_dim)
    key = attention_tensor(attn.k, normalized, attn.n_heads, attn.key_value_proj_dim)
    value = attention_tensor(attn.v, normalized, attn.n_heads, attn.key_value_proj_dim)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=bias,
                                             dropout_p=0., scale=1.)
    out = out.transpose(1, 2).contiguous().flatten(2)
    residual = state if indices is None else take_tokens(state, indices)
    result = block.layer[-1](residual + attn.o(out))
    if indices is not None:
        result = state.scatter(1, indices[..., None].expand(-1, -1, state.shape[-1]), result)
    return result


def token_scores(model, state):
    return model.head(model.encoder.encoder.final_layer_norm(state)).abs().amax(-1)


def attention_bias(stack, mask, bias_cache):
    width = mask.shape[1]
    # Evaluation-only cache. Each entry is the original relative bias without a padding mask.
    if width not in bias_cache:
        relative = stack.block[0].layer[0].SelfAttention.compute_bias(
            width, width, device=mask.device, cache_position=torch.arange(width, device=mask.device))
        bias_cache[width] = relative.half()
        if len(bias_cache) > 64:
            bias_cache.pop(next(iter(bias_cache)))
    return bias_cache[width].masked_fill(mask[:, None, None, :] == 0, -torch.inf)


def prefix_fused(model, x, mask, depth, bias_cache):
    stack = model.encoder.encoder
    bias = attention_bias(stack, mask, bias_cache)
    fast_bias = padded_stride_bias(bias)
    state = stack.embed_tokens(x)
    for block in stack.block[:depth]:
        state = fused_block(block, state, fast_bias)
    return state, bias


def finish_fused(model, state, bias, mask, policy, depth):
    stack = model.encoder.encoder
    score = token_scores(model, state) if policy.selector == "saliency" else None
    indices, _ = selection(policy, state, mask, score)
    batch_ids = torch.arange(state.shape[0], device=state.device)
    heads = torch.arange(bias.shape[1], device=state.device)
    selected_bias = padded_stride_bias(bias[batch_ids[:, None, None], heads[None, :, None], indices[:, None, :], :])
    for block in stack.block[depth:]:
        state = fused_block(block, state, selected_bias, indices)
    return model.head(pool(stack, state, mask))


@torch.inference_mode()
def predict(model, tokenizer, texts, saved, name, fast, bias_cache, *, batcher=chunks):
    stack = model.encoder.encoder
    policy = Policy(**saved["spec"])
    head = JointHead.from_dict(saved["head"])
    p = np.empty((len(texts), 2), dtype=np.float32)

    def prepare_at(start):
        items = texts[start:start+BLOCK]
        return use_inputs(items, tokenizer, wide=True), lexical_features(items)

    start = time.perf_counter()
    dims = dimensions(model)
    d, f = dims["hidden_size"], dims["intermediate_size"]
    forward, volume = 0., 0.
    with torch.autocast("cuda", dtype=torch.float16):
        for at, (data, side) in ordered_batches(range(0, len(texts), BLOCK), prepare_at, True):
            sequences = data.views["head"]
            for group in batcher(sequences, range(len(sequences))):
                x, mask = pack(sequences, group, data.pad, "cuda")
                tick = time.perf_counter()
                if fast:
                    state, bias = prefix_fused(model, x, mask, EARLY, bias_cache)
                    logits = finish_fused(model, state, bias, mask, policy, EARLY)
                else:
                    state, bias, positions = blocks(stack, x, mask)
                    scores = token_scores(model, state) if policy.selector == "saliency" else None
                    logits, _ = reduced_forward(model, state, bias, positions, mask, policy, scores, dimensions(model))
                logits = logits.float().cpu().numpy()
                forward += time.perf_counter()-tick
                batch, length = x.shape
                q = min(length, max(8, ceil(length*policy.keep)))
                dense = EARLY*(2*length*(4*d*d+2*d*f)+4*length*length*d)
                selected = (dims["layers"]-EARLY)*(4*(length+q)*d*d+4*q*d*f+4*length*q*d)
                volume += batch*(dense+selected+4*d)/1e9
                features = np.column_stack((logits, side[group]))
                p[at+np.asarray(group)] = as_common_scale(head(features), saved["thresholds"])
    return p, {"wall_seconds": time.perf_counter()-start, "forward_seconds": forward, "analytic_gflops": volume, "variant": name}


def benchmark(output, mode, repeat):
    cfg = json.loads((HERE.parent/"round8/tokens_frozen/policy.json").read_text())
    row = cfg["configurations"][mode]
    _, _, texts, y, ids, cal = dataset(public=True)
    model, tokenizer = load_checkpoint(PORTABLE, "cuda")
    tokenizer = DeferredOffsets(tokenizer)
    samples, predictions = {"eager": [], "fused": []}, {}
    cache = {}
    for engine in samples:
        predict(model, tokenizer, texts[:2048], row, mode, engine=="fused", cache)
    labels = tuple(samples)
    for i in range(repeat):
        for engine in labels if i%2==0 else labels[::-1]:
            p, cost = predict(model, tokenizer, texts, row, mode, engine=="fused", cache)
            if engine in predictions and not np.array_equal(p>.5, predictions[engine]>.5):
                raise ValueError(f"Unstable {engine} predictions")
            predictions[engine] = p
            samples[engine].append(cost)
            print(i+1, engine, cost, flush=True)
    old = HERE.parent/"round8/tokens_final"/mode/"predictions.csv"
    if not np.array_equal(predict_bits(old, ids, {"thresholds":(.5,.5)}), predictions["eager"]>.5):
        raise ValueError("Reference selection does not replay the round 8 tool")
    gflops = json.loads((old.parent/"metrics.json").read_text())["cost"]["analytic_gflops"]
    results = {}
    for engine, p in predictions.items():
        cost = {key:median(t[key] for t in samples[engine]) for key in ("wall_seconds","forward_seconds")}
        cost["analytic_gflops"] = gflops
        save(output/engine, y, ids, cal, p, cost, (.5,.5))
        metrics = json.loads((output/engine/"metrics.json").read_text())
        metrics["local_formula"] = competition_formula(metrics["scores"]["mean"]["f1"],cost["forward_seconds"],gflops)
        results[engine] = metrics
    write_json(output/"analysis.json", dict(mode=mode, trials=samples, results=results,
               binary_changes=int(np.count_nonzero((predictions["eager"]>.5)!=(predictions["fused"]>.5))),
               device=torch.cuda.get_device_name()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"fused")
    parser.add_argument("--mode",default="query_saliency_50")
    parser.add_argument("--repeat",type=int,default=3)
    args=parser.parse_args()
    benchmark(args.output,args.mode,args.repeat)
