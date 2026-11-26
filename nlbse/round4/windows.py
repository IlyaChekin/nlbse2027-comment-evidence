"""Build and pack head, tail and comment-centered views of one function."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comment_spans import comment_bits

SIX = ROOT / "submission/entry_checkpoint"
BLOCK = 2048
BATCH = 32
CAP = 512
BODY = CAP - 2
TAILS = ("tail64", "tail128", "tail256", "tail512", "overlap128", "comment128", "comment192")
SPLICES = ("splice64", "splice128", "splice192")
VIEWS = ("head", *TAILS, *SPLICES)


@dataclass
class Inputs:
    view: dict
    lengths: np.ndarray
    missing_comment: np.ndarray
    pad: int


def tokenizer_at(checkpoint=SIX):
    return RobertaTokenizerFast.from_pretrained(checkpoint, local_files_only=True)


def load_model(checkpoint=SIX, device="cuda"):
    return RobertaForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, attn_implementation="eager").eval().to(device)


def comment_start(length, positions, width):
    if not positions:
        return max(0, length - width)
    idx = np.asarray(positions)
    # Evaluate a bounded contiguous window ending or starting at missing comment tokens.
    starts = np.clip(np.concatenate((idx - 16, idx - width + 17)), 0, max(0, length - width))
    counts = np.searchsorted(idx, starts + width) - np.searchsorted(idx, starts)
    return int(starts[np.argmax(counts)])


def comment_window(tokens, positions, width):
    if not positions:
        return tokens[-width:]
    start = comment_start(len(tokens), positions, width)
    return tokens[start:start+width]


def encode(texts, tokenizer, views=VIEWS):
    unknown = set(views) - set(VIEWS)
    if unknown:
        raise ValueError(f"Unknown views: {sorted(unknown)}")
    prepared = {name: [] for name in views}
    lengths, comments = [], []
    need_comments = any(name.startswith("comment") for name in views)
    for start in range(0, len(texts), 1024):
        group = texts[start:start+1024]
        enc = tokenizer(group, add_special_tokens=False, return_offsets_mapping=need_comments, verbose=False)
        for j, tokens in enumerate(enc["input_ids"]):
            n = len(tokens)
            long = n > BODY
            if long and need_comments:
                flags = comment_bits(group[j], enc["offset_mapping"][j])
                positions = [i for i in range(BODY, n) if flags[i]]
            else:
                positions = []
            lengths.append(n + 2)
            comments.append(bool(positions))
            for name in views:
                if name == "head" or (not long and name in SPLICES):
                    body = tokens[:BODY]
                elif not long:
                    prepared[name].append([])
                    continue
                elif name.startswith("tail"):
                    body = tokens[-(int(name[4:])-2):]
                elif name.startswith("splice"):
                    width = int(name[6:])
                    body = tokens[:BODY-width] + tokens[-width:]
                elif name == "overlap128":
                    body = tokens[-min(126, n-BODY+24):]
                else:
                    body = comment_window(tokens, positions, int(name[7:])-2)
                prepared[name].append([tokenizer.bos_token_id, *body, tokenizer.eos_token_id])
    if "head" in views:
        assert prepared["head"][:8] == tokenizer(texts[:8], truncation=True, max_length=CAP)["input_ids"]
    return Inputs(prepared, np.asarray(lengths, dtype=np.int32), np.asarray(comments), tokenizer.pad_token_id)


def chunks(sequences, ids):
    order = sorted(ids, key=lambda i: len(sequences[i]))
    yield from (order[i:i+BATCH] for i in range(0, len(order), BATCH))


def pack(sequences, ids, pad, device):
    width = max(len(sequences[i]) for i in ids)
    data = np.full((len(ids), width), pad, dtype=np.int64)
    mask = np.zeros_like(data)
    for row, i in enumerate(ids):
        n = len(sequences[i])
        data[row, :n] = sequences[i]
        mask[row, :n] = 1
    return torch.as_tensor(data, device=device), torch.as_tensor(mask, device=device)


def forward(model, inputs):
    return model(input_ids=inputs[0], attention_mask=inputs[1]).logits.float().sigmoid()


@torch.inference_mode()
def cache_probabilities(model, data, name, device):
    count = len(data.lengths)
    sequences = data.view[name]
    scores = np.full((count, 2), np.nan, dtype=np.float32)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for start in range(0, count, BLOCK):
            ids = [i for i in range(start, min(count, start+BLOCK)) if sequences[i]]
            for group in chunks(sequences, ids):
                scores[group] = forward(model, pack(sequences, group, data.pad, device)).cpu().numpy()
    return scores
