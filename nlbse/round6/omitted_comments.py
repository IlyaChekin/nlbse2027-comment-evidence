"""Test whether never-seen comment tokens account for the second-window gain."""
import argparse
import csv
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from common import (PORTABLE, PUBLIC, TRAIN, forward_rows, load, load_checkpoint,
                    prepare, save, sha256, write_json)
from analyze import changes, group_codes, predict_bits, read, verify
from comment_spans import LEXEME, comment_bits
from masked_context import OUT as ORIGINAL, forward_masked
from project_model import OUT as PROJECT
from round2_report import bootstrap_delta
from windows import BODY, comment_start

TAG = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b", re.IGNORECASE | re.ASCII)


def tag_spans(text):
    for lexeme in LEXEME.finditer(text):
        if lexeme[0].startswith(("//", "/*")):
            yield from TAG.finditer(text, *lexeme.span())


def windows_and_masks(texts, rows, tokenizer):
    """Rebuild the original windows and partition the comments at the head cutoff."""
    selected = [texts[row] for row in rows]
    prepared = prepare(selected, tokenizer, ("cover128",))
    encoded = tokenizer(selected, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    views, packed = ([[] for _ in texts] for _ in range(2))
    full, omitted, seen, omitted_without_tags, has_omitted_tag = {}, {}, {}, {}, {}
    for row, text, window, tokens, offsets in zip(
            rows, selected, prepared.views["cover128"], encoded["input_ids"],
            encoded["offset_mapping"], strict=True):
        comments = np.asarray(comment_bits(text, offsets), dtype=bool)
        omitted_positions = np.flatnonzero(comments[BODY:]) + BODY
        start = comment_start(len(tokens), omitted_positions.tolist(), 126)
        stop = start + len(window) - 2
        if window[1:-1] != tokens[start:stop]:
            raise ValueError(f"The reconstructed window differs at row {row}")
        full[row] = comments[start:stop]
        omitted[row] = full[row] & (np.arange(start, stop) >= BODY)
        seen[row] = full[row] & ~omitted[row]
        positions = np.flatnonzero(omitted[row]) + start
        if not len(positions):
            raise ValueError(f"The selected window has no omitted comment tokens at row {row}")
        views[row] = window
        packed[row] = [window[0], *(tokens[i] for i in positions), window[-1]]
        tags = list(tag_spans(text))
        tag_mask = np.asarray([any(a < tag.end() and tag.start() < b for tag in tags)
                               for a, b in offsets[start:stop]], dtype=bool)
        has_omitted_tag[row] = bool((omitted[row] & tag_mask).any())
        omitted_without_tags[row] = omitted[row] & ~tag_mask
    inputs = SimpleNamespace(views={"cover128": views}, pad=prepared.pad)
    return inputs, packed, full, omitted, seen, omitted_without_tags, has_omitted_tag


def read_head(path, ids, thresholds, gold):
    with (path / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if ids != [int(row["row"]) for row in rows]:
        raise ValueError("Frozen head rows are out of order")
    p = np.asarray([(row["p_vulnerability"], row["p_MAT"]) for row in rows], dtype=np.float32)
    verify(gold, p > thresholds, read(path / "metrics.json"))
    return p


def run(split, output):
    original = ORIGINAL / split
    with np.load(original / "rows.npz") as receipt:
        ids = receipt["ids"].tolist()
        eligible = receipt["eligible"]
        crossing = receipt["crossing"]
        original_counts = receipt["comment_in_window"]
    config = read(original / "design.json")
    thresholds = config["thresholds"]
    path, checkpoint = (PUBLIC, PORTABLE) if split == "public" else (TRAIN, PROJECT / "checkpoint")
    texts, y, fields = load(path, audit=True)
    texts, y = [texts[i] for i in ids], y[ids]
    groups = group_codes(zip(*(np.asarray(fields[key])[ids] for key in ("Projectname", "Filepath")), strict=True))
    head = read_head(original / "head", ids, thresholds, y)
    rows = np.flatnonzero(eligible)
    if len(rows) != config["window_calls"] or (head[rows, 1] > thresholds[1]).any():
        raise ValueError("The stored routing flags disagree with the frozen head")

    model, tokenizer = load_checkpoint(checkpoint, "cuda")
    inputs, packed, full, omitted, seen, without_tags, tags = windows_and_masks(texts, rows, tokenizer)
    if not np.array_equal([full[i].sum() for i in rows], original_counts[rows]):
        raise ValueError("Matched comment lengths differ from the original study")

    calibration = np.zeros(len(y), dtype=bool)
    bits = {name: predict_bits(original / name / "predictions.csv", ids, config)
            for name in ("head", "full", "comment_only", "code_only")}
    result = {name: read(original / name / "metrics.json")["scores"] for name in bits}
    variants = (("comment_only_replay", full), ("omitted_only", omitted), ("seen_only", seen),
                ("omitted_without_tags", without_tags), ("omitted_packed", None),
                ("omitted_tag_rule", None), ("all_comment_tag_rule", None))
    for name, mask in variants:
        if mask is not None:
            p, cost = forward_masked(model, inputs, eligible, mask, "comment_only")
            if cost["analytic_gflops"] != config["additional_gflops"]:
                raise ValueError("Masking changed the original padded operation shapes")
        elif name == "omitted_packed":
            cost = dict(analytic_gflops=0., forward_seconds=0.)
            p = forward_rows(model, packed, rows, inputs.pad, "cuda", cost)
        else:
            cost = {}
        combined = head.copy()
        if name == "omitted_tag_rule":
            combined[rows, 1] = [tags[i] for i in rows]
        elif name == "all_comment_tag_rule":
            combined[:, 1] = [next(tag_spans(text), None) is not None for text in texts]
        else:
            combined[rows, 1] = np.maximum(head[rows, 1], p[rows, 1])
        predictions = combined > thresholds
        if name == "comment_only_replay" and not np.array_equal(predictions, bits["comment_only"]):
            raise ValueError("The original comment-only intervention could not be replayed")
        bits[name] = predictions
        save(output / name, y, ids, calibration, combined, dict(
            extra_windows=len(rows) if cost else 0, extra_only=True, **cost), thresholds)
        result[name] = read(output / name / "metrics.json")["scores"]

    contrasts = {}
    for a, b in (("head", "omitted_only"), ("seen_only", "omitted_only"),
                 ("omitted_only", "omitted_packed"), ("full", "omitted_only"),
                 ("head", "omitted_without_tags"), ("omitted_without_tags", "omitted_only"),
                 ("head", "omitted_tag_rule"), ("head", "all_comment_tag_rule")):
        contrasts[f"{b}_minus_{a}"] = dict(
            changes=changes(y, bits[a], bits[b]), interval=bootstrap_delta(y, bits[a], bits[b], groups))
    recovered = (~bits["head"][:, 1]) & bits["full"][:, 1] & (y[:, 1] == 1)
    report = dict(split=split, rows=len(y), thresholds=thresholds, first_pass="frozen CSV",
                  checkpoint_sha256=sha256(checkpoint / "model.safetensors"),
                  source_rows_sha256=sha256(original / "rows.npz"),
                  source_head_sha256=sha256(original / "head/predictions.csv"),
                  num_windows=len(rows), windows_with_seen_comments=sum(bool(seen[i].any()) for i in rows),
                  windows_crossing=int(np.count_nonzero(crossing[rows])),
                  average_omitted_tokens=float(np.mean([omitted[i].sum() for i in rows])),
                  rule_recovered_with_tag=sum(tags[i] for i in np.flatnonzero(recovered)),
                  full_recovered=int(recovered.sum()),
                  scores=result, paired=contrasts)
    write_json(output / "analysis.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("public", "project"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.split, args.output)
