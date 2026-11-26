"""Build candidate contexts and align omitted C-style comment spans."""
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "round4")]
from comment_spans import LEXEME
from windows import BODY, comment_start, comment_window

CUE = re.compile(r"\b(?:todo|fixme|hack|xxx|workaround|temporary|refactor|revisit)\b", re.I)
COMMENT_VIEWS = ("cover64", "cover128", "cover192", "anchor128", "packed128", "seen128",
                 "splice_comment")
COMPACT_VIEW = "omitted128"
LONG_VIEWS = ("tail128", "tail256", "tail512", "overlap256", "extend768", "splice_tail")
VIEWS = ("head", *COMMENT_VIEWS, *LONG_VIEWS)


@dataclass
class InputViews:
    views: dict
    lengths: np.ndarray
    omitted: np.ndarray
    groups: np.ndarray
    crossing: np.ndarray
    cue: np.ndarray
    pad: int

    @property
    def missing(self):
        return self.omitted > 0


def annotation_spans(text, offsets, boundary):
    """Use the established lexer and retain distinct spans and cue offsets."""
    matches = [m for m in LEXEME.finditer(text) if m[0].startswith(("//", "/*"))]
    tokens = [[] for _ in matches]
    current = 0
    for i, (start, end) in enumerate(offsets):
        while current < len(matches) and matches[current].end() <= start:
            current += 1
        if current < len(matches) and matches[current].start() < end and end > start:
            tokens[current].append(i)
    results = []
    for match, positions in zip(matches, tokens, strict=True):
        if positions and positions[-1] >= boundary:
            cue = CUE.search(match[0])
            location = None
            if cue:
                char_position = match.start() + cue.start()
                location = next((idx for idx in positions if offsets[idx][1] > char_position), positions[0])
            results.append((positions, location))
    return results


def anchored(tokens, spans, width, boundary):
    """Prefer windows that restore a comment beginning or a marked note."""
    scores = np.zeros(len(tokens), dtype=np.int32)
    offsets = []
    for positions, cue in spans:
        missing = [i for i in positions if i >= boundary]
        scores[missing] += 1
        scores[positions[0]] += 12
        offsets.extend((positions[0], missing[0], missing[-1]))
        if cue is not None:
            scores[cue] += 24
            offsets.append(cue)
    offsets = np.asarray(offsets)
    starts = np.clip(np.concatenate((offsets-16, offsets-width//2, offsets-width+17)),
                     0, max(0, len(tokens)-width))
    prefix = np.concatenate(([0], np.cumsum(scores)))
    end = np.minimum(starts+width, len(tokens))
    best = int(starts[np.argmax(prefix[end]-prefix[starts])])
    return tokens[best:best+width]


def packed(tokens, spans, width, separator, boundary):
    ranked = sorted(spans, key=lambda p: (p[1] is not None, sum(i >= boundary for i in p[0])), reverse=True)[:2]
    ranked.sort(key=lambda p: p[0][0])
    per_span = (width - (len(ranked)-1)*len(separator)) // len(ranked)
    ranges = []
    for positions, cue in ranked:
        first, last = positions[0], positions[-1]
        target = cue if cue is not None else max(first, boundary)
        start = max(0, min(target-12, len(tokens)-per_span))
        if last-first < per_span-12:
            start = max(0, min(first-8, len(tokens)-per_span))
        stop = start+per_span
        if ranges and ranges[-1][1] >= start and stop-ranges[-1][0] <= width:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], stop))
        else:
            ranges.append((start, stop))
    out = []
    previous = 0
    for start, stop in ranges:
        if out and start > previous:
            out.extend(separator)
        out.extend(tokens[max(start, previous):stop])
        previous = max(previous, stop)
    return out[:width]


def build_view(name, tokens, spans, positions, separator, boundary):
    if name == "head":
        return tokens[:boundary]
    if (name in COMMENT_VIEWS or name == COMPACT_VIEW) and not positions:
        return []
    if name == COMPACT_VIEW:
        start = comment_start(len(tokens), positions, 126)
        return [tokens[i] for i in positions if start <= i < start + 126]
    if name.startswith("cover"):
        return comment_window(tokens, positions, int(name[5:])-2)
    if name == "anchor128":
        return anchored(tokens, spans, 126, boundary)
    if name == "packed128":
        return packed(tokens, spans, 126, separator, boundary)
    if name == "seen128":
        return tokens[boundary-126:boundary]
    if name == "splice_comment":
        fragment = packed(tokens, spans, 126, separator, boundary)
        return tokens[:boundary-len(fragment)-len(separator)] + separator + fragment
    if name.startswith("tail"):
        return tokens[-(int(name[4:])-2):]
    if name == "overlap256":
        start = boundary-32
        return tokens[start:start+254]
    if name == "extend768":
        return tokens[:766]
    if name == "splice_tail":
        return tokens[:boundary-128] + tokens[-128:]
    raise ValueError(f"Unknown view: {name}")


def prepare(texts, tokenizer, names=VIEWS, *, metadata=True, head_limit=BODY+2):
    names = tuple(dict.fromkeys(names))
    if set(names) - {*VIEWS, COMPACT_VIEW}:
        raise ValueError(f"Unsupported views: {names}")
    need_spans = metadata or any(name in COMMENT_VIEWS or name == COMPACT_VIEW for name in names)
    seqs = {name: [] for name in names}
    boundary = head_limit-2
    lengths, omitted, groups, crossing, cues = [], [], [], [], []
    separator = tokenizer("\n", add_special_tokens=False)["input_ids"]
    for start in range(0, len(texts), 1024):
        batch = texts[start:start+1024]
        enc = tokenizer(batch, add_special_tokens=False, return_offsets_mapping=need_spans, verbose=False)
        for row, tokens in enumerate(enc["input_ids"]):
            n = len(tokens)
            long = n > boundary
            spans = annotation_spans(batch[row], enc["offset_mapping"][row], boundary) if long and need_spans else []
            positions = sorted({idx for span, _ in spans for idx in span if idx >= boundary})
            lengths.append(n+2)
            omitted.append(len(positions))
            groups.append(len(spans))
            crossing.append(any(p[0] < boundary for p, _ in spans))
            cues.append(any(cue is not None for _, cue in spans))
            for name in names:
                body = build_view(name, tokens, spans, positions, separator, boundary) if name == "head" or long else []
                seqs[name].append([tokenizer.bos_token_id, *body, tokenizer.eos_token_id]
                                  if body or name == "head" else [])
    if texts and "head" in names:
        original = tokenizer(texts[:8], truncation=True, max_length=head_limit)["input_ids"]
        if seqs["head"][:8] != original:
            raise ValueError("The first window differs from tokenizer truncation")
    return InputViews(seqs, np.asarray(lengths, dtype=np.int32), np.asarray(omitted, dtype=np.int32),
                      np.asarray(groups, dtype=np.int32), np.asarray(crossing, dtype=bool),
                      np.asarray(cues, dtype=bool), tokenizer.pad_token_id)
