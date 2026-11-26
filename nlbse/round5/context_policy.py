"""Specify one-pass and conditional two-pass rules for the same classifier."""
from dataclasses import dataclass

import numpy as np

THRESHOLDS = (0.70, 0.50)
ADAPTIVE = ("anchor128", "packed128", "cover64", "cover128")


@dataclass(frozen=True)
class Policy:
    name: str
    comment: str = ""
    gate: str = "missing"
    pool: str = "max"
    comment_vul_mix: float = 0.
    code: str = ""
    margin: float = 0.10
    code_mix: float = 0.25
    first: str = ""
    first_scope: str = "long"
    head_limit: int = 512


def policies():
    yield Policy("head")
    yield Policy("reference", "cover128")
    yield Policy("skip_positive", "cover128", gate="skip")
    yield Policy("seen_control", "seen128")
    yield Policy("equal_length_tail", "tail128")
    yield Policy("all_long_tail", "tail512", gate="long", pool="mean", comment_vul_mix=0.5)
    for view in ("cover64", "cover192", "anchor128", "packed128", "adaptive"):
        yield Policy(view, view, gate="skip")
    yield Policy("skip_short_fragments", "cover128", gate="evidence")
    yield Policy("comment_average", "cover128", pool="mean")
    yield Policy("splice_comment", first="splice_comment", first_scope="missing")
    yield Policy("extend_on_comment", first="extend768", first_scope="missing")
    yield Policy("extend_on_long", first="extend768")
    yield Policy("splice_tail", first="splice_tail")
    for view in ("tail128", "tail256", "overlap256"):
        for margin in (0.08, 0.18):
            for mix in (0.25, 0.5):
                yield Policy(f"comment_{view}_{margin:.2f}_{mix:.2f}", "cover128", gate="skip",
                             code=view, margin=margin, code_mix=mix)


def view_names(policy):
    comments = ADAPTIVE if policy.comment == "adaptive" else (policy.comment,)
    return tuple(dict.fromkeys(name for name in ("head", policy.first, *comments, policy.code) if name))


def first_rows(policy, data):
    if not policy.first:
        return np.zeros(len(data["lengths"]), dtype=bool)
    return data["omitted"] > 0 if policy.first_scope == "missing" else data["lengths"] > policy.head_limit


def jobs(policy, head, data):
    missing = data["omitted"] > 0
    long = data["lengths"] > policy.head_limit
    if policy.gate == "long":
        comment_rows = long
    elif policy.gate == "skip":
        comment_rows = missing & (head[:,1] <= THRESHOLDS[1])
    elif policy.gate == "evidence":
        comment_rows = missing & (head[:,1] <= THRESHOLDS[1]) & ((data["omitted"] >= 6) | data["cue"])
    else:
        comment_rows = missing
    requests = []
    if policy.comment == "adaptive":
        crossing = data["crossing"]
        multiple = (data["spans"] > 1) & ~crossing
        short = (data["omitted"] <= 48) & ~crossing & ~multiple
        for name, keep in zip(ADAPTIVE, (crossing, multiple, short, ~(crossing | multiple | short)), strict=True):
            requests.append((name, "comment", comment_rows & keep))
    elif policy.comment:
        requests.append((policy.comment, "comment", comment_rows))
    if policy.code:
        code_rows = long & ~comment_rows & (np.abs(head[:,0] - THRESHOLDS[0]) <= policy.margin)
        requests.append((policy.code, "code", code_rows))
    return requests


def combine(policy, first, second, mask, role, final):
    before, after = first[mask], second[mask]
    if role == "comment":
        if policy.pool == "max":
            final[mask,1] = np.maximum(before[:,1], after[:,1])
        else:
            final[mask,1] = (before[:,1]+after[:,1])*0.5
        if policy.comment_vul_mix:
            final[mask,0] = before[:,0]+policy.comment_vul_mix*(after[:,0]-before[:,0])
    else:
        final[mask,0] = before[:,0]+policy.code_mix*(after[:,0]-before[:,0])
