"""Compare exact MAT annotation semantics with the current word-boundary rule."""
import csv
from pathlib import Path
import re
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "round14"))
from audit import dataset, split, write_json, checksum, LEXEME, TAG
from mat import cut_at_budget, evaluate
from marker import LETTERS, LICENSE, upstream_tag

MARKER = re.compile("todo|fixme|xxx|hack", re.I | re.ASCII)
SOURCE_COMMENT = re.compile(r"/\*.*?\*/|(?<!:)//[^\n]*", re.S)
# Despite DOTALL for blocks, the upstream line alternative only consumes through LF.
METHODS = ("baseline", "union_lexical", "anchor_lexical", "union_upstream",
           "replace_lexical", "replace_upstream", "consensus_priority")


def evidence(text):
    # Case-insensitive STEM also acts as a fast path on punctuation-normalized text.
    if not MARKER.search(text) and not MARKER.search(LETTERS.sub("", text)):
        return False, False, False, False, False, False, False
    comments = [span[0] for span in LEXEME.finditer(text) if span[0].startswith(("//", "/*"))]
    strict = any(TAG.search(comment) for comment in comments)
    lexical = any(map(upstream_tag, comments))
    loose = any(upstream_tag(match[0]) for match in SOURCE_COMMENT.finditer(text))
    anchored = any(upstream_tag(comment) and MARKER.match(comment[2:].lstrip("* \t\r\n"))
                   for comment in comments)
    licensed = any(TAG.search(comment) and LICENSE.search(comment) for comment in comments)
    stem = any(MARKER.search(comment) for comment in comments)
    return strict, lexical, loose, anchored, licensed, stem, bool(comments)


def features(texts, ids):
    return np.asarray([evidence(texts[i]) for i in ids], dtype=bool).T


def method_scores(raw):
    b, lexical, loose, head, _, _, _ = raw
    return dict(baseline=4.*b,
                union_lexical=4.*b+lexical,
                anchor_lexical=4.*b+head,
                union_upstream=4.*b+loose,
                replace_lexical=3.*lexical+b,
                replace_upstream=3.*loose+b,
                consensus_priority=np.select((b & lexical, b, lexical & head, lexical, loose),
                                             (6., 5., 4., 3., 2.)))


def default_threshold(name):
    return {"replace_lexical": 2., "replace_upstream": 2.}.get(name, 0.5)


def lookup(gold, feat):
    b, lexical, loose, head, licensed, stem, _ = feat
    cases = {"base_without_upstream": b & ~loose, "new_lexical": ~b & lexical, "new_upstream": ~b & loose,
             "new_anchor": ~b & head, "new_punctuation_join": ~b & lexical & ~stem,
             "new_lax_comment_parse": ~b & loose & ~lexical, "license_text": licensed}
    return {name: {"positive": int((hit & gold).sum()), "negative": int((hit & ~gold).sum())}
            for name, hit in cases.items()}


def selection(scores, gold, base):
    budget = int((base & ~gold).sum())
    return {key: cut_at_budget(value, gold, budget, .5) for key, value in scores.items()}


def measurements(gold, base, feat, bits):
    result = evaluate(gold, base, bits)
    out = gold & ~base & bits
    _, lexical, loose, head, _, stem, _ = feat
    result["recovered"] = int(out.sum())
    result["recovered_by_word_join"] = int((out & lexical & ~stem).sum())
    result["recovered_by_lexical_suffix_prefix"] = int((out & lexical & stem).sum())
    result["recovered_by_loose_comment_parse"] = int((out & loose & ~lexical).sum())
    result["recovered_with_anchor"] = int((out & head).sum())
    return result


def score(name, texts, ids, label, project, policy=None):
    tick = time.perf_counter()
    feat = features(texts, ids)
    gold, groups = label[ids], project[ids]
    b = feat[0]
    scores = method_scores(feat)
    if policy is None:
        policy = selection(scores, gold, b)
    methods = {}
    predictions = []
    columns = ["row", "project", "gold", "base", "lexical_annotation", "loose_annotation",
               "anchor", "license", "stem", "has_comment_with_marker_hint"]
    for key in METHODS:
        p = scores[key]
        fixed_cut = policy[key]
        fixed = p > fixed_cut
        default = p > default_threshold(key)
        matched = np.zeros_like(b)
        thresholds = {}
        per_family = {}
        for family in sorted(set(groups)):
            where = groups == family
            fp = int((b[where] & ~gold[where]).sum())
            threshold = cut_at_budget(p[where], gold[where], fp, .5)
            thresholds[family] = threshold
            matched[where] = p[where] > threshold
            per_family[family] = evaluate(gold[where], b[where], matched[where])
            if per_family[family]["extra_fp"] > 0:
                raise ValueError(f"FP budget exceeded for {name}/{family}/{key}")
        methods[key] = dict(default=measurements(gold, b, feat, default),
                            frozen=measurements(gold, b, feat, fixed),
                            matched=measurements(gold, b, feat, matched),
                            frozen_threshold=fixed_cut, project_thresholds=thresholds, projects=per_family,
                            residual_ap=float(average_precision_score(gold[~b], p[~b])))
        columns.extend(f"{key}_{k}" for k in ("score", "default", "frozen", "matched"))
        predictions.extend((p, default.astype(int), fixed.astype(int), matched.astype(int)))
    if name != "fit":
        with (HERE / f"annotation_{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(zip(ids, groups, gold.astype(int), *feat.astype(int), *predictions, strict=True))
    elapsed = time.perf_counter()-tick
    print(name, "seconds", round(elapsed, 3), lookup(gold, feat),
          [(key, m["frozen"]["recovered"], m["frozen"]["lost"], m["frozen"]["extra_fp"],
            m["matched"]["recovered"], m["matched"]["extra_fp"]) for key, m in methods.items()], flush=True)
    return dict(rows=len(ids), evidence=lookup(gold, feat), metrics=methods), policy


def main():
    texts, labels, projects, _ = dataset()
    groups = split()
    public, public_y, _, _ = dataset(public=True)
    fit_results, _ = score("fit", texts, groups["fit"], labels[:, 1].astype(bool), projects)
    dev_results, threshold = score("dev", texts, groups["dev"], labels[:, 1].astype(bool), projects)
    selected = max(METHODS, key=lambda key: (
        dev_results["metrics"][key]["frozen"]["recovered"] - dev_results["metrics"][key]["frozen"]["lost"],
        key == "baseline",
    ))
    config = dict(selected=selected, thresholds=threshold, partition_sha256=checksum(ROOT/"round13/partition.npz"),
                  code_sha256=checksum(Path(__file__)), marker_sha256=checksum(HERE/"marker.py"), manifest_sha256=checksum(ROOT/"data/manifest.json"),
                  annotation_source={
                      "record": "https://doi.org/10.5281/zenodo.12567874",
                      "annotateForSATD.py": "a6a96dee976f7103f9e0bfd2ae35e8c63c2150cde0fe40f95bc764384a9e7bdd",
                      "utils.py": "b5cd9f9e0cc49d580df918a3cb89439c5f059d773789125902233fcb480c515b",
                  })
    write_json(HERE / "annotation_policy.json", config)
    result = {"fit": fit_results, "dev": dev_results}
    for name in ("test_a", "test_b"):
        result[name], _ = score(name, texts, groups[name], labels[:, 1].astype(bool), projects, threshold)
    result["public"], _ = score("public", public, np.arange(len(public)), public_y[:, 1].astype(bool),
                                np.repeat("public", len(public)), threshold)
    write_json(HERE / "annotation_results.json", result)


if __name__ == "__main__":
    main()
