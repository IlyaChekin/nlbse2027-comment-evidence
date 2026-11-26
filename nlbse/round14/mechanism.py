"""Check which visible tag changes drive the three held-out MAT repairs."""

import csv
import json

import numpy as np

from audit import HERE, LEXEME, STEM, dataset, write_json
from mat import edge, load_policy, score_one


def edit(text, restore=False):
    output, offset = [], 0
    for m in LEXEME.finditer(text):
        if not m[0].startswith(("//", "/*")):
            continue
        replacement = STEM.sub(lambda word: f" {word[0]} " if restore else "q"*len(word[0]), m[0])
        output.extend((text[offset:m.start()], replacement))
        offset = m.end()
    return "".join((*output, text[offset:]))


def shape(text):
    signals = []
    for m in LEXEME.finditer(text):
        if m[0].startswith(("//", "/*")):
            body = m[0][2:]
            for mark in STEM.finditer(body):
                left = body[mark.start()-1:mark.start()] if mark.start() else ""
                right = body[mark.end():mark.end()+1]
                signals.append(dict(tag=mark[0].upper(), before=edge(left), after=edge(right), uppercase=mark[0].isupper()))
    return signals


def main():
    policy, heads = load_policy("project")
    mode = "marker_geometry"
    recipe = policy["by_mode"][mode]
    texts, _, _, _ = dataset()
    out = []
    for group in ("test_a","test_b"):
        with (HERE / f"{group}_{mode}.csv").open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["label"] == r["fixed"] == "1" and r["base"] == "0"]
        for r in rows:
            index = int(r["row"])
            text = texts[index]
            inputs = [text, edit(text), edit(text, restore=True)]
            observed, masked, restored = map(float, score_one(inputs, mode, heads[recipe["key"]]))
            if not np.isclose(observed, float(r["score"])):
                raise ValueError("Frozen geometry changed after the archived test")
            out.append(dict(row=index, project=r["project"], group=group, visible=shape(text),
                            original_score=observed, masked_stem_score=masked, restored_boundary_score=restored,
                            below_cutoff_after_masking=masked <= recipe["threshold"],
                            recognized_by_reference_after_restoring_boundary=restored == 2.))
    write_json(HERE/"mechanism.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
