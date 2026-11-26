"""Find inconsistent MAT labels for equal inputs, and compute a finite FP oracle."""

import argparse
from collections import Counter, defaultdict

import numpy as np

from audit import HERE, dataset, split, write_json
from mat import example


def inspect(texts, gold, reference=None):
    base = np.fromiter((example(t, code_required=False)[0] for t in texts), dtype=bool, count=len(texts))
    mixed = defaultdict(lambda: [0, 0])
    for text, y in zip(texts, gold, strict=True):
        mixed[text][int(y)] += 1
    absent = defaultdict(lambda: [0, 0])
    untagged_index = np.flatnonzero(~base)
    for i in untagged_index:
        absent[texts[i]][int(gold[i])] += 1
    gold_untagged = gold & ~base
    budget = int((base & ~gold).sum())
    pos_groups = Counter((n, p) for n, p in absent.values() if p > 0)
    out = dict(rows=len(texts), distinct_inputs=len(mixed), exact_label_conflicts=sum(n > 0 and p > 0 for n,p in mixed.values()),
               baseline_tp=int((base & gold).sum()), baseline_fp=budget, baseline_fn=int(gold_untagged.sum()),
               false_negatives_with_exact_negative=sum(p for n,p in absent.values() if n),
               oracle_at_no_added_fp=sum(p for n,p in absent.values() if not n),
               untagged_positive_input_group_support={f"{n} neg / {p} pos": v for (n,p),v in sorted(pos_groups.items())})
    if reference is not None:
        report = Counter()
        for i in np.flatnonzero(gold_untagged):
            labels = reference.get(texts[i], 0)
            report[{0:"absent",1:"train_negative_only",2:"train_positive_only",3:"train_disagrees"}[labels]] += 1
        out["train_exact_presence"] = dict(report)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("project","public"))
    stage = p.parse_args().stage
    train, labels, _, _ = dataset()
    ids = split()
    reference = defaultdict(int)
    if stage == "public":
        from pipeline import select_indices
        fit, _ = select_indices(False)
    else:
        fit = ids["fit"]
    for i in fit:
        reference[train[i]] |= 1 << int(labels[i,1])
    if stage == "public":
        x,y,_,_ = dataset(True)
        out = {"public":inspect(x,y[:,1].astype(bool),reference)}
    else:
        out = {k:inspect([train[i] for i in ids[k]],labels[ids[k],1].astype(bool),reference)
               for k in ("dev","test_a","test_b")}
    write_json(HERE/f"ambiguity_{stage}.json",out)
    print(out,flush=True)


if __name__ == "__main__":
    main()
