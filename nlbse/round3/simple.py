"""Try length-stratified thresholds with the exact saved six-block logits."""
import csv

import numpy as np
from transformers import RobertaTokenizerFast

from protocol import ROOT, SIX, dataset, length_thresholds, save, threshold


def main():
    texts, labels, ids, nfit, calibration = dataset()
    texts, labels, ids = texts[nfit:], labels[nfit:], ids[nfit:]
    with (ROOT / "runs/own6_packed_tune/predictions.csv").open(newline="") as file:
        source = {int(r["row"]): (float(r["p_vulnerability"]), float(r["p_MAT"])) for r in csv.DictReader(file)}
    p = np.array([source[i] for i in ids])
    tokenizer = RobertaTokenizerFast.from_pretrained(SIX, local_files_only=True)
    lengths = np.array([len(row) for begin in range(0, len(texts), 1024)
                        for row in tokenizer(texts[begin:begin+1024], verbose=False)["input_ids"]])
    frozen = threshold(labels[calibration], p[calibration])
    th, edges = length_thresholds(labels[calibration], p[calibration], lengths[calibration])
    mask = ~calibration
    groups = np.digitize(lengths[mask], edges, right=True)
    common = (ids[mask], labels[mask], p[mask])
    save("simple_reference_cached", *common, p[mask] > frozen, {}, dict(thresholds=frozen.tolist()))
    save("simple_length_cached", *common, p[mask] > th[groups], {},
         dict(thresholds=th.tolist(), edges=edges, selection_group_sizes=np.bincount(groups).tolist()))


if __name__ == "__main__":
    main()
