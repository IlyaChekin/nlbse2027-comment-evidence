"""Cache the six-block scores on grouped training holdout only."""
import json
import time

import numpy as np

from windows import ROOT, VIEWS, cache_probabilities, encode, load_model, tokenizer_at
from continuation_select import split
from pipeline import TRAIN, load

HERE = ROOT / "round4"


def main():
    ids, cal = split()
    texts, labels, _ = load(TRAIN)
    start = time.perf_counter()
    data = encode([texts[i] for i in ids], tokenizer_at())
    print("encoded", len(ids), "overlength", int((data.lengths > 512).sum()),
          "missing_comment", int(data.missing_comment.sum()), flush=True)
    model = load_model()
    out = {}
    for view in VIEWS:
        out[view] = cache_probabilities(model, data, view, "cuda")
        print(view, "scored", int(np.isfinite(out[view][:,0]).sum()),
              "seconds", round(time.perf_counter()-start, 2), flush=True)
    pending = HERE / "scores.part.npz"
    length_by_view = {f"length_{k}": np.array(list(map(len, value)), dtype=np.int16)
                      for k,value in data.view.items()}
    np.savez_compressed(pending, ids=ids, calibration=cal, labels=labels[ids], lengths=data.lengths,
                        missing_comment=data.missing_comment, **out, **length_by_view)
    pending.replace(HERE / "train_holdout_scores.npz")
    config = model.config
    (HERE / "dimensions.json").write_text(json.dumps(
        dict(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
             layers=config.num_hidden_layers, labels=config.num_labels), indent=2) + "\n")


if __name__ == "__main__":
    main()
