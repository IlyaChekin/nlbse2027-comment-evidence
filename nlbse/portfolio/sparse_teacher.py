"""Cache training-only soft labels from the frozen joint NLBSE teacher."""
import hashlib

import numpy as np

from protocol import HERE, dataset
from pipeline import select_indices, write_json
from t5_entry import PORTABLE, load, probabilities

OUT = HERE / "learned_sparse"


def main():
    texts, labels, _, _, _, _ = dataset(full=True)
    fit, _ = select_indices(pilot=False)
    model, tokenizer = load(PORTABLE, "cuda")
    predictions, timing = probabilities(model, tokenizer, texts, "cuda", "comment")
    if predictions.shape != labels.shape or len(predictions) != len(fit):
        raise ValueError("Teacher and fit examples are misaligned")
    with (PORTABLE / "model.safetensors").open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    pending = OUT / "teacher.part.npz"
    np.savez_compressed(pending, rows=fit, probabilities=predictions)
    pending.replace(OUT / "teacher.npz")
    write_json(OUT / "teacher.json", dict(weights_sha256=digest, fit_rows=len(fit),
                                         mode="comment", cost=timing))
    print("Cached teacher scores for", len(fit), "fit rows", timing, flush=True)


if __name__ == "__main__":
    main()
