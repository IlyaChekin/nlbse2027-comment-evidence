"""Probe more conservative joint margins using the frozen early-exit classifier."""
import numpy as np
import torch
from transformers import RobertaTokenizerFast

from ideas import (DEVICE, collect, compare_exit, encode, exit_features, load_model, make_early_head,
                   measure, probabilities, routed)
from pipeline import seed_all, write_json
from protocol import OUT, SIX, TWELVE, dataset, save, threshold


def main():
    seed_all(42)
    texts, y, rows, nfit, is_cal = dataset()
    texts, y, rows = texts[nfit:], y[nfit:], rows[nfit:]
    cal, test = np.flatnonzero(is_cal).tolist(), np.flatnonzero(~is_cal).tolist()
    data = encode(texts, RobertaTokenizerFast.from_pretrained(SIX, local_files_only=True))
    full = load_model(TWELVE)
    early = make_early_head()
    early.load_state_dict(torch.load(OUT / "early_exit/trained_head.pt", map_location="cpu", weights_only=True))
    early.eval().to(DEVICE)

    def two(ids):
        a,b = exit_features(full,data,ids)
        return probabilities(early(a)), probabilities(b)

    cheap, expensive = collect(data, cal, two)
    e, f = threshold(y[cal], cheap), threshold(y[cal], expensive)
    gate, tested = compare_exit(y[cal], cheap, expensive, e, f,
                                cutoffs=(.85,.875,.9,.925,.95,.97,.985,.99,1.01))
    both = sum(p.numel() for model in (early, full) for p in model.parameters())
    early_th, final_th = [torch.as_tensor(p, device=DEVICE, dtype=torch.float32) for p in (e,f)]
    # If every efficient gate fails the quality criterion, measure the .85 tradeoff as a diagnostic.
    cutoff = gate["cutoff"] if gate["rate"] else .85
    p, bits, cost = measure("early_exit_followup", data, test,
                            lambda ids: routed(full, early, data, ids, early_th, final_th, cutoff), both)
    save("early_exit_followup", rows[test], y[test], p, bits, cost, dict(
        thresholds=f.tolist(), early_thresholds=e.tolist(), measured_cutoff=cutoff,
        selected_by_quality_rule=gate, gate_search=tested, diagnostic_only=not gate["rate"],
    ))


if __name__ == "__main__":
    main()
