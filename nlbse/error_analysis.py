"""Record 24 diagnostic errors without exporting function bodies."""
import csv
import json
import random
import re
from collections import Counter, defaultdict

from pipeline import ROOT, PUBLIC, load, write_json
from transformers import RobertaTokenizerFast


def main():
    path = ROOT / "runs/final_public"
    with (path / "predictions.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    cfg = json.loads((path / "config.json").read_text())
    selected = (ROOT / "runs/final_fit/checkpoint")
    tokenizer = RobertaTokenizerFast.from_pretrained(selected, local_files_only=True)
    texts, labels, _ = load(PUBLIC)
    groups = defaultdict(list)
    for row in rows:
        i = int(row["row"])
        for j, label in enumerate(("vulnerability", "MAT")):
            gold, predicted = int(labels[i][j]), int(row[label])
            if gold != predicted:
                groups[label, "false_positive" if predicted else "false_negative"].append(i)
    rng = random.Random(42)
    samples = []
    for (label, category), indices in sorted(groups.items()):
        for i in rng.sample(indices, min(6, len(indices))):
            function = texts[i]
            full = tokenizer.encode(function, truncation=False)
            markers = [m.start() for m in re.finditer(r"//|/\*", function)]
            last_marker_outside = bool(markers and len(tokenizer.encode(function[:markers[-1]],
                                                                        truncation=False)) >= cfg["length"])
            samples.append(dict(row=i, label=label, category=category, tokens=len(full),
                                truncated=len(full) > cfg["length"], comment_like_markers=len(markers),
                                last_comment_like_marker_beyond_cap=last_marker_outside))
    write_json(ROOT / "reports/error_sample.json", dict(
        sample=samples, public_error_counts={f"{a}_{b}": len(c) for (a,b),c in groups.items()},
        interpretation="Marker search is heuristic and can match literals. Sample is balanced across error types.",
        sampled_diagnostics=dict(truncated=sum(r["truncated"] for r in samples),
                                 marker_beyond_cap=sum(r["last_comment_like_marker_beyond_cap"] for r in samples))))
    print(Counter((r["label"], r["category"]) for r in samples))


if __name__ == "__main__":
    main()
