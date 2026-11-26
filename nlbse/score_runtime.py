"""Apply the competition's published formula to an already profiled evaluation."""
import argparse
import json
from pathlib import Path


def competition_formula(mean_f1, total_seconds, total_gflops):
    return (0.6 * mean_f1 + 0.2 * max(1 - total_seconds / 1000, 0)
            + 0.2 * max(1 - total_gflops / 3_000_000, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric_receipts", nargs="+", type=Path)
    args = parser.parse_args()
    results = {}
    for path in args.metric_receipts:
        payload = json.loads(path.read_text())
        cost = payload["cost"]
        if not cost["rows"] == 30655 or cost["profiler_counted_gflops"] is None:
            parser.error(f"{path}: expected profiled full public validation")
        score = competition_formula(payload["scores"]["mean"]["f1"], cost["forward_seconds"],
                                    cost["profiler_counted_gflops"])
        results[path.parent.name] = dict(formula_on_local_measurements=score, scope="public validation",
                                         official_organizer_rank="pending")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
