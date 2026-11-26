"""Verify archived decisions and recompute the score bounds without GPU or dataset."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def metrics(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(2*tp+fp+fn, 1))


def tally(archived, name, limit, threshold):
    counts = Counter()
    key = f"{name}_{limit}"
    for row in archived:
        g, b, p = (row[k] == "1" for k in ("label_mat", "base_mat", key))
        if threshold is not None and p != (float(row[f"{name}_score"]) > threshold):
            raise ValueError("Prediction failed to reproduce from the archived score and threshold")
        counts["tp"] += g and p
        counts["fp"] += not g and p
        counts["fn"] += g and not p
        counts["recovered"] += g and not b and p
        counts["lost"] += g and b and not p
        if g and not b and p:
            counts["recovered_with_comment" if row["has_comment"] == "1" else "recovered_without_comment"] += 1
    return metrics(counts), counts


def holdouts():
    result = []
    integrity = {}
    for source, groups in (("test", ("test_a", "test_b")), ("public", ("public",))):
        summary = read(HERE/f"project_{source}.json")
        for group in groups:
            path = HERE/f"project_{group}.csv"
            archived = rows(path)
            ids = [int(r["row"]) for r in archived]
            if any(a >= b for a, b in zip(ids, ids[1:])):
                raise ValueError("Prediction rows have duplicated IDs or changed order")
            with path.open("rb") as stream:
                integrity[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
            reference, _ = tally(archived, "baseline", "fixed", .5)
            for method, record in summary[group]["methods"].items():
                for mode in ("fixed", "matched"):
                    actual, counts = tally(archived, method, mode, record[f"{mode}_threshold"])
                    expected = record[mode]
                    if expected["base"] != reference or expected["candidate"] != actual or (
                        any(expected[key] != counts[key] for key in
                            ("recovered", "lost", "recovered_with_comment", "recovered_without_comment"))
                    ):
                        raise ValueError(f"Result mismatch for {group} {method} {mode}")
                    if mode == "matched" and actual["fp"] != reference["fp"]:
                        raise ValueError("Matched comparison has a different number of false positives")
                    result.append(dict(group=group, rows=len(archived), method=method, mode=mode,
                                       baseline_fn=reference["fn"], baseline_fp=reference["fp"],
                                       tp=actual["tp"], fp=actual["fp"], fn=actual["fn"], f1=actual["f1"],
                                       recovered=counts["recovered"], lost=counts["lost"],
                                       recovered_with_comment=counts["recovered_with_comment"],
                                       recovered_without_comment=counts["recovered_without_comment"],
                                       threshold=record[f"{mode}_threshold"], residual_ap=record["residual_ap"]))
    return result, integrity


def source_observations():
    table = []
    for part in ("heldout", "public"):
        path = HERE / ("source_audit_public.json" if part == "public" else "source_audit.json")
        if not path.exists():
            continue
        source = read(path)
        observed = Counter((p["project"], p["mat_positive"], p["observation"]) for p in source["observations"])
        for key, counts in source["totals"].items():
            project, population = key.split("/", 1)
            positive = population == "missed_positive"
            if any(n != observed[(project, positive, condition)] for condition, n in counts.items()):
                raise ValueError("Source observation summary differed from the receipts")
            exact = sum(counts.get(status, 0) for status in (
                "marker_in_adjacent_preceding_comment", "marker_inside_matched_span", "match_without_adjacent_marker"))
            table.append(dict(split=part, project=project, population=population, rows=sum(counts.values()),
                              exact=exact, preceding=counts.get("marker_in_adjacent_preceding_comment", 0),
                              inside=counts.get("marker_inside_matched_span", 0),
                              without_tag=counts.get("match_without_adjacent_marker", 0),
                              source_unavailable=counts.get("source_unavailable", 0),
                              no_match=counts.get("no_token_match", 0),
                              multiple_matches=counts.get("multiple_token_matches", 0)))
    return table


def bounds():
    reference = read(ROOT/"round16/context_public/results.json")["full"]
    vul = reference["quality"]["vulnerability"]["f1"]
    cost = reference["cost"]
    base, _ = tally(rows(HERE/"project_public.csv"), "baseline", "fixed", .5)
    def project(recovered, fp=base["fp"]):
        mat = metrics(dict(tp=base["tp"]+recovered, fp=fp, fn=base["fn"]-recovered))["f1"]
        score = (.3*(vul+mat) + .2*max(1-cost["forward_seconds"]/1000, 0)
                 + .2*max(1-cost["analytic_gflops"]/3_000_000, 0))
        return dict(additional_tp=recovered, fp=fp, mat_f1=mat, projected_score=score)
    if abs(project(0)["projected_score"]-reference["score"]) > 1e-12:
        raise ValueError("The formula failed to reproduce the measured round 16 score")
    output = dict(baseline=base, vulnerability_f1=vul, cost=cost, measured_score=reference["score"],
                  to_reach_target=next(project(n) for n in range(base["fn"]+1) if project(n)["projected_score"] >= .97),
                  if_all_visible_comments_recovered=project(19), plus_one=project(1),
                  if_all_misses_recovered=project(base["fn"]),
                  max_with_zero_mat_fp_and_no_recall_gain=project(0, 0))
    source = [row for row in source_observations()
              if row["split"] == "public" and row["population"] == "missed_positive"]
    if source:
        total = sum(row["preceding"]+row["inside"] for row in source)
        output["if_all_source_confirmed_public_misses_recovered"] = project(total)
    return output


if __name__ == "__main__":
    output, digests = holdouts()
    write(HERE/"results.csv", output)
    source_table = source_observations()
    if source_table:
        write(HERE/"source_results.csv", source_table)
    limits = bounds()
    (HERE/"score_limits.json").write_text(json.dumps(limits, indent=2)+"\n")
    (HERE/"prediction_hashes.json").write_text(json.dumps(digests, indent=2)+"\n")
    print(json.dumps(limits, indent=2))
