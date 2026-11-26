"""Count conflicting labels for identical short model inputs in both rotations."""

from collections import Counter
import json

from discovery import data
from evaluate import inputs, raw
from rotate import HERE, PLAN, ROOT, fingerprint
from pipeline import write_json


def audit(rows, target, texts, gold):
    def summarize(key):
        negative = Counter(key(texts[row]) for row in rows if not gold[row, 0])
        positive = Counter(key(texts[row]) for row in rows if gold[row, 0])
        contradictions = negative.keys() & positive.keys()
        affected = [key(texts[row]) for row in target if key(texts[row]) in contradictions]
        return {"conflicting_groups": len(contradictions),
                "target_misses_with_conflict": len(affected),
                "opposite_label_rows_for_target_groups": sum(negative[value] for value in set(affected))}

    return {"rows": len(rows), "target_misses": len(target), "exact_input": summarize(lambda value: value),
            "code_without_comments_or_whitespace": summarize(fingerprint)}


def main():
    texts, gold, projects, _, parts = inputs()
    policy = json.loads(PLAN.read_text())
    cap, threshold = policy["short_chars_max"], policy["jointly_low_cutoff"]
    compact = data(ROOT / "round12/heads/compact_raw_test.csv")
    unix = data(ROOT / "round12/heads/unix_raw_test.csv")
    current = raw("test")
    old = [i for i in compact if len(texts[i]) <= cap]
    old_target = [i for i in old if gold[i, 0] and compact[i] < threshold and unix[i] < threshold]
    new = [i for i in parts["test"] if len(texts[i]) <= cap]
    new_target = [i for i, a, b in zip(parts["test"], current["compact"], current["unix"], strict=True)
                  if len(texts[i]) <= cap and gold[i, 0] and a < threshold and b < threshold]
    result = {"discovery": audit(old, old_target, texts, gold), "replication": audit(new, new_target, texts, gold),
              "replication_target_projects": dict(Counter(projects[i] for i in new_target))}
    write_json(HERE / "collision_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
