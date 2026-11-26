"""Count comment evidence among MAT errors without exporting source text."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round10"))
from students import ROOT, dataset, write_json
from comment_spans import LEXEME
from frontier import TAG
from neighbor_eval import table


def evidence(text):
    comments = [s[0] for s in LEXEME.finditer(text) if s[0].startswith(("//", "/*"))]
    if any(TAG.search(comment) for comment in comments):
        return "comment_with_standard_tag"
    if TAG.search(text):
        return "standard_tag_outside_comment"
    return "comment_without_standard_tag" if comments else "no_parsed_comment"


def main():
    _, _, texts, y, row_ids, _ = dataset(public=True)
    predictions = table(ROOT / "round10/student_context/student_lexical/predictions.csv")
    if not np.array_equal(predictions[:,0], row_ids):
        raise ValueError("Unexpected prediction rows")
    columns = defaultdict(lambda: dict(rows=0, positives=0, false_negative=0, false_positive=0))
    for text, positive, prediction in zip(texts, y[:,1], predictions[:,3], strict=True):
        item = columns[evidence(text)]
        item["rows"] += 1
        item["positives"] += int(positive)
        item["false_negative"] += int(positive and not prediction)
        item["false_positive"] += int(prediction and not positive)
    write_json(HERE / "mat" / "evidence_audit.json", dict(columns))
    print(dict(columns),flush=True)


if __name__ == "__main__":
    main()
