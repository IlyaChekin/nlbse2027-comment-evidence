"""Run the model frozen in the release manifest on an ordered CSV."""
import argparse
import json
from pathlib import Path

import students


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entry = json.loads((students.HERE / "entry.json").read_text())
    model = json.loads((students.HERE / "runs" / entry["family"] / "selected.json").read_text())
    if model["settings"]["sha256"] != entry["checkpoint_sha256"]:
        raise ValueError("The selected model does not match the release manifest")
    mode = entry["mode"]
    if mode.startswith("compact_"):
        import compact_exit
        compact_exit.predict(args.input, args.output, lexical=mode.startswith("compact_lexical"),
                             deferred_offsets=mode.endswith("_deferred"))
    elif mode in {"student_context", "student_lexical", "unix_context", "unix_lexical"}:
        import student_context
        student_context.predict(args.input, args.output, fused=True, lexical=mode.endswith("_lexical"), family=entry["family"])
    else:
        students.predict(entry["family"], args.input, args.output, fused=(mode == "student_fused"))


if __name__ == "__main__":
    main()
