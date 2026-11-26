"""Fetch and verify only the pinned NLBSE assets needed for a given run."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "nlbse/data/manifest.json"
REPOS = {"official": "NLBSE/nlbse27-code-comment-classification", "codebert": "microsoft/codebert-base"}
MODEL_KEYS = {"official": "official", "codebert": "pretrained_codebert"}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(path, url, expected):
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Unexpected bytes in {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    if digest(partial) != expected:
        raise ValueError(f"Downloaded file failed SHA-256: {path}")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="Also fetch the train split and base encoder.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for record in manifest["files"]:
        if args.training or record["split"] == "test":
            fetch(ROOT / record["path"], record["url"], record["sha256"])
            print(record["path"], flush=True)
    for destination, expected in manifest["model_assets_sha256"].items():
        subdir, asset = Path(destination).parts[-2:]
        if subdir == "codebert" and not args.training:
            continue
        revision = manifest["model_revisions"][MODEL_KEYS[subdir]]
        fetch(ROOT / destination, f"https://huggingface.co/{REPOS[subdir]}/resolve/{revision}/{asset}", expected)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
