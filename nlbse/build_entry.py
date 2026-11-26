"""Build a self-contained local NLBSE entry with exactly the frozen weights."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOP = ROOT.parent
SUBMISSION = ROOT / "submission"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def add_files(archive, paths):
    for path in sorted(set(paths)):
        compression = zipfile.ZIP_STORED if path.suffix in {".safetensors", ".gz"} else zipfile.ZIP_DEFLATED
        archive.write(path, path.relative_to(TOP), compress_type=compression)


def main():
    selection = json.loads((ROOT / "reports/round2_selection.json").read_text())
    selected = selection["candidates"][selection["winner"]]
    checkpoint = SUBMISSION / "entry_checkpoint"
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    settings = json.loads((checkpoint / "config.json").read_text())
    verification = json.loads((ROOT / "reports/round2_entry_smoke.json").read_text())
    weights_hash = sha256(weights)
    if verification["model"] != selection["winner"] or verification["weights_sha256"] != weights_hash:
        raise ValueError("The exported weights differ from the independently tested checkpoint.")
    thresholds = ROOT / "reports" / f"{selection['winner']}_frozen_thresholds.json"
    shutil.copyfile(thresholds, SUBMISSION / "entry_thresholds.json")
    receipt = dict(
        architecture="shared CodeBERT encoder with two sigmoid outputs",
        output_order=["vulnerability", "MAT"],
        encoder_blocks=settings["num_hidden_layers"],
        source_run=selected["checkpoint"], bucket_window=2048, batch_size=32, max_length=512,
        thresholds=selected["thresholds"], weights_sha256=weights_hash,
    )
    card = SUBMISSION / "entry_manifest.json"
    card.write_text(json.dumps(receipt, indent=2) + "\n")

    files = [
        *ROOT.glob("*.py"), *ROOT.glob("configs/*.sh"),
        ROOT / "README.md", ROOT / "STATUS.md", ROOT / "data/manifest.json",
        ROOT / "data/split_manifest.csv.gz", ROOT / "data/data_audit.json",
        *ROOT.glob("reports/round2_*"), *ROOT.glob("reports/*pilot_selection.json"),
        *ROOT.glob("reports/*frozen_thresholds.json"), ROOT / "reports/packing_tune_comparison.json",
        ROOT / "paper/build_round2.py", ROOT / "paper/round2.pdf", *ROOT.glob("paper/round2/*.tex"),
        ROOT / "verification/official_rules.sha256", ROOT / "verification/official_rules.ipynb", ROOT / "verification/run_colab.md", ROOT / "third_party/CodeBERT_baseline.ipynb",
        *checkpoint.iterdir(), ROOT / "submission/entry_thresholds.json", card,
        TOP / "verification/requirements-resolved.txt", TOP / "verification/bootstrap_gpu.sh",
    ]
    for run in ROOT.glob("runs/*"):
        if run.is_dir():
            files.extend(p for p in run.iterdir() if p.is_file() and p.suffix in {".csv", ".json", ".log"})
    output = SUBMISSION / "NLBSE2027_entry_v2.zip"
    partial = output.with_suffix(".zip.part")
    with zipfile.ZipFile(partial, "w") as archive:
        add_files(archive, files)
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive check failed: {bad}")
    partial.replace(output)
    (SUBMISSION / "NLBSE2027_entry_v2.sha256").write_text(f"{sha256(output)}  {output.name}\n")
    print(output, output.stat().st_size, receipt["weights_sha256"], flush=True)


if __name__ == "__main__":
    main()
