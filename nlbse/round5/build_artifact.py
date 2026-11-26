"""Assemble the standalone context classifier and its reproducible study."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOP = ROOT.parent
SUBMISSION = ROOT / "submission"
PORTFOLIO = ROOT / "portfolio"
FULL = PORTFOLIO / "runs/full/codet5p"
PORTABLE = FULL / "portable"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    frozen = json.loads((HERE/"selected_inference.json").read_text())
    report = json.loads((HERE/"analysis.json").read_text())
    weights = PORTABLE/"model.safetensors"
    if sha256(weights) != frozen["checkpoint_sha256"]:
        raise ValueError("The packaged checkpoint differs from the evaluated checkpoint")
    if report["primary"] != frozen["config"]["name"] or report["rows"] != 30655:
        raise ValueError("The frozen policy and validated results disagree")
    selected = report["results"][report["primary"]]
    manifest = HERE/"artifact_manifest.json"
    manifest.write_text(json.dumps(dict(policy=frozen["config"], labels=["vulnerability", "MAT"],
                                        thresholds=frozen["thresholds"], weights_sha256=sha256(weights),
                                        data_revision=json.loads((ROOT/"data/manifest.json").read_text())["dataset_revision"],
                                        mean_F1=selected["scores"]["mean"]["f1"],
                                        public_rows=report["rows"],
                                        paper="nlbse/paper/context.pdf"), indent=2)+"\n")
    paths = {
        *(p for p in HERE.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".json", ".csv", ".npz", ".log", ".md"}),
        ROOT/"STATUS.md", ROOT/"data/manifest.json", ROOT/"data/split_manifest.csv.gz",
        *(ROOT/f"{name}.py" for name in ("fetch_inputs", "pipeline", "comment_spans", "continuation_select",
                                        "score_runtime", "round2_report", "cuda_launch")),
        *(ROOT/"round4"/name for name in ("compare.py", "windows.py", "public_results.json")),
        *(PORTFOLIO/name for name in ("t5_entry.py", "neural.py", "protocol.py", "analyze.py",
                                     "fetch_backbones.sh", "backbones.sha256", "backbone_weights.sha256")),
        PORTFOLIO/"upstream/CodeT5-LICENSE", PORTFOLIO/"upstream/SOURCES.md",
        *PORTABLE.iterdir(), *(FULL/name for name in ("config.json", "training.json")),
        *(FULL/"portable_results/comment/public"/name for name in ("metrics.json", "predictions.csv")),
        ROOT/"paper/build_context.py", ROOT/"paper/build_round2.py", ROOT/"paper/context.pdf",
        *ROOT.glob("paper/context/*.tex"), *ROOT.glob("paper/context/*.log"),
        ROOT/"verification/official_rules.ipynb", ROOT/"third_party/CodeBERT_baseline.ipynb",
        TOP/"verification/requirements-resolved.txt", TOP/"verification/bootstrap_gpu.sh",
    }
    output = SUBMISSION/"NLBSE2027_context_study.zip"
    pending = output.with_suffix(".zip.part")
    stored = {".safetensors", ".gz", ".npz"}
    with zipfile.ZipFile(pending, "w") as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(TOP),
                          compress_type=zipfile.ZIP_STORED if path.suffix in stored else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(pending) as archive:
        damaged = archive.testzip()
        if damaged:
            raise ValueError(f"Archive CRC failed for {damaged}")
    pending.replace(output)
    output.with_suffix(".sha256").write_text(f"{sha256(output)}  {output.name}\n")
    print(output, output.stat().st_size)


if __name__ == "__main__":
    main()
