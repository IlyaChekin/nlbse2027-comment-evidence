"""Package the audited research and the independently runnable local tools."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOP = ROOT.parent
ROUND4 = ROOT / "round4"
SUBMISSION = ROOT / "submission"
CHECKPOINT = SUBMISSION / "entry_checkpoint"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    report = json.loads((HERE / "analysis.json").read_text())
    key = "score_candidate"
    chosen = report["public"]["models"][key]
    frozen = json.loads((ROUND4 / "entry_config.json").read_text())
    if chosen["thresholds"] != frozen["thresholds"] or report["public"]["rows"] != 30655:
        raise ValueError("The working configuration differs from the evaluated release")
    weights = CHECKPOINT / "model.safetensors"
    if sha256(weights) != frozen["model"].rsplit(" ", 1)[-1]:
        raise ValueError("The checkpoint differs from the frozen weight digest")
    sparse = HERE / "runs/full/sparse_roles/model.joblib"
    portable = HERE / "runs/full/codet5p/portable"
    t5 = report["public"]["codet5p"]
    t5_choice = json.loads((portable / "frozen.json").read_text())
    if t5_choice != t5["frozen"]:
        raise ValueError("The portable CodeT5+ config differs from the evaluated release")
    t5_selected = t5[f"t5_{t5_choice['mode']}"]
    score_key = "formula_using_local_time_and_analytic_gflops"
    finalists = {"codebert": chosen, "codet5p": t5_selected}
    paths = [
        ROOT / "README.md", ROOT / "STATUS.md", ROOT / "data/manifest.json",
        ROOT / "data/split_manifest.csv.gz", ROOT / "data/data_audit.json",
        *ROOT.glob("*.py"), *ROOT.glob("configs/*.sh"),
        *ROOT.glob("round3/*.py"), *ROUND4.glob("*.py"), *ROUND4.glob("*.json"),
        *ROUND4.glob("*.csv"), *ROUND4.glob("*.npz"), *ROUND4.glob("*.log"),
        *ROUND4.glob("entry_verify/*/metrics.json"), *ROUND4.glob("entry_verify/*/predictions.csv"),
        *ROUND4.glob("entry_verify/*.log"), *HERE.glob("*.py"), *HERE.glob("*.md"),
        *HERE.glob("*.sh"), *HERE.glob("*.sha256"), *HERE.glob("*.csv"), *HERE.glob("*.log"), HERE / "analysis.json",
        *(path for path in (HERE / "runs").rglob("*")
          if path.is_file() and path.suffix in {".json", ".csv", ".log", ".sha256"}),
        sparse, *portable.iterdir(), *HERE.glob("upstream/*"),
        *CHECKPOINT.iterdir(), *ROOT.glob("paper/portfolio/*.tex"),
        *ROOT.glob("paper/portfolio/*.log"), ROOT / "paper/build_portfolio.py", ROOT / "paper/build_round2.py",
        ROOT / "paper/portfolio.pdf", TOP / "verification/requirements-resolved.txt",
        TOP / "verification/bootstrap_gpu.sh", ROOT / "verification/official_rules.ipynb",
        ROOT / "verification/official_rules.sha256", ROOT / "third_party/CodeBERT_baseline.ipynb",
    ]
    manifest = SUBMISSION / "portfolio_manifest.json"
    manifest.write_text(json.dumps(dict(
        data_revision=json.loads((ROOT / "data/manifest.json").read_text())["dataset_revision"],
        public_rows=report["public"]["rows"], label_order=["vulnerability", "MAT"],
        comment_window=dict(weights_sha256=sha256(weights), config="nlbse/round4/entry_config.json",
                            mean_f1=chosen["scores"]["mean"]["f1"]),
        sparse=dict(weights_sha256=sha256(sparse), config="nlbse/portfolio/runs/full/sparse_roles/metrics.json",
                    mean_f1=report["public"]["sparse"]["scores"]["mean"]["f1"]),
        codet5p=dict(weights_sha256=sha256(portable/"model.safetensors"),
                     config=str((portable/"frozen.json").relative_to(TOP)), mode=t5_choice["mode"],
                    mean_f1=t5_selected["scores"]["mean"]["f1"]),
        best_local_gpu_formula=max(finalists, key=lambda name: finalists[name][score_key]),
        input="CSV column Function", paper="nlbse/paper/portfolio.pdf"
    ), indent=2) + "\n")
    output = SUBMISSION / "NLBSE2027_research_portfolio.zip"
    pending = output.with_suffix(".zip.part")
    stored = {".safetensors", ".joblib", ".gz", ".npz"}
    with zipfile.ZipFile(pending, "w") as archive:
        for path in sorted({*paths, manifest}):
            archive.write(path, path.relative_to(TOP),
                          compress_type=zipfile.ZIP_STORED if path.suffix in stored else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(pending) as archive:
        corrupted = archive.testzip()
        if corrupted:
            raise ValueError(f"Archive check failed at {corrupted}")
    pending.replace(output)
    output.with_suffix(".sha256").write_text(f"{sha256(output)}  {output.name}\n")
    print(output, output.stat().st_size)


if __name__ == "__main__":
    main()
