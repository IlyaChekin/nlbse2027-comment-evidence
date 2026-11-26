"""Package reproducible follow-up experiments and the standalone weights."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOP = ROOT.parent
SUBMISSION = ROOT / "submission"
PAPER = ROOT / "paper"
PORTFOLIO = ROOT / "portfolio"
FULL = PORTFOLIO / "runs/full/codet5p"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    report = read(HERE/"analysis.json")
    sources = {"frozen": FULL/"portable",
               "fixed384": HERE/"training/fixed384", "mixed": HERE/"training/mixed",
               "cross_project": HERE/"project/checkpoint"}
    hashes = {name:sha256(path/"model.safetensors") for name,path in sources.items()}
    for name,digest in report["continuation"]["frozen"]["weights_sha256"].items():
        if digest!=hashes[name]:
            raise ValueError(f"Checkpoint changed for {name}")
    if hashes["cross_project"]!=read(HERE/"project/training.json")["checkpoint"]:
        raise ValueError("The project checkpoint changed")
    if hashes["frozen"]!=report["router"]["frozen"]["checkpoint_sha256"]:
        raise ValueError("The routing checkpoint changed")
    manifest = dict(weights_sha256=hashes, router_sha256=sha256(HERE/"router/weights.npz"),
                    dataset_revision=read(ROOT/"data/manifest.json")["dataset_revision"],
                    public_policy=report["continuation"]["frozen"]["preferred_384"],
                    project_rows=report["project_split"]["splits"]["heldout"]["rows"],
                    output_order=["vulnerability","MAT"], supplement="nlbse/paper/context_followup.pdf")
    (HERE/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    paths = {
        *(p for p in HERE.rglob("*") if p.is_file() and p.suffix in
          {".py",".sh",".json",".csv",".npy",".npz",".log",".md",".safetensors",".txt"}),
        ROOT/"STATUS.md", ROOT/"data/manifest.json", ROOT/"data/split_manifest.csv.gz",
        *(ROOT/f"{name}.py" for name in ("fetch_inputs","pipeline","comment_spans","continuation_select",
                                        "score_runtime","round2_report","cuda_launch")),
        *(ROOT/"round4"/n for n in ("compare.py","windows.py")),
        *(ROOT/"round5"/n for n in ("test_context.py","cache_context.py","context_views.py","context_policy.py",
                                     "select_context.py","analysis.json")),
        *(PORTFOLIO/n for n in ("t5_entry.py","neural.py","protocol.py","analyze.py","fetch_backbones.sh",
                                "backbones.sha256","backbone_weights.sha256")),
        PORTFOLIO/"upstream/CodeT5-LICENSE", PORTFOLIO/"upstream/SOURCES.md",
        PAPER/"build_context_followup.py", PAPER/"build_round2.py", PAPER/"context_followup.pdf",
        *PAPER.glob("context_followup/*.tex"), *PAPER.glob("context_followup/*.log"),
        TOP/"verification/requirements-resolved.txt",
        *sources["frozen"].iterdir(),
    }
    for ref in ("public/skip_positive","budget/cap384_cover64"):
        directory = ROOT/"round5/runs"/ref
        paths.update(directory/n for n in ("metrics.json","predictions.csv"))
    output = SUBMISSION/"NLBSE2027_context_followup.zip"
    pending = output.with_suffix(".zip.part")
    stored = {".safetensors",".gz",".npz"}
    with zipfile.ZipFile(pending,"w") as archive:
        for path in sorted(paths):
            archive.write(path,path.relative_to(TOP),
                          compress_type=zipfile.ZIP_STORED if path.suffix in stored else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(pending) as archive:
        wrong = archive.testzip()
        if wrong:
            raise ValueError(f"Archive CRC failed for {wrong}")
    pending.replace(output)
    output.with_suffix(".sha256").write_text(f"{sha256(output)}  {output.name}\n")
    print(output,output.stat().st_size)


if __name__=="__main__":
    main()
