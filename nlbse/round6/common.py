"""Shared paths, reproducible sampling and export for the three context experiments."""
from pathlib import Path
import hashlib
import sys

import numpy as np
from safetensors.torch import save_model
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "round5"), str(ROOT / "portfolio"), str(ROOT)]
from cache_context import PORTABLE, dimensions, forward_rows, load as load_checkpoint
from context_policy import THRESHOLDS, Policy
from context_views import prepare
from pipeline import PUBLIC, SPLIT, TRAIN, load, metric, select_indices, write_json
from protocol import dataset, save
from test_context import run_policy
from score_runtime import competition_formula

SEED = 42
VARIANTS = (
    Policy("head384", head_limit=384),
    Policy("comment384", comment="cover64", gate="skip", head_limit=384),
    Policy("comment512", comment="cover128", gate="skip"),
)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def stratified(rows, gold, amount, seed=SEED):
    rows = np.array(rows)
    if len(rows) <= amount:
        return np.sort(rows)
    labels = gold[rows]
    selected, _ = train_test_split(rows, train_size=amount, random_state=seed,
                                   stratify=2*labels[:,0]+labels[:,1])
    return np.sort(selected)


def export(model, tokenizer, folder):
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(folder)
    model.encoder.config.save_pretrained(folder)
    pending = folder / "model.part.safetensors"
    save_model(model.eval().half(), pending)
    pending.replace(folder / "model.safetensors")
    return sha256(folder / "model.safetensors")
