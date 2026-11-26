"""Evaluate frozen MAT and a human-supervised probe on tag-free annotated comments."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from external_data import load_source, normalize, text_hash

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round6"))
from common import PORTABLE, SEED, THRESHOLDS, TRAIN, load_checkpoint, sha256, write_json
from analyze import group_codes
from round2_report import bootstrap_delta
from router import first_pass
from windows import comment_start

CAPS = (512, 256, 128)
WIDTH = 126


def tokenizer_views(examples, tokenizer):
    token_ids, target = [], []
    for off in range(0, len(examples), 512):
        batch = examples[off:off+512]
        encoded = tokenizer([row.text for row in batch], add_special_tokens=False,
                            return_offsets_mapping=True, verbose=False)
        for row, ids, offsets in zip(batch, encoded["input_ids"], encoded["offset_mapping"], strict=True):
            positions = [i for i, (a, b) in enumerate(offsets) if a < row.end and b > row.start and b > a]
            if not positions:
                raise ValueError(f"Failed to locate annotated comment {row.identifier}")
            token_ids.append(ids)
            target.append(positions)
    return token_ids, target


def infer(model, tokenizer, seqs):
    inputs = SimpleNamespace(views={"head": [[tokenizer.bos_token_id, *row, tokenizer.eos_token_id]
                                            for row in seqs]}, pad=tokenizer.pad_token_id)
    p, embedding, _ = first_pass(model, inputs)
    return p[:, 1], embedding


def measures(gold, p):
    if not len(gold) or not gold.any() or gold.all():
        return {"positives": int(gold.sum()), "rows": len(gold)}
    hard = p > .5
    precision, recall, f1, _ = precision_recall_fscore_support(gold, hard, average="binary", zero_division=0)
    return dict(precision=float(precision), recall=float(recall), f1=float(f1),
                average_precision=float(average_precision_score(gold, p)),
                positive_predictions=int(hard.sum()), rows=len(gold), positives=int(gold.sum()))


def paired(gold, a, b, clusters):
    a, b = a > .5, b > .5
    if not (gold.any() and not gold.all()):
        return None
    result = bootstrap_delta(np.repeat(gold[:, None], 2, axis=1), np.repeat(a[:, None], 2, axis=1),
                             np.repeat(b[:, None], 2, axis=1), group_codes(clusters))
    return dict(delta_f1_95=result["MAT"],
                recovered_positive=int((~a & b & gold).sum()),
                lost_positive=int((a & ~b & gold).sum()),
                added_false_positive=int((~a & b & ~gold).sum()),
                cleared_false_positive=int((a & ~b & ~gold).sum()))


def probes(train, embeddings, labels):
    # The same fixed classifiers are used in every held-out fold. A distinct
    # lexical control measures whether the frozen representation is needed.
    feature_model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=.1, class_weight="balanced", random_state=SEED, max_iter=500))
    lexical_model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=60000),
                                   LogisticRegression(C=1., class_weight="balanced", random_state=SEED, max_iter=500))
    return feature_model.fit(embeddings, labels), lexical_model.fit(train, labels)


def blend(head, second):
    return np.maximum(head, second)


def contextual_views(tokens, target, row_ids, cap):
    heads, matched, full, compact = [], [], [], []
    body = cap-2
    for i in row_ids:
        ids, positions = tokens[i], target[i]
        omitted = [j for j in positions if j >= body]
        start = comment_start(len(ids), omitted, WIDTH)
        heads.append(ids[:body])
        matched.append(ids[max(0, body-WIDTH):body])
        full.append(ids[start:start+WIDTH])
        compact.append([ids[j] for j in omitted if start <= j < start+WIDTH])
    if any(not v for v in compact):
        raise ValueError("An omitted window lost all annotated comment tokens")
    return {"head": heads, "seen_control": matched, "full": full, "compact": compact}


def evaluate(examples, tokenizer, model, gold, tokens, target, probe_models, output, name):
    group = np.asarray([row.group for row in examples])
    comments = [row.comment for row in examples]
    native_results = {}
    for cap in CAPS:
        body = cap-2
        rows = np.array([i for i, positions in enumerate(target) if positions[-1] >= body], dtype=int)
        if not len(rows):
            native_results[str(cap)] = {"rows": 0}
            continue
        print(name, cap, "eligible", len(rows), "positive", int(gold[rows].sum()), flush=True)
        views = contextual_views(tokens, target, rows, cap)
        predictions, vectors = {}, {}
        for view, seq in views.items():
            predictions[view], vectors[view] = infer(model, tokenizer, seq)
        head = predictions["head"]
        scores = {"mat_head": head, "mat_plus_seen": blend(head, predictions["seen_control"]),
                  "mat_plus_full": blend(head, predictions["full"]),
                  "mat_plus_compact": blend(head, predictions["compact"])}
        probe_head, probe_seen, probe_packed = (np.empty(len(rows), dtype=np.float32) for _ in range(3))
        lexical = np.empty(len(rows), dtype=np.float32)
        for test_group, (probe, lex) in probe_models.items():
            local = group[rows] == test_group if name == "cppsatd" else np.ones(len(rows), dtype=bool)
            if not local.any():
                continue
            probe_head[local] = probe.predict_proba(vectors["head"][local])[:, 1]
            probe_seen[local] = probe.predict_proba(vectors["seen_control"][local])[:, 1]
            probe_packed[local] = probe.predict_proba(vectors["compact"][local])[:, 1]
            decoded = tokenizer.batch_decode([views["compact"][i] for i in np.flatnonzero(local)],
                                              skip_special_tokens=True, clean_up_tokenization_spaces=False)
            lexical[local] = lex.predict_proba(decoded)[:, 1]
        scores.update(probe_head=probe_head, probe_plus_seen=blend(probe_head, probe_seen),
                      probe_plus_compact=blend(probe_head, probe_packed),
                      probe_compact_only=probe_packed, lexical_compact_only=lexical)
        subsets = {"all": np.ones(len(rows), dtype=bool),
                   "fully_omitted": np.array([target[i][0] >= body for i in rows])}
        record = {}
        for subset, keep in subsets.items():
            index = rows[keep]
            labels = gold[index]
            clusters = [examples[i].cluster for i in index]
            paired_keys = (("mat_head", "mat_plus_full"), ("mat_plus_seen", "mat_plus_full"),
                           ("mat_head", "mat_plus_compact"), ("probe_head", "probe_plus_compact"),
                           ("probe_plus_seen", "probe_plus_compact"), ("mat_plus_compact", "probe_plus_compact"))
            suffix = {key: measures(labels, p[keep]) for key, p in scores.items()}
            record[subset] = dict(rows=len(index), positives=int(labels.sum()), scores=suffix,
                                  paired={f"{b}_minus_{a}": paired(labels, scores[a][keep], scores[b][keep], clusters)
                                          for a, b in paired_keys})
        native_results[str(cap)] = record
        output.mkdir(parents=True, exist_ok=True)
        with (output/f"{name}_{cap}_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "group", "cluster", "text_sha256", "gold_human_satd",
                             "target_first_token", "target_last_token", *scores])
            writer.writerows((examples[i].identifier, examples[i].group, examples[i].cluster,
                              text_hash(examples[i].text), int(gold[i]), target[i][0], target[i][-1],
                              *(p[j] for p in scores.values())) for j, i in enumerate(rows))
    return native_results


def main(output, isolate_tesoro=False):
    cpp, cpp_stats = load_source("cppsatd")
    java, java_stats = load_source("tesoro", isolate_tesoro)
    cpp_keys = {normalize(row.comment) for row in cpp}
    java = [row for row in java if normalize(row.comment) not in cpp_keys]
    java_stats["removed_cpp_comment_overlap"] = java_stats["retained"]-len(java)
    java_stats["retained_after_overlap"] = len(java)
    java_stats["positive_after_overlap"] = sum(row.label for row in java)
    official_projects = set(pq.read_table(TRAIN, columns=["Projectname"]).column(0).to_pylist())
    project_overlap = {row.group.casefold() for row in cpp} & {name.casefold() for name in official_projects}
    if project_overlap:
        raise ValueError(f"CppSATD projects overlap the official training set: {project_overlap}")
    print("Filtering:", dict(cpp_stats), dict(java_stats), flush=True)
    model, tokenizer = load_checkpoint(PORTABLE, "cuda")
    cpp_tokens, cpp_target = tokenizer_views(cpp, tokenizer)
    just_comment = [[tokens[i] for i in positions[:WIDTH]]
                    for tokens, positions in zip(cpp_tokens, cpp_target, strict=True)]
    _, cpp_emb = infer(model, tokenizer, just_comment)
    y = np.asarray([row.label for row in cpp])
    fold_groups = np.asarray([row.group for row in cpp])
    text = np.asarray([row.comment for row in cpp])
    models = {}
    for train, test in LeaveOneGroupOut().split(cpp_emb, y, fold_groups):
        group = fold_groups[test[0]]
        print("Fit CppSATD probe holding out", group, "training rows", len(train), flush=True)
        models[group] = probes(text[train], cpp_emb[train], y[train])

    cpp_results = evaluate(cpp, tokenizer, model, y, cpp_tokens, cpp_target, models, output, "cppsatd")
    print("Fit CppSATD probe for transfer to Tesoro", flush=True)
    java_models = {"all": probes(text, cpp_emb, y)}
    java_tokens, java_target = tokenizer_views(java, tokenizer)
    java_y = np.asarray([row.label for row in java])
    java_results = evaluate(java, tokenizer, model, java_y, java_tokens, java_target, java_models, output, "tesoro")
    write_json(output/"analysis.json", dict(
        apparatus=torch.cuda.get_device_name(), checkpoint_sha256=sha256(PORTABLE/"model.safetensors"),
        thresholds=dict(original=THRESHOLDS[1], human_probe=.5), primary_cutoff=512,
        protocol="CppSATD leave-one-project-out supervision. Tesoro transfer from CppSATD only.",
        masking="No MAT tags anywhere in the model input. Gold is the sole annotated comment.",
        tesoro_other_comments_removed=isolate_tesoro,
        cppsatd_projects=sorted(set(fold_groups)), cppsatd_overlap_official_training=sorted(project_overlap),
        filtering={"cppsatd": dict(cpp_stats), "tesoro": dict(java_stats)},
        results={"cppsatd": cpp_results, "tesoro": java_results}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"external")
    parser.add_argument("--isolate-tesoro", action="store_true", help="Blank out other Java comments before tokenization.")
    args = parser.parse_args()
    main(args.output, args.isolate_tesoro)
