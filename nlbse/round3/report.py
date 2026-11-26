"""Verify round-3 coverage and write matched differences and cost comparisons."""
import csv
import json
from pathlib import Path

import numpy as np
from transformers import RobertaTokenizerFast

from protocol import OUT, SIX, dataset
from pipeline import SPLIT, metric, write_json
from round2_report import bootstrap_delta
from ideas import LEXEME

RESULTS = Path(__file__).resolve().parent
NAMES = {
    "six_control": "Контроль · 6 блоков",
    "length_thresholds": "1. Пороги по длине",
    "twelve_control": "Контроль · 12 блоков",
    "extra_context": "2. Начало и конец длинной функции",
    "split_roles": "3. Ветви кода и комментариев",
    "early_exit": "4. Ранний выход · исходная сетка",
    "early_exit_followup": "4. Ранний выход · уточнённая сетка",
}
COMPARATORS = {
    "length_thresholds": "six_control",
    "extra_context": "twelve_control",
    "split_roles": "six_control",
    "early_exit": "twelve_control",
    "early_exit_followup": "twelve_control",
}


def read_run(name, expected_row_ids, label):
    run = OUT / name
    scores = json.loads((run / "metrics.json").read_text())
    with (run / "predictions.csv").open(newline="") as file:
        pred = list(csv.DictReader(file))
    if [int(r["row"]) for r in pred] != expected_row_ids:
        raise ValueError(f"Misordered predictions or inconsistent row coverage for {name}")
    bits = np.array([[int(r["vulnerability"]), int(r["MAT"])] for r in pred])
    if not np.isin(bits, (0, 1)).all() or metric(label, bits) != scores["scores"]:
        raise ValueError(f"Prediction checks failed for {name}")
    p = np.array([[float(r["p_vulnerability"]), float(r["p_MAT"])] for r in pred])
    if not np.isfinite(p).all():
        raise ValueError(f"Nonfinite probabilities for {name}")
    return scores, bits, p


def context_ablation(all_results, texts, gold):
    tokenizer = RobertaTokenizerFast.from_pretrained(SIX, local_files_only=True)
    lengths = np.array([len(row) for begin in range(0, len(texts), 1024)
                        for row in tokenizer(texts[begin:begin+1024], verbose=False)["input_ids"]])
    scopes = {"all": lengths >= 0, "within_512": lengths <= 512, "over_512": lengths > 512}
    variants = ("twelve_control", "extra_context")
    comparisons = {}
    for mode in variants:
        for th_source in variants:
            threshold_values = np.array(all_results[th_source][0]["config"]["thresholds"])
            predictions = all_results[mode][2] > threshold_values
            comparisons[f"{mode}_with_{th_source}_thresholds"] = {
                name: dict(rows=int(mask.sum()), scores=metric(gold[mask], predictions[mask]))
                for name, mask in scopes.items()
            }
    write_json(RESULTS / "context_ablation.json", comparisons)


def main():
    texts, y, rows, nfit, is_cal = dataset()
    gold = y[nfit:][~is_cal]
    ids = rows[nfit:][~is_cal].tolist()
    all_results = {name: read_run(name, ids, gold) for name in NAMES}
    data = []
    for name, (row, _, _) in all_results.items():
        cost = row["cost"]
        data.append(dict(name=name, **{f"{label}_{field}": value for label, metrics in row["scores"].items()
                                      for field, value in metrics.items()}, gflops=cost["profiler_counted_gflops"],
                         wall_seconds=cost["wall_seconds"], parameters=cost["parameters"],
                         early_rows=cost["early_rows"]))
    with (RESULTS / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    with SPLIT.open(newline="") as file:
        tune_hashes = {int(r["row"]): r["function_sha256"] for r in csv.DictReader(file) if r["partition"] == "tune"}
    groups = {}
    group_indices = [groups.setdefault(tune_hashes[i], len(groups)) for i in ids]
    differences = {}
    for name, baseline in COMPARATORS.items():
        before, after = all_results[baseline][1], all_results[name][1]
        differences[name] = dict(comparator=baseline, delta95=bootstrap_delta(gold, before, after, group_indices),
                                 label_changes=dict(zip(("vulnerability", "MAT"),
                                                        (before != after).sum(axis=0).tolist())))
    write_json(RESULTS / "paired_intervals.json", dict(draws=2000, clusters=len(groups), comparisons=differences))
    has_comment = np.array([any(m[0].startswith(("//", "/*")) for m in LEXEME.finditer(text))
                            for text, calibration in zip(texts[nfit:], is_cal, strict=True) if not calibration])
    write_json(RESULTS / "split_role_diagnostic.json", {
        section: dict(rows=int(mask.sum()), positives=gold[mask].sum(axis=0).tolist(),
                      scores={name: metric(gold[mask], all_results[name][1][mask])
                              for name in ("six_control", "split_roles")})
        for section, mask in (("comment_detected_in_full_text", has_comment), ("no_comment_detected", ~has_comment))
    })
    context_ablation(all_results, [text for text, in_cal in zip(texts[nfit:], is_cal, strict=True) if not in_cal], gold)
    lines = ["# NLBSE · четыре идеи",
             "", f"Всего {len(ids):,} одинаковых строк внутренней выборки. Подтверждение качества второго окна на публичной выборке хранится отдельно.",
             "", "| Вариант | Vul F1 | MAT F1 | Средний F1 | GFLOPs | Wall time, с |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in data:
        lines.append(f"| {NAMES[r['name']]} | {r['vulnerability_f1']:.4f} | {r['MAT_f1']:.4f} | "
                     f"{r['mean_f1']:.4f} | {r['gflops']:,.0f} | {r['wall_seconds']:.1f} |")
    lines += ["", "Время включает профилированную работу с подготовленными токенами на RTX 4090. "
                    "Профилирование выполнялось окнами. Оценки времени относятся к этому прогону.",
              "", "Бутстрэп использует группы точных совпадений текста. Интервалы и протокол порогов приложены."]
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
