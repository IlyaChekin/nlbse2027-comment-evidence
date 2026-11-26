"""Audit paired predictions and generate the extra-context study results."""
import csv
import hashlib
import json

import numpy as np
from transformers import AutoTokenizer

from cache_context import HERE, PORTABLE
from context_policy import Policy, THRESHOLDS
from context_views import prepare
from analyze import changes, group_codes, predict_bits, verify
from pipeline import PUBLIC, SPLIT, load, metric, write_json
from round2_report import bootstrap_delta
from select_context import apply_cached
from t5_entry import FULL

DISPLAY = {
    "head": "Первые 512 токенов",
    "reference": "Прежнее окно комментария 128",
    "skip_positive": "128 с пропуском положительных MAT",
    "cover64": "64 с пропуском положительных MAT",
    "extend_on_long": "Первое окно до 768",
    "seen_control": "Уже прочитанный фрагмент 128",
    "equal_length_tail": "Хвост 128 на тех же строках",
    "all_long_tail": "Хвост 512 на всех длинных",
}


def read(path):
    return json.loads(path.read_text())


def name_of(name):
    if name.startswith("cap"):
        head, view = name[3:].split("_", 1)
        return f"Первое окно {head}" if view == "head" else f"Первое {head} и окно комментария {view[5:]}"
    return DISPLAY.get(name, name)


def dec(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def train_checks(freeze, budget):
    dims = read(HERE/"cache_config.json")["dimensions"]
    with np.load(HERE/"holdout_scores.npz") as stored:
        data = dict(stored)
    y, cal = data["gold"], data["calibration"]
    reference = freeze["candidates"]["reference"]
    control = apply_cached(Policy(**reference["config"]), data, dims)[0] > THRESHOLDS
    matches = {}
    for key, row in freeze["candidates"].items():
        p = apply_cached(Policy(**row["config"]), data, dims)[0] > THRESHOLDS
        matches[key] = (p != control).sum(axis=0).tolist()
    allowed = {}
    all_candidates = {**freeze["candidates"], **budget["candidates"]}
    minimum = reference["scores"]["mean"]["f1"]
    for key in (*freeze["shortlisted"], *budget["shortlisted"]):
        row = all_candidates[key]
        if row["scores"]["mean"]["f1"] >= minimum:
            allowed[key] = row
    preferred = max(allowed, key=lambda k: allowed[k]["projected_public_cost_formula"])
    return preferred, matches, all_candidates


def audit():
    freeze = read(HERE/"selection.json")
    budget = read(HERE/"budget_selection.json")
    preferred, holdout_changes, trained = train_checks(freeze, budget)

    texts, y, fields = load(PUBLIC, audit=True)
    n = len(y)
    ids = list(range(n))
    groups = group_codes(zip(fields["Projectname"], fields["Filepath"]))
    results, bits = {}, {}
    for directory, policies in (("public", freeze["shortlisted"]), ("diagnostic", freeze["diagnostic"]),
                                ("budget", budget["shortlisted"])):
        for key in policies:
            result = HERE/"runs"/directory/key
            receipt = read(result/"metrics.json")
            if tuple(receipt["thresholds"]) != THRESHOLDS:
                raise ValueError(f"{key} has different decision thresholds")
            prediction = predict_bits(result/"predictions.csv", ids, receipt)
            verify(y, prediction, receipt)
            bits[key] = prediction
            results[key] = dict(**receipt, scope=directory)

    previous = FULL/"portable_results/comment/public"
    old_result = read(previous/"metrics.json")
    old = predict_bits(previous/"predictions.csv", ids, old_result)
    verify(y, old, old_result)
    historical_changed = (old != bits["reference"]).sum(axis=0).tolist()
    redundant_changed = (bits["reference"] != bits["skip_positive"]).sum(axis=0).tolist()
    if any(historical_changed):
        raise ValueError(f"Reference replay changed binary decisions: {historical_changed}")

    ref_f1 = results["reference"]["scores"]["mean"]["f1"]
    primary = preferred
    if results[preferred]["scores"]["mean"]["f1"] < ref_f1 - 0.001:
        primary = "reference" if any(redundant_changed) else "skip_positive"

    tokenizer = AutoTokenizer.from_pretrained(PORTABLE, local_files_only=True)
    inputs = prepare(texts, tokenizer, ("head",))
    with SPLIT.open(newline="") as file:
        train_hashes = {r["function_sha256"] for r in csv.DictReader(file)}
    novel = np.asarray([hashlib.sha256(text.encode()).hexdigest() not in train_hashes for text in texts])
    tracked = ("head", "reference", "skip_positive", "cover64", primary)
    masks = dict(short=inputs.lengths <= 512, long=inputs.lengths > 512,
                 omitted_comment=inputs.missing, crosses_boundary=inputs.crossing,
                 wholly_omitted=inputs.missing & ~inputs.crossing, omitted_with_cue=inputs.missing & inputs.cue,
                 exact_novel=novel)
    cohort = {name: dict(rows=int(mask.sum()), positives=y[mask].sum(axis=0).tolist(),
                         scores={key: metric(y[mask], bits[key][mask]) for key in dict.fromkeys(tracked)})
              for name, mask in masks.items()}
    del texts, inputs
    comparisons = {"reference_minus_head": (bits["head"], bits["reference"])}
    for k in ("skip_positive", "cover64", "extend_on_long", "seen_control", "equal_length_tail", "all_long_tail", primary):
        if k in bits and k != "reference":
            comparisons[k+"_minus_reference"] = bits["reference"], bits[k]
    left, right = budget["shortlisted"]
    comparisons[right+"_minus_"+left] = bits[left], bits[right]
    paired = {key: dict(interval=bootstrap_delta(y, a, b, groups), changes=changes(y,a,b))
              for key, (a,b) in comparisons.items()}

    write_json(HERE/"selected_inference.json", dict(
        config=trained[primary]["config"], thresholds=THRESHOLDS,
        checkpoint_sha256=read(HERE/"cache_config.json")["checkpoint"],
        selection=dict(holdout_preference=preferred, public_quality_guard=0.001)))
    report = dict(rows=n, primary=primary, preferred=preferred, results=results, cohorts=cohort,
                  paired=paired, holdout_changes_vs_reference=holdout_changes,
                  historical_replay_changed_labels=historical_changed,
                  redundant_pass_skip_changed_labels=redundant_changed,
                  diagnostic_policies=freeze["diagnostic"], shorter_budget_policies=budget["shortlisted"])
    write_json(HERE/"analysis.json", report)
    with (HERE/"public_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("policy", "vulnerability_P", "vulnerability_R", "vulnerability_F1",
                         "MAT_P", "MAT_R", "MAT_F1", "mean_F1", "extra_windows",
                         "replaced_first_windows", "analytic_GFLOPs", "forward_seconds", "wall_seconds", "local_formula"))
        for key, r in results.items():
            c = r["cost"]
            writer.writerow((key, *(r["scores"][label][stat] for label in ("vulnerability", "MAT")
                                    for stat in ("precision", "recall", "f1")), r["scores"]["mean"]["f1"],
                             c["extra_windows"], c["replaced_first_windows"], c["analytic_gflops"],
                             c["forward_seconds"], c["wall_seconds"], r["local_formula"]))
    return report


def markdown(r):
    rows = [
        "# Выбор дополнительного контекста для CodeT5+",
        f"Выбран вариант «{name_of(r['primary'])}». Воспроизведение исходного инструмента "
        f"совпало по всем {2*r['rows']:,} бинарным решениям.",
        "## Публичные результаты",
    ]
    table = ["| Политика | Vulnerability F1 | MAT F1 | F1 | Второй проход | GFLOPs | Локальный балл |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, result in r["results"].items():
        c = result["cost"]
        cells = [name_of(key), *(dec(result["scores"][m]["f1"]) for m in ("vulnerability","MAT","mean")),
                 str(c["extra_windows"]), dec(c["analytic_gflops"], 0), dec(result["local_formula"])]
        table.append("| " + " | ".join(cells) + " |")
    rows.append("\n".join(table))
    omitted = r["cohorts"]["omitted_comment"]
    crossing = r["cohorts"]["crosses_boundary"]
    pair = r["paired"]["reference_minus_head"]["changes"]["MAT"]
    rows.append(f"За пределами первого окна оказались комментарии у {omitted['rows']} функций. "
                f"У {crossing['rows']} из них хотя бы один комментарий пересёк границу окна. "
                f"Старое правило прочитало пропущенные фрагменты и вернуло {pair['extra_tp']} верных MAT-решений "
                f"при {pair['extra_fp']} дополнительных ложных срабатываниях.")
    rows.append(f"Пропуск уже положительных MAT сократил число вторых проходов с "
                f"{r['results']['reference']['cost']['extra_windows']} до {r['results']['skip_positive']['cost']['extra_windows']}. "
                f"Изменения бинарных решений по двум меткам: {r['redundant_pass_skip_changed_labels']}.")
    results = r["results"]
    rows.append(f"На тех же строках и при той же длине повторное чтение уже виденного фрагмента дало "
                f"{dec(results['seen_control']['scores']['mean']['f1'])} F1, а чтение хвоста дало "
                f"{dec(results['equal_length_tail']['scores']['mean']['f1'])}. Удлинение первой части до 768 "
                f"для длинных функций дало {dec(results['extend_on_long']['scores']['mean']['f1'])} F1.")
    rows.append("Медиана трёх прогретых прогонов используется для основных вариантов. "
                "Диагностические контроли прошли один раз. GFLOPs рассчитаны по формам плотных операций; "
                "время на RTX 4090 и оценки по формуле относятся к публичному сплиту.")
    rows.append("Варианты сравнивались на 18 444 строках внутреннего holdout из официального train. "
                "Выбор заморожен перед публичным прогоном. Все первичные сравнения сохраняют одни пороги. "
                "Метрики, маршрутизация, полный тайминг, интервалы и срезы лежат в [analysis.json](analysis.json).")
    return "\n\n".join(rows) + "\n"


def main():
    report = audit()
    (HERE/"results.md").write_text(markdown(report))
    print("Verified", report["rows"], "public rows. Selected", report["primary"], flush=True)


if __name__ == "__main__":
    main()
