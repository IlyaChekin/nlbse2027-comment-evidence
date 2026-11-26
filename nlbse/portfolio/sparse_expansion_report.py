"""Audit classification outputs and cost of the learned sparse NLBSE variants."""
import csv
import hashlib
import json

from analyze import group_codes, predict_bits, verify
from pipeline import PUBLIC, load, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from sparse_expansion import ARMS, BASE, HERE, LINEAR_DISTILLED, OUT, QUANTIZED

NAMES = dict(bm25="BM25 с биграммами", linear="Дообученная линейная часть",
             expansion="Разреженное расширение", distilled="Разреженное расширение с дистилляцией",
             linear_distilled="Линейная модель с дистилляцией",
             distilled_int8="Разреженное расширение INT8")


def read(path):
    return json.loads(path.read_text())


def dec(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def main():
    _, y, fields = load(PUBLIC, audit=True)
    ids = list(range(len(y)))
    groups = group_codes(zip(fields["Projectname"], fields["Filepath"]))
    results, bits = {}, {}
    reference = read(HERE / "analysis.json")["public"]["codet5p"]["t5_comment"]
    gpu_score = reference["formula_using_local_time_and_analytic_gflops"]
    ratio = 61311 / len(y)
    baseline = read(BASE / "public/metrics.json")
    extras = tuple(arm for arm in (LINEAR_DISTILLED, QUANTIZED)
                   if (OUT / "runs" / arm / "public/metrics.json").is_file())
    evaluated = (*ARMS, *extras)
    for name in ("bm25", *evaluated):
        directory = OUT / "runs" / name
        r = read(directory / "public/metrics.json")
        if name == "bm25":
            frozen = baseline
            weights = BASE / "model.joblib"
        else:
            frozen = read(directory / "holdout/metrics.json")
            weights = directory / "model.npz"
        if r["thresholds"] != frozen["thresholds"]:
            raise ValueError(f"Threshold mismatch for {name}")
        if name == QUANTIZED and r["thresholds"] != results["distilled"]["thresholds"]:
            raise ValueError("The INT8 export must keep the FP32 thresholds")
        p = predict_bits(directory / "public/predictions.csv", ids, r)
        verify(y, p, r)
        if name == "bm25":
            verify(y, p, baseline)
        bits[name] = p
        with weights.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        f1 = r["scores"]["mean"]["f1"]
        runtime = r["cost"]["median_seconds"]
        results[name] = dict(**r, sha256=digest, selection_mean_f1=frozen["scores"]["mean"]["f1"],
                             zero_cost_ceiling=competition_formula(f1, 0., 0.),
                             time_only_ceiling=competition_formula(f1, runtime, 0.),
                             simulated_hidden_volume_ceiling=competition_formula(f1, runtime * ratio, 0.))
    pairs = (("bm25", "linear"), ("bm25", "expansion"), ("bm25", "distilled"),
             ("linear", "expansion"), ("expansion", "distilled"))
    if LINEAR_DISTILLED in extras:
        pairs += (("bm25", LINEAR_DISTILLED), ("linear", LINEAR_DISTILLED), (LINEAR_DISTILLED, "distilled"))
    quantization = None
    if QUANTIZED in extras:
        pairs += (("bm25", QUANTIZED), ("distilled", QUANTIZED))
        quantization = dict(changed_labels_by_task=(bits["distilled"] != bits[QUANTIZED]).sum(axis=0).tolist(),
                            weight_reduction=1-results[QUANTIZED]["cost"]["weights_bytes"]
                            /results["distilled"]["cost"]["weights_bytes"])
    comparisons = {f"{b}_minus_{a}": bootstrap_delta(y, bits[a], bits[b], groups) for a, b in pairs}
    with (OUT / "public_table.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("model", "vulnerability_F1", "MAT_F1", "mean_F1", "CPU_seconds", "weights_bytes",
                         "active_features", "zero_cost_ceiling", "time_only_ceiling", "holdout_F1"))
        for name, r in results.items():
            c = r["cost"]
            writer.writerow((name, *(r["scores"][key]["f1"] for key in ("vulnerability", "MAT", "mean")),
                             c["median_seconds"], c["weights_bytes"], c["nnz"],
                             r["zero_cost_ceiling"], r["time_only_ceiling"], r["selection_mean_f1"]))
    write_json(OUT / "analysis.json", dict(rows=len(y), results=results, paired_delta_95=comparisons,
                                          quantization=quantization,
                                          codet5_public_local_formula=gpu_score,
                                          codet5_simulated_hidden_volume=competition_formula(
                                              reference["scores"]["mean"]["f1"],
                                              reference["forward_seconds_median"]*ratio,
                                              reference["analytic_gflops"]*ratio)))
    best = max(results, key=lambda name: results[name]["time_only_ceiling"])
    table = ["| Вариант | Vulnerability F1 | MAT F1 | Средний F1 | CPU, с | Веса, МБ | Потолок с учётом времени |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        c = r["cost"]
        row = [NAMES[name], *(dec(r["scores"][key]["f1"]) for key in ("vulnerability", "MAT", "mean")),
               dec(c["median_seconds"], 2), dec(c["weights_bytes"]/1e6, 2), dec(r["time_only_ceiling"])]
        table.append("| " + " | ".join(row) + " |")
    gain = []
    for name in evaluated:
        lo, hi = comparisons[f"{name}_minus_bm25"]["mean"]
        gain.append(f"- {NAMES[name]} относительно BM25 от {dec(100*lo, 2)} до {dec(100*hi, 2)} п.п.")
    text = [
        "# Обучаемое разреженное расширение",
        f"Максимальный локальный потолок среди CPU-вариантов у модели «{NAMES[best]}» "
        f"({dec(results[best]['time_only_ceiling'])}). Потолок учитывает измеренное время и предполагает "
        f"нулевую вычислительную стоимость. Локальный GPU-балл CodeT5+ по тому же публичному набору — {dec(gpu_score)}.",
        "\n".join(table),
        "Парные 95%-е интервалы разницы среднего F1 получены bootstrap по группам проекта и файла.",
        "\n".join(gain),
        "Обе разреженные модели сохраняют лексическую часть BM25 и учат дополнительные признаки "
        "через максимум по вкладам термов и log(1+ReLU). Из 64 дополнительных признаков сохраняется до 12 "
        "на функцию. Это перенос идеи SPLADE на классификацию с ограниченным числом каналов. "
        "Учитель CodeT5+ предсказывал только строки fit и использовался для вариантов с дистилляцией.",
        "Все итоговые пороги подобраны на 3 042 train-строках. Сравнение вариантов доступно также на "
        "18 444 других train-строках. Публичные 30 655 функций пройдены трижды на Intel Core i7-7700 "
        "с учётом токенизации, признаков и обеих меток. Официальный рейтинг зависит от измерения операций, "
        "Colab T4 и скрытых данных.",
        "Для воспроизведения в том же окружении выполните `python3 nlbse/portfolio/sparse_teacher.py` и "
        "`python3 nlbse/portfolio/sparse_expansion.py fit` на CUDA. Затем выполните `python3 "
        "nlbse/portfolio/sparse_expansion.py public`. Для дополнительного контроля повторите fit и public "
        "с `--linear-distilled`. Итоговые таблицы строит `python3 nlbse/portfolio/sparse_expansion_report.py`. "
        "Для INT8-экспорта выполните `quantize` и `public --int8`. "
        "Команда `predict` принимает название варианта и CSV с колонкой Function. "
        "Детали и контрольные суммы в [analysis.json](analysis.json).",
    ]
    if quantization is not None:
        changes = sum(quantization["changed_labels_by_task"])
        reduction = dec(100*quantization["weight_reduction"], 1)
        text.insert(3, f"INT8-экспорт матрицы расширения уменьшил размер файла на {reduction}%. "
                    f"При прежних порогах изменилось {changes} из {2*len(y):,} бинарных решений.")
    (OUT / "results.md").write_text("\n\n".join(text) + "\n")
    print("Audited", len(y), "rows for", len(results), "models", flush=True)


if __name__ == "__main__":
    main()
