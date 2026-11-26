"""Audit the lightweight CPU baselines and write the comparison tables."""
import csv
import hashlib
import json

from analyze import group_codes, predict_bits, verify
from pipeline import PUBLIC, load, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from simple_bench import ALL_MODES, HERE, OUT, PREVIOUS

MODELS = (*ALL_MODES, "previous_sparse")
T5 = HERE / "runs/full/codet5p/portable_results/comment/public"
NAMES = {
    "count": "Счётчики",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "tfidf_comments": "TF-IDF + канал комментариев",
    "bm25_comments": "BM25 + канал комментариев",
    "logtfidf": "TF-IDF с логарифмом частоты",
    "logtfidf_comments": "TF-IDF с логарифмом + комментарии",
    "bm25_comments_bigram": "BM25 + биграммы + комментарии",
    "previous_sparse": "Прежняя линейная модель",
}


def read(path):
    return json.loads(path.read_text())


def fixed_threshold(mode):
    source = PREVIOUS if mode == "previous_sparse" else OUT / "runs" / mode / "holdout"
    return read(source / "metrics.json")["thresholds"]


def model_hash(mode):
    source = PREVIOUS if mode == "previous_sparse" else OUT / "runs" / mode
    with (source / "model.joblib").open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compare():
    _, y, fields = load(PUBLIC, audit=True)
    ids = list(range(len(y)))
    groups = group_codes(zip(fields["Projectname"], fields["Filepath"]))
    t5 = read(T5 / "metrics.json")
    t5_bits = predict_bits(T5 / "predictions.csv", ids, t5)
    verify(y, t5_bits, t5)
    gpu = read(HERE / "analysis.json")["public"]["codet5p"]["t5_comment"]
    score = gpu["formula_using_local_time_and_analytic_gflops"]
    hidden_rows = 61311
    ratio = hidden_rows / len(y)
    projected_gpu = competition_formula(
        gpu["scores"]["mean"]["f1"], ratio * gpu["forward_seconds_median"], ratio * gpu["analytic_gflops"])
    results, bits = {}, {}
    for mode in MODELS:
        path = OUT / "runs" / mode / "public"
        r = read(path / "metrics.json")
        if r["thresholds"] != fixed_threshold(mode):
            raise ValueError(f"{mode}: public thresholds differ from the frozen configuration")
        q = predict_bits(path / "predictions.csv", ids, r)
        verify(y, q, r)
        if mode == "previous_sparse":
            original = read(PREVIOUS / "public/metrics.json")
            verify(y, q, original)
        bits[mode] = q
        f1 = r["scores"]["mean"]["f1"]
        results[mode] = dict(model=mode, **r, sha256=model_hash(mode),
                             zero_cost_score_ceiling=0.4+0.6*f1,
                             projected_ceiling_at_hidden_volume=competition_formula(
                                 f1, ratio*r["cost"]["median_seconds"], 0.))
    pairs = (("tfidf", "bm25"), ("tfidf_comments", "bm25_comments"),
             ("logtfidf", "bm25"), ("logtfidf_comments", "bm25_comments"),
             ("bm25", "bm25_comments"), ("bm25_comments", "bm25_comments_bigram"),
             ("previous_sparse", "bm25_comments_bigram"))
    paired = {f"{b}_minus_{a}": bootstrap_delta(y, bits[a], bits[b], groups) for a, b in pairs}
    write_json(OUT / "analysis.json", dict(rows=len(y), results=results, paired=paired,
                                          codet5_local_formula=score,
                                          necessary_f1_even_at_zero_cost=(score-0.4)/0.6,
                                          simulated_hidden_volume=dict(rows=hidden_rows, ratio=ratio,
                                                                       projected_gpu=projected_gpu)))
    with (OUT / "public_table.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        header = [f"{label}_{name}" for label in ("vulnerability", "MAT", "mean")
                  for name in ("precision", "recall", "f1")]
        writer.writerow(["model", *header, "median_seconds", "weights_bytes", "nnz", "zero_cost_score_ceiling"])
        for mode, r in results.items():
            cost = r["cost"]
            values = [r["scores"][label][name] for label in ("vulnerability", "MAT", "mean")
                      for name in ("precision", "recall", "f1")]
            writer.writerow([mode, *values, *(cost[k] for k in ("median_seconds", "weights_bytes", "nnz")),
                             r["zero_cost_score_ceiling"]])
    print("Verified", len(y), "public rows for", len(results), "models")


def dec(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def report():
    r = read(OUT / "analysis.json")
    results = r["results"]
    best = max(ALL_MODES, key=lambda m: results[m]["scores"]["mean"]["f1"])
    fastest = min(ALL_MODES, key=lambda m: results[m]["cost"]["median_seconds"])
    old = results["previous_sparse"]
    best_row = results[best]
    speed = old["cost"]["median_seconds"]/best_row["cost"]["median_seconds"]
    table = ["| Модель | Vulnerability F1 | MAT F1 | Средний F1 | CPU, с | Веса, МБ | Потолок балла |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in results.items():
        f1 = [dec(row["scores"][name]["f1"]) for name in ("vulnerability", "MAT", "mean")]
        cost = row["cost"]
        cells = [NAMES[mode], *f1, dec(cost["median_seconds"], 2), dec(cost["weights_bytes"]/1e6, 2),
                 dec(row["zero_cost_score_ceiling"])]
        table.append("| " + " | ".join(cells) + " |")

    intervals = []
    for name in ("bm25_minus_tfidf", "bm25_comments_minus_tfidf_comments", "bm25_minus_logtfidf",
                 "bm25_comments_bigram_minus_previous_sparse"):
        low, high = r["paired"][name]["mean"]
        intervals.append(f"от {dec(100*low, 2)} до {dec(100*high, 2)} п.п.")

    text = [
        "# Сверхпростые бейзлайны NLBSE",
        f"Из новых вариантов самый точный — **{NAMES[best]}**. Он ускоряет прежнюю CPU-модель в {dec(speed, 1)} раза. "
        f"Минимальное время у варианта «{NAMES[fastest]}».",
        "\n".join(table),
        f"Парные 95%-е интервалы для разницы среднего F1 между BM25 и TF-IDF составили {intervals[0]} "
        f"на униграммах и {intervals[1]} с отдельным каналом комментариев. Bootstrap группирует строки по проекту и файлу.",
        f"При замене обычного TF-IDF на TF-IDF с логарифмом частоты парная разница BM25 минус TF-IDF на униграммах "
        f"составляет {intervals[2]}",
        f"Для лучшей новой модели с биграммами парная разница с прежней линейной моделью составляет {intervals[3]}",
        f"Для равенства с локальным баллом CodeT5+ ({dec(r['codet5_local_formula'])}) даже при нулевой стоимости "
        f"требуется средний F1 {dec(r['necessary_f1_even_at_zero_cost'])}. В этой проверке лучшая новая линейная "
        f"модель имеет потолок {dec(best_row['zero_cost_score_ceiling'])}. "
        "Это сравнение с локальным замером, а официальный балл зависит от Colab T4 и скрытой выборки.",
        f"В отдельном сценарии объём увеличен до 61 311 строк, а F1 оставлен на публичном уровне. "
        f"При линейном масштабировании локального времени и операций оценка CodeT5+ составила "
        f"{dec(r['simulated_hidden_volume']['projected_gpu'])}. "
        f"Потолок лучшей новой CPU-модели с учётом времени и нулевой цены операций составляет "
        f"{dec(best_row['projected_ceiling_at_hidden_volume'])}. Это сценарий для сравнения стоимости.",
        "Фиксированы 193 100 обучающих строк, 3 042 строки для порогов и 18 444 строки для выбора. "
        "Таблица показывает все 30 655 публичных функций. Время на Intel Core i7-7700 измерялось трижды с прогревом "
        "и включает токенизацию, вычисление признаков и прогноз обеих меток. Использовались блоки по 1 024 строки "
        "и те же настройки потоков для всех CPU-вариантов. Векторы BM25 — веса признаков логистической модели. "
        "Индекс поиска для инференса не требуется.",
        "Точные метрики, парные интервалы и контрольные суммы в [таблице](public_table.csv) и [анализе](analysis.json).",
        "Для воспроизведения на той же среде используйте `python3 nlbse/portfolio/simple_bench.py holdout`, "
        "затем `python3 nlbse/portfolio/simple_bench.py public`. Повторите обе команды с `--followup` "
        "и запустите `python3 nlbse/portfolio/simple_report.py`. "
        "Версии данных и подготовка среды описаны в [инструкциях портфеля](../README.md).",
    ]
    (OUT / "results.md").write_text("\n\n".join(text) + "\n")


if __name__ == "__main__":
    compare()
    report()
