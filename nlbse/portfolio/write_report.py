"""Write the Russian research summary from the audited portfolio results."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {
    "sparse": "Линейная модель на n-граммах",
    "sparse_roles": "Линейная модель с раздельными каналами",
    "forest": "Деревья на признаках структуры",
    "memory": "Поиск по токенным шинглам",
    "byte": "Байтовая CNN",
    "codebert": "CodeBERT, 6 блоков",
    "graphcodebert": "GraphCodeBERT, 6 блоков",
    "codet5p": "CodeT5+, 6 блоков энкодера",
    "control": "CodeBERT, первое окно",
    "full_tail": "CodeBERT, полный хвост",
    "score_candidate": "CodeBERT, окно комментария",
    "quality_candidate": "CodeBERT, выборочный полный хвост",
    "t5_head": "CodeT5+, первое окно",
    "t5_comment": "CodeT5+, окно комментария",
}


def dec(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def f1(row, name="mean"):
    return row["scores"][name]["f1"]


def interval(row):
    a, b = [100*x for x in row["delta95"]["mean"]]
    return f"от {dec(a, 2)} до {dec(b, 2)} п.п."


def prose(report):
    pilot, public = report["pilot"], report["public"]
    t5 = public["codet5p"]
    other = {k: t5[k] for k in ("t5_head", "t5_comment")}
    gpu = {**public["models"], **other}
    sparse = public["sparse"]
    best = max(gpu, key=lambda k: gpu[k]["formula_using_local_time_and_analytic_gflops"])
    best_quality = max(gpu, key=lambda k: f1(gpu[k]))
    primary = gpu["score_candidate"]
    old = gpu["control"]
    full_tail = gpu["full_tail"]
    key = f"t5_{t5['frozen']['mode']}"
    fp = public["paired"]["score_candidate"]
    mat = fp["changes"]["MAT"]
    gain_kept = (f1(primary)-f1(old))/(f1(full_tail)-f1(old))
    added_ops = (primary["analytic_gflops"]/old["analytic_gflops"]-1)*100

    yield "# NLBSE 2027 · портфель моделей"
    yield (f"На публичных 30 655 функциях максимальный средний F1 у варианта «{NAMES[best_quality]}» "
           f"({dec(f1(gpu[best_quality]))}). По локальной формуле среди измеренных GPU-вариантов лидирует "
           f"«{NAMES[best]}» ({dec(gpu[best]['formula_using_local_time_and_analytic_gflops'])}). "
           f"Компактная CPU-модель получила {dec(f1(sparse))} F1 и занимает "
           f"{dec(public['sparse_streaming']['weights_bytes']/1e6, 2)} МБ. "
           "Сроки, документация и следующий запуск указаны в [STATUS.md](../STATUS.md).")
    yield "## Сравнение на одинаковом объёме обучения"
    yield ("У каждого пилота 20 000 строк fit, 3 042 строки калибровки и 18 444 строки сравнения. "
           "Группы одинакового полного текста удерживаются вместе в разбиении train. "
           "Энкодеры используют исходные предобученные веса, а остальные модели обучаются на данных задачи.")
    yield "\n".join([
        "| Семейство | Vulnerability F1 | MAT F1 | Средний F1 |",
        "|---|---:|---:|---:|",
        *(f"| {NAMES[k]} | {dec(f1(r,'vulnerability'))} | {dec(f1(r,'MAT'))} | {dec(f1(r))} |"
          for k, r in pilot["models"].items()),
    ])
    yield (f"У CodeT5+ преимущество над парным CodeBERT составило "
           f"{dec(100*(f1(pilot['models']['codet5p'])-f1(pilot['models']['codebert'])), 2)} п.п. "
           f"(парный 95%-й интервал {interval(pilot['paired']['codet5p_minus_codebert'])}). "
           "GraphCodeBERT с обычным токенным входом отстал главным образом по MAT. "
           "У поиска по шинглам и байтовой сети в этом бюджете низкий F1 по MAT.")
    yield (f"Разделение лексических признаков кода, комментариев и хвоста повысило средний F1 "
           f"линейного пилота на {dec(100*(f1(pilot['models']['sparse_roles'])-f1(pilot['models']['sparse'])), 2)} п.п. "
           f"(парный 95%-й интервал {interval(pilot['paired']['sparse_roles_minus_sparse'])}). "
           "Изменение также включает новые символьные n-граммы, логарифмирование счётчиков и больший вес каналов комментариев.")
    yield "## Полная публичная выборка"
    yield "\n".join([
        "| Вариант | Средний F1 | Вторых окон | Операции, тыс. GFLOPs | Время GPU, с | Локальная формула |",
        "|---|---:|---:|---:|---:|---:|",
        *(f"| {NAMES[k]} | {dec(f1(r))} | {r['extra_windows']} | {dec(r['analytic_gflops']/1000, 1)} "
          f"| {dec(r['forward_seconds_median'], 2)} | {dec(r['formula_using_local_time_and_analytic_gflops'])} |"
          for k,r in gpu.items()),
    ])
    yield (f"Короткое второе окно у CodeBERT сохранило {dec(100*gain_kept, 1)}% прироста F1 полного хвоста "
           f"с прибавкой {dec(added_ops, 2)}% операций к одному окну. Порог MAT составил 0,50 в обоих случаях. "
           f"Изменение дало {mat['extra_tp']} дополнительных верных MAT-положительных решений и "
           f"{mat['extra_fp']} ложное срабатывание. Решения по vulnerability совпадают. "
           f"Парный 95%-й интервал прироста среднего F1 {interval(fp)}")
    yield (f"После продолжения CodeT5+ на 193 100 строках по внутренним данным зафиксирован вариант «{NAMES[key]}». "
           f"При переносе короткого окна средний F1 на публичной выборке изменился с "
           f"{dec(f1(other['t5_head']))} до {dec(f1(other['t5_comment']))} "
           f"(парный 95%-й интервал изменения {interval(t5['t5_comment_minus_t5_head'])}). "
           "В публичной таблице использован переносимый FP16 checkpoint.")
    if key == "t5_comment":
        contrast = t5["t5_comment_minus_score_candidate"]
        novel = public["cohorts"]["no_exact_train_text"]["scores"]["t5_comment"]["mean"]["f1"]
        yield (f"CodeT5+ с коротким окном опережает CodeBERT с тем же правилом на "
               f"{dec(100*(f1(other[key])-f1(primary)), 2)} п.п. "
               f"(парный 95%-й интервал {interval(contrast)}). "
               f"После исключения точных совпадений с train его средний F1 составляет {dec(novel)}.")
    yield (f"Полная линейная модель получила {dec(f1(sparse))} F1, в том числе "
           f"{dec(f1(sparse,'MAT'))} по MAT. Потоковый CLI воспроизвёл все бинарные решения "
           f"с медианой {dec(public['sparse_streaming']['median_seconds'], 2)} с на Intel Core i7-7700. "
           f"Теоретический потолок конкурсной формулы при нулевой стоимости равен "
           f"{dec(public['sparse_zero_cost_score_cap'])}. "
           "Сопоставление места этой модели зависит от замера времени и учёта разреженных операций.")
    yield ("Время GPU получено на RTX 4090 как медиана трёх запусков без времени токенизации. "
           "Операции оценены по размерностям плотных проекций и внимания с учётом padding. "
           "Отдельный запуск на одинаковых 2 048 строках проверил учёт PyTorch с десятью повторениями "
           "для CodeBERT с первым и коротким окнами. Итоговый конкурсный балл требует T4 и скрытых данных.")
    yield ("Полные precision, recall, F1, доверительные интервалы, данные разбиения и инструкции "
           "в [public_table.csv](public_table.csv), [analysis.json](analysis.json) и [README.md](README.md).")


def main():
    report = json.loads((HERE / "analysis.json").read_text())
    (HERE / "results.md").write_text("\n\n".join(prose(report)) + "\n")


if __name__ == "__main__":
    main()
