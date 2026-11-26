"""Audit all three context experiments, paired intervals, and final tables."""
import csv
import json

import numpy as np

from common import HERE, PUBLIC, ROOT, TRAIN, load, metric, write_json
from analyze import changes, group_codes, predict_bits, verify
from round2_report import bootstrap_delta

TRAINING = HERE / "training"
ROUTER = HERE / "router"
PROJECT = HERE / "project"
MASKING = HERE / "masking"


def read(path):
    return json.loads(path.read_text())


def case(path, y, ids):
    receipt = read(path/"metrics.json")
    bits = predict_bits(path/"predictions.csv", ids, receipt)
    verify(y, bits, receipt)
    return receipt, bits


def pair(y, left, right, groups):
    return dict(interval=bootstrap_delta(y,left,right,groups),changes=changes(y,left,right))


def group_data(fields):
    return group_codes(zip(fields["Projectname"],fields["Filepath"]))


def finetune(y,ids,groups):
    matrix = read(TRAINING/"public_summary.json")
    frozen = read(TRAINING/"frozen.json")
    receipts,bits = {},{}
    for key, r in matrix.items():
        path = TRAINING/"runs/public"/key
        receipt, p = case(path,y,ids)
        if receipt["scores"]["mean"]["f1"] != r["mean_F1"]:
            raise ValueError("Training summary differs from predictions")
        receipts[key] = dict(**receipt,local_formula=r["local_formula"])
        bits[key] = p
    comparisons = {}
    for policy in ("head384","comment384","comment512"):
        for a,b in (("frozen","fixed384"),("frozen","mixed"),("fixed384","mixed")):
            before, after = bits[f"{a}/{policy}"],bits[f"{b}/{policy}"]
            comparisons[f"{b}_minus_{a}/{policy}"] = pair(y,before,after,groups)
    baseline = case(ROOT/"round5/runs/public/skip_positive",y,ids)[1]
    historical = (bits["frozen/comment512"]!=baseline).sum(axis=0).tolist()
    if any(historical):
        raise ValueError("The original checkpoint changed at the same input policy")
    return dict(frozen=frozen,holdout=read(TRAINING/"holdout_summary.json"),public=receipts,paired=comparisons,
                historical_label_changes=historical)


def routed(y,ids,groups,train_y,train_fields):
    frozen = read(ROUTER/"frozen.json")
    result,bits = {},{}
    for k in ("stop","rule","learned"):
        result[k],bits[k] = case(ROUTER/"runs/public"/k,y,ids)
    reference = case(ROOT/"round5/runs/budget/cap384_cover64",y,ids)[1]
    agreement = (bits["rule"]!=reference).sum(axis=0).tolist()
    if any(agreement):
        raise ValueError("The routed comparison changed the matched rule's predictions")
    costs = {k:result[k]["cost"]["extra_gflops"] for k in result}
    if costs["learned"] > costs["rule"]+1e-7:
        raise ValueError("Public routing exceeds matched second-pass budget")
    with np.load(ROUTER/"train_actions.npz") as cache:
        review = cache["ids"][cache["folds"]==4].tolist()
    review_y = train_y[review]
    review_group = group_codes((train_fields["Projectname"][i],train_fields["Filepath"][i]) for i in review)
    holdout = {}
    holdout_bits = {}
    for k in ("stop","rule","learned","oracle_diagnostic"):
        holdout[k],holdout_bits[k] = case(ROUTER/"runs/holdout"/k,review_y,review)
    return dict(public=result,holdout=holdout,frozen=frozen,
                paired=pair(y,bits["rule"],bits["learned"],groups),
                internal_pair=pair(review_y,holdout_bits["rule"],holdout_bits["learned"],review_group),
                reference_label_changes=agreement)


def ablation(source, labels, columns):
    root = MASKING/source
    design = read(root/"design.json")
    with np.load(root/"rows.npz") as cache:
        ids = cache["ids"].tolist()
        take,cross = cache["eligible"],cache["crossing"]
        families = cache["group"]
        y = labels[ids]
        group = group_codes((columns["Projectname"][i],columns["Filepath"][i]) for i in ids)
        per_window = dict(median_comment_tokens=float(np.median(cache["comment_in_window"][take])) if take.any() else 0.,
                          median_code_tokens=float(np.median((cache["content_in_window"]-cache["comment_in_window"])[take]))
                          if take.any() else 0., boundary_crossings=int((take & cross).sum()))
    receipts,bits = {},{}
    for name in ("head","full","comment_only","code_only"):
        receipts[name],bits[name] = case(root/name,y,ids)
        if not np.array_equal(bits[name][:,0],bits["head"][:,0]) or not np.array_equal(bits[name][~take],bits["head"][~take]):
            raise ValueError("Intervention changed an untargeted label or function")
    extras = [receipts[name]["cost"]["analytic_gflops"] for name in ("full","comment_only","code_only")]
    if len(set(extras))!=1 or not design["identical_shapes"]:
        raise ValueError("Dense computation shapes differ across masking conditions")
    cohorts = {"selected":dict(rows=int(take.sum()), positives=y[take].sum(axis=0).tolist(),
                               variants={k:metric(y[take],v[take]) for k,v in bits.items()})}
    if source == "project":
        for family in sorted(set(families)):
            mask = families==family
            candidates = mask & take
            cohorts[family] = dict(rows=int(mask.sum()), positives=y[mask].sum(axis=0).tolist(),
                                   window_calls=int(candidates.sum()),selected_mat_positive=int(y[candidates,1].sum()),
                                   variants={k:metric(y[mask],v[mask]) for k,v in bits.items()})
    comparisons = {f"{b}_minus_{a}":pair(y,bits[a],bits[b],group)
                   for a,b in (("head","full"),("head","comment_only"),("head","code_only"),
                               ("code_only","comment_only"),("full","comment_only"))}
    return dict(variants=receipts,paired=comparisons,cohorts=cohorts,design=design,window_content=per_window)


def audit():
    _,y,fields = load(PUBLIC,audit=True)
    _,train_y,train_fields = load(TRAIN,audit=True)
    ids = list(range(len(y)))
    groups = group_data(fields)
    result = dict(
        continuation=finetune(y,ids,groups),
        router=routed(y,ids,groups,train_y,train_fields),
        masking=dict(public=ablation("public",y,fields),project=ablation("project",train_y,train_fields)),
        project_split=read(PROJECT/"split.json"),
        references=dict(prior_local_formula=read(ROOT/"round5/analysis.json")["results"]["skip_positive"]["local_formula"]),
    )
    write_json(HERE/"analysis.json",result)
    with (HERE/"results.csv").open("w",newline="") as file:
        out = csv.writer(file)
        out.writerow(("experiment","dataset","policy","vulnerability_P","vulnerability_R","vulnerability_F1",
                      "MAT_P","MAT_R","MAT_F1","mean_F1","dense_GFLOPs","forward_seconds","extra_windows","local_formula"))
        sets = [("training","public",result["continuation"]["public"]),
                ("routing","public",result["router"]["public"])]
        sets += [("masking",scope,r["variants"]) for scope,r in result["masking"].items()]
        for study,scope,run in sets:
            for name,r in run.items():
                c=r["cost"]
                out.writerow((study,scope,name,*(r["scores"][label][stat] for label in ("vulnerability","MAT")
                                               for stat in ("precision","recall","f1")),r["scores"]["mean"]["f1"],
                              c.get("analytic_gflops"),c.get("forward_seconds"),
                              c.get("extra_windows",sum(c.get("actions",[0])[1:])),r.get("local_formula")))
    return result


def f1(row,label="mean"):
    return row["scores"][label]["f1"]


def dec(number,digits=4):
    return f"{number:.{digits}f}".replace(".",",")


def bounds(pair_result,label="mean"):
    low,high=pair_result["interval"][label]
    return f"{dec(100*low,2)}…{dec(100*high,2)} п.п."


def markdown(r):
    t,o,m = r["continuation"],r["router"],r["masking"]
    lines = ["# Три продолжения эксперимента с дополнительным контекстом",
             "## 1. Дообучение при разных бюджетах",
             "Две ветви продолжили обучение от одного checkpoint на одних 40 000 строках. "
             "Первичная проверка использует общий порог, одно и то же окно и одинаковую процедуру второго прохода.",
             "| Модель | 384, один проход | 384+64 | 512+128 | Балл 384+64, RTX 4090 |",
             "|---|---:|---:|---:|---:|"]
    for arm,label in (("frozen","Исходная"),("fixed384","Фиксированное 384"),("mixed","Длины 256/384/512")):
        values = [dec(f1(t["public"][f"{arm}/{view}"])) for view in ("head384","comment384","comment512")]
        lines.append("| "+" | ".join((label,*values,dec(t["public"][arm+"/comment384"]["local_formula"])))+" |")
    lines += ["", f"При одинаковом инференсе 384+64 парный 95%-й интервал для смешанной длины минус "
                    f"фиксированные 384 составил {bounds(t['paired']['mixed_minus_fixed384/comment384'])}. "
                    f"На внутренней выборке выбран `{t['frozen']['preferred_384']}`.",
              "## 2. Выбор действия по ожидаемой пользе",
              "Голова обучена на отдельной группе результатов замороженного энкодера. На каждом блоке "
              "её второй этап укладывается в расчётный бюджет правила с окном 64. Широкий проход "
              "учитывается полностью. Время первого прохода измерялось совместно, время маршрутизации и "
              "дочитывания отдельно. В таблице указаны общие операции с учётом маленькой головы.",
              "| Политика | F1 | Комментарий | До 768 | GFLOPs | Инференс, с |",
              "|---|---:|---:|---:|---:|---:|"]
    for key,label in (("stop","Стоп"),("rule","Правило"),("learned","Обучаемая голова")):
        row=o["public"][key];c=row["cost"]
        lines.append("| "+" | ".join((label,dec(f1(row)),str(c["actions"][1]),str(c["actions"][2]),
                                       dec(c["analytic_gflops"],0),dec(c["inference_seconds"],2)))+" |")
    lines += ["",f"Голова минус правило на публичной выборке даёт парный 95%-й интервал "
                 f"{bounds(o['paired'])}. На проверочной части train метки были доступны отдельному "
                 f"диагностическому варианту, который достиг {dec(f1(o['holdout']['oracle_diagnostic']))} F1.",
              "## 3. Что именно даёт улучшение", 
              "| Вход второго окна | Публичный MAT F1 | Публичный средний F1 | MAT F1 на новых проектах | Средний F1 на новых проектах |",
              "|---|---:|---:|---:|---:|"]
    for key,label in (("head","Первое окно"),("full","Текст целиком"),("comment_only","Только комментарии"),
                      ("code_only","Только соседний код")):
        cells = [dec(f1(m[s]["variants"][key],task)) for s in ("public","project") for task in ("MAT","mean")]
        lines.append("| "+" | ".join((label,*cells))+" |")
    lines += ["", f"Masking сохраняет исходные позиции, идентификаторы токенов, специальные токены и "
                  f"формы батчей. На публичных данных использованы одни и те же "
                  f"{m['public']['design']['window_calls']} вторых окон, на отложенных проектах "
                  f"{m['project']['design']['window_calls']}. Парные интервалы разницы MAT F1 между окном "
                  f"с комментариями и окном с кодом составили "
                  f"{bounds(m['public']['paired']['comment_only_minus_code_only'],'MAT')} на публичных данных и "
                  f"{bounds(m['project']['paired']['comment_only_minus_code_only'],'MAT')} на отложенных проектах.",
              "Для отложенных Chromium, QEMU и FFmpeg энкодер обучен заново от исходных весов на 40 000 "
              "функциях других проектов. Точные копии отложенных функций исключены из fit и калибровки. "
              "Состав предобучения Salesforce не проверялся.",
              "Точные прогнозы, стоимости и интервалы доступны в [analysis.json](analysis.json) и "
              "[results.csv](results.csv). Официальный рейтинг требует T4 и скрытых данных."]
    out=[]
    for i,line in enumerate(lines):
        if i and not line.startswith("|") and lines[i-1] and not lines[i-1].startswith("|"):
            out.append("")
        elif i and line.startswith("|") and not lines[i-1].startswith("|"):
            out.append("")
        out.append(line)
    (HERE/"results.md").write_text("\n".join(out)+"\n")


if __name__ == "__main__":
    r = audit()
    markdown(r)
    print("All three experiments audited",flush=True)
