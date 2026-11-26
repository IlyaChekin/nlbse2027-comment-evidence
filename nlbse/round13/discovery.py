"""Audit the original confident misses without retuning on their labels."""
import csv
from collections import Counter
import json

import numpy as np

from rotate import HERE, ROOT, PLAN
from bindings import LANGUAGE, Parser
from pipeline import TRAIN, load, write_json


def data(path):
    with path.open(newline="") as f:
        return {int(r["row"]):float(r["p_vulnerability"]) for r in csv.DictReader(f)}


def classify(text,parser):
    root=parser.parse(text.encode()).root_node
    functions=[node for node in root.named_children if node.type=="function_definition"]
    if not functions:
        return "unrecognized_function"
    body=functions[0].child_by_field_name("body")
    if body is None:
        return "missing_body"
    children=[node for node in body.named_children if node.type!="comment"]
    if not children:
        return "empty_body"
    if len(children)==1:
        return f"single_{children[0].type}"
    return "multiple_statements"


def main():
    a=data(ROOT/"round12/heads/compact_raw_test.csv")
    b=data(ROOT/"round12/heads/unix_raw_test.csv")
    texts,y,m=load(TRAIN,audit=True)
    cfg=json.loads(PLAN.read_text())
    low=[i for i in a if a[i]<.05 and b[i]<.05]
    missed=[i for i in low if y[i,0]==1]
    threshold=cfg["short_chars_max"]
    parser=Parser(LANGUAGE)
    def hist(ids):
        selected=[i for i in ids if len(texts[i])<=threshold]
        return dict(total=len(ids),short=len(selected),chars_by_quantile=np.quantile([len(texts[i]) for i in ids],[0,.25,.5,.75,1]).tolist(),
                    short_syntax=dict(Counter(classify(texts[i],parser) for i in selected)),
                    project_raw=dict(Counter(m["Projectname"][i] for i in selected)))
    result=dict(length_cutoff=threshold,original_joint_misses=hist(missed),
                original_joint_correct_negatives=hist([i for i in low if not y[i,0]]))
    if len(missed)!=1562:
        raise ValueError("The original discovery slice is different")
    write_json(HERE/"discovery.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
