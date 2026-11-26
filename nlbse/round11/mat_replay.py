"""Check when a transparent comment rule reproduces the fitted MAT head."""
from pathlib import Path
from statistics import median
import time

import numpy as np

import comment_head as mat
from students import dataset, write_json
from comment_spans import LEXEME
from frontier import TAG


def tagged(text):
    if not TAG.search(text):
        return False
    return any(s[0].startswith(("//", "/*")) and TAG.search(text, *s.span())
               for s in LEXEME.finditer(text))


def predict(texts):
    return np.fromiter(map(tagged,texts), dtype=bool, count=len(texts))


def main():
    import json
    result={}
    config=json.loads((mat.OUT/"selected.json").read_text())["chosen"]
    for public in (False,True):
        _,_,texts,y,ids,_=dataset(full=True,public=public)
        with np.load(mat.OUT/("public.npz" if public else "holdout.npz")) as cache:
            if cache["ids"].tolist()!=ids or not np.array_equal(cache["y"],y):
                raise ValueError("Unexpected output alignment")
            fitted=cache["p"]>config["threshold"]
        samples=[]
        expected=None
        for _ in range(3):
            start=time.perf_counter()
            rule=predict(texts)
            samples.append(time.perf_counter()-start)
            if expected is not None and not np.array_equal(expected,rule):
                raise ValueError("The same source functions produced different outcomes")
            expected=rule
        result["public" if public else "internal"]=dict(rows=len(y), changed=int((rule!=fitted).sum()),
            positive_changes=int(((rule!=fitted)&(y[:,1]==1)).sum()),wall_seconds=median(samples),trials=samples)
    write_json(mat.OUT/"rule_replay.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
