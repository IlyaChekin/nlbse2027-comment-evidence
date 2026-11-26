"""Check whether cold depthwise convolutions can be replaced without changing decisions."""
import argparse
from pathlib import Path
from statistics import median

import numpy as np

import cascade
from neighbor_eval import table
from students import dataset, metric, save, write_json
from score_runtime import competition_formula

HERE=Path(__file__).resolve().parent


def check(method):
    _,_,texts,y,ids,cal=dataset(public=True)
    pipe=cascade.Pipeline("small_lexical",256,"gold_disagreement",shifted=(method=="shift"))
    reference=table(HERE/"cascade256_gold_disagreement/small_lexical/predictions.csv")
    if reference[:,0].tolist()!=ids:
        raise ValueError("Unexpected reference row order")
    pipe.run(texts[:2048])
    results=[]
    changed=None
    expected=None
    for _ in range(3):
        p,cost=pipe.run(texts)
        score=metric(y,p)
        diff=np.count_nonzero(p!=reference[:,2:],axis=0).tolist()
        if expected is not None and not np.array_equal(expected,p):
            raise ValueError("Unstable predictions on repeat")
        expected=p
        changed=diff
        results.append(cost)
        print(method,cost,diff,flush=True)
    cost={k:median(r[k] for r in results) for k in results[0]}
    out=HERE/f"cold_{method}"
    save(out,y,ids,cal,p,cost,(.5,.5))
    write_json(out/"analysis.json",dict(method=method,changed_by_label=changed,cost=cost,
          trials=results,metrics=score,score=competition_formula(score["mean"]["f1"],cost["forward_seconds"],cost["analytic_gflops"])))


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("method",choices=("shift",))
    check(args.parse_args().method)
