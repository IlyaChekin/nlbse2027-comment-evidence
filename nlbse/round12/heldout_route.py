"""Check whether a development-budgeted second view also helps on unseen projects."""
import csv
from types import SimpleNamespace

import numpy as np

from backbones import frozen
from augmentation import DEST as AUG, variants
from project_heads import calibration, corpus, OUT as HEADS
from project_split import HERE
from efficient_aug import run
from pipeline import metric, write_json
from students import group_codes
from round2_report import bootstrap_delta

OUT=HERE/"heldout_route"


def reference(path,ids):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows]!=ids.tolist():
        raise ValueError("Wrong comparison rows")
    return np.asarray([[int(r["pred_vulnerability"]),int(r["pred_MAT"])] for r in rows],dtype=bool)


def main():
    texts,y,projects,paths,split=corpus()
    dev,test=split["dev"],split["test"]
    dev_p,_=variants([texts[i] for i in dev])
    a,b=dev_p["aug_raw"],dev_p["aug_norm"]
    grid=[]
    for lo in (.24,.31,.36,.4,.45,.52):
        for hi in (.58,.65,.73,.81,.89):
            active=(a>lo)&(a<hi)
            fraction=float(active.mean())
            if fraction>.15:
                continue
            r=calibration(y[dev,0],np.where(active,.5*(a+b),a),projects[dev])
            grid.append(dict(view="selective",lower=lo,upper=hi,threshold=r["cutoff"],
                             dev_objective=r["objective"],dev_fraction=fraction))
    rule=max(grid,key=lambda v:(v["dev_objective"],-v["dev_fraction"]))
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"policy.json",dict(chosen=rule,grid=grid))
    net,tok=frozen("canonical")
    pipe=SimpleNamespace(encoder=net,tok=tok,student={"thresholds":(.5,.5)})
    pred,cost=run(pipe,[texts[i] for i in test],rule)
    baseline=reference(HEADS/"compact_raw_test.csv",test)
    aug=reference(AUG/"aug_mean.csv",test)
    bits=np.column_stack((pred[:,0]>rule["threshold"],baseline[:,1]))
    group=projects[test]
    clusters=group_codes(zip(group,[paths[i] for i in test],strict=True))
    with (OUT/"predictions.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("row","project","p_vulnerability","pred_vulnerability","pred_MAT"))
        writer.writerows(zip(test,group,pred[:,0],bits[:,0].astype(int),bits[:,1].astype(int),strict=True))
    summary=dict(metrics=metric(y[test],bits),cost=cost,
        by_project={g:metric(y[test[group==g]],bits[group==g]) for g in np.unique(group)},
        interval_vs_base=bootstrap_delta(y[test],baseline,bits,clusters),
        interval_vs_full_view=bootstrap_delta(y[test],aug,bits,clusters))
    write_json(OUT/"analysis.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    main()
