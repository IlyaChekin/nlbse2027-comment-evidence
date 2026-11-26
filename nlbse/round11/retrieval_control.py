"""Control for training a new scalar calibration of vulnerability logits."""
from pathlib import Path
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round10"))
from neighbor_eval import OUT, ROOT, FAMILIES, dataset, table, cutoff, receipt_for
from students import PUBLIC, load, group_codes, metric, write_json
from round2_report import bootstrap_delta


def reshape(score):
    return logit(score.clip(1e-6,1-1e-6)).reshape(-1,1)


def main():
    _,_,_,y,ids,_=dataset(full=True)
    with np.load(ROOT/"round8/frozen/holdout.npz") as a:
        if a["rows"].tolist()!=ids:
            raise ValueError("Unexpected calibration indices")
        fold=a["folds"]
    fit,cal=fold<=2,fold==3
    _,_,_,public,public_ids,_=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    metrics={}
    for family in FAMILIES:
        root,selected=receipt_for(family)
        stem=Path(selected["checkpoint"]).stem
        with np.load(root/f"{stem}_holdout.npz") as raw:
            if raw["ids"].tolist()!=ids:
                raise ValueError("Unexpected encoder calibration indices")
            x=reshape(raw["p"][:,0])
        model=LogisticRegression(C=.2,max_iter=400).fit(x[fit],y[fit,0])
        threshold=cutoff(y[cal],model.predict_proba(x[cal])[:,1])
        before=table(root/"public/predictions.csv")
        after=table(OUT/family/"predictions.csv")
        if before[:,0].tolist()!=public_ids or after[:,0].tolist()!=public_ids:
            raise ValueError("Unexpected public indices")
        calibration=model.predict_proba(reshape(before[:,1]))[:,1]>threshold
        base=np.column_stack((calibration,after[:,3])).astype(bool)
        fused=after[:,2:].astype(bool)
        metrics[family]=dict(scalar_calibration=metric(public,base),retrieval=metric(public,fused),
            improvement_interval=bootstrap_delta(public,base,fused,groups),scalar_threshold=threshold)
    write_json(OUT/"calibration_control.json",metrics)
    print(metrics,flush=True)


if __name__=="__main__":
    main()
