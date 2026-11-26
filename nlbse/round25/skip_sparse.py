"""Tune length-based decisions to skip the existing sparse correction."""
import argparse
from pathlib import Path
from statistics import median
import time

import joblib
import numpy as np

from length_eval import HERE,R24,PARTS,f,checksum,json_at,write_json,source_contexts,match
from sparse_model import aligned
from fuse_sparse import calibrate_sparse

OUT=HERE/"skip_sparse"
RANGES=((0,128),(0,256),(0,512),(0,768),(128,512),(256,512),(512,10**8),(768,10**8))


def tuning(rotation):
    tr,tt,fit,cal,review=source_contexts(rotation)
    if tt is None:
        tt=tr
    root=OUT/rotation
    root.mkdir(parents=True,exist_ok=True)
    options={}
    for low,high in RANGES:
        on=tr.active&(tr.lengths>low)&(tr.lengths<=high)
        outside=tt.active&(tt.lengths>low)&(tt.lengths<=high)
        y=tr.frame["y"]
        replaced=tr.state["p"]
        source=tr.bits[cal&on]
        fp=int((source&(y[cal&on]==0)).sum())
        _,cut=f.strict_budget(replaced[cal&on],y[cal&on],fp)
        changed=tr.bits.copy()
        changed[on]=replaced[on]>cut
        metrics=f.comparison(y[cal],tr.bits[cal],changed[cal])
        pred=tt.bits.copy()
        pred[outside]=tt.state["p"][outside]>cut
        keep=np.ones(len(pred),dtype=bool) if review is None else review
        review_metrics=f.comparison(tt.frame["y"][keep],tt.bits[keep],pred[keep])
        name=f"{low}-{high if high<10**8 else 'max'}"
        record=dict(lo=low,hi=high,cutoff=cut,calibration=metrics,frozen=review_metrics,
                    eval_skips=int((outside&keep).sum()),cal_skips=int((on&cal).sum()))
        if review is None:
            obs,by=match(tt.frame["y"],tt.frame["project"],tt.bits,outside,tt.state["p"][outside])
            record["matched"]=f.comparison(tt.frame["y"],tt.bits,obs)
            record["per_project"]=by
        else:
            # Profile the avoided sparse processing, including vectorization.
            package=joblib.load(R24/"official/tuned/models.joblib")["token"]
            text=[s for s,k in zip(tr.texts,on&cal,strict=True) if k]
            times=[]
            nnz=0
            if text:
                for _ in range(4):
                    start=time.perf_counter()
                    sparse,nnz=aligned(package["expert"],text,"token")
                    calibrate_sparse(package["oof"],sparse)
                    times.append(time.perf_counter()-start)
            ms=median(times) if times else 0.
            quality=.3*(metrics["f1_after"]-metrics["f1_before"])
            saved=.2*f.PUBLIC_N/cal.sum()*(ms/1000+4*nnz/3e15)
            record.update(proxy_gain=float(quality+saved),f1_change=quality,saved_time=ms)
        options[name]=record
        np.savez_compressed(root/f"{name}.npz",ids=tt.frame["ids"][keep],control=tt.bits[keep],frozen=pred[keep],
                            skipped=outside[keep])
        print(rotation,name,metrics["delta_tp"],record.get("matched",review_metrics)["delta_tp"],
              review_metrics["delta_tp"],review_metrics["delta_fp"],record.get("proxy_gain",0.),flush=True)
    write_json(root/"results.json",dict(options=options,cal_rows=int(cal.sum()),code_sha256=checksum(Path(__file__))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("rotation",choices=(*PARTS,"official"))
    tuning(p.parse_args().rotation)
