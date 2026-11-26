"""Report paired, project-stratified file bootstrap intervals for the fixed detector."""

import csv
from collections import defaultdict

import numpy as np

from audit import HERE, dataset, write_json


def sample_stats(path, paths, draws=2000):
    groups = defaultdict(lambda: defaultdict(lambda: np.zeros(6,dtype=np.int32)))
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            label,base,candidate=[row[key]=="1" for key in ("label","base","fixed")]
            stats=groups[row["project"]][paths[int(row["row"])]]
            stats += (label and base, not label and base, label and not base,
                      label and candidate, not label and candidate, label and not candidate)
    records=[np.vstack(list(files.values())) for files in groups.values()]
    rng=np.random.default_rng(42)
    values=np.empty((draws,2))
    for draw in range(draws):
        before,after=np.asarray([r[rng.integers(len(r),size=len(r))].sum(axis=0) for r in records]).sum(axis=0).reshape(2,3)
        values[draw]=(after[0]-before[0],2*after[0]/max(after[0]+after.sum(),1)-2*before[0]/max(before[0]+before.sum(),1))
    delta_tp,delta_f1=np.quantile(values,[.025,.975],axis=0).T.tolist()
    return dict(samples=draws,files=sum(len(g) for g in records),delta_tp=delta_tp,delta_f1=delta_f1)


def main():
    _,_,_,paths=dataset()
    result={name:sample_stats(HERE/f"{name}_marker_geometry.csv",paths) for name in ("test_a","test_b")}
    write_json(HERE/"intervals.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
