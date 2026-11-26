"""Account for remaining errors and benefits of the frozen rule policy."""
import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def data(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def count(y, p):
    tp = sum(a and b for a,b in zip(y,p,strict=True))
    fp = sum(not a and b for a,b in zip(y,p,strict=True))
    fn = sum(a and not b for a,b in zip(y,p,strict=True))
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fn+fp,1))


def public():
    folder = HERE/"official"
    env = json.loads((folder/"policy.json").read_text())
    perf = json.loads((folder/"results.json").read_text())[env["selected"]]
    before = data(folder/"reference_public.csv")
    after = data(folder/f"{env['selected']}_public.csv")
    by_band = {}
    for a,b in zip(before,after,strict=True):
        if any(a[k]!=b[k] for k in ("row","label_vulnerability","label_mat","mat")):
            raise ValueError("Published entries changed during the audit")
        p = float(a["probability"])
        band = "below_gate" if p<=.08 else "above_gate" if p>=.96 else "inside_gate"
        y,old,new = (b["label_vulnerability"]=="1",a["vulnerability"]=="1",b["vulnerability"]=="1")
        by_band.setdefault(band,[]).append((y,old,new))
    summary = {}
    for band, rows in by_band.items():
        y,old,new=zip(*rows,strict=True)
        summary[band]=dict(rows=len(rows), before=count(y,old), after=count(y,new),
            repaired=sum(a and not b and c for a,b,c in rows),
            newly_missed=sum(a and b and not c for a,b,c in rows),
            corrected_fp=sum(not a and b and not c for a,b,c in rows),
            new_fp=sum(not a and not b and c for a,b,c in rows))
    y=[r["label_vulnerability"]=="1" for r in after]
    a=[r["vulnerability"]=="1" for r in before]
    b=[r["vulnerability"]=="1" for r in after]
    oracle=[x if x==truth else z for x,z,truth in zip(a,b,y,strict=True)]
    best=count(y,oracle)
    cost=perf["cost"]
    mat=perf["quality"]["MAT"]["f1"]
    efficiency=.2*(1-cost["forward_seconds"]/1000)+.2*(1-cost["analytic_gflops"]/3_000_000)
    target_mat=(.97-efficiency)/.3-perf["quality"]["vulnerability"]["f1"]
    return dict(by_gate_band=summary, frozen_pair_oracle=best,
                oracle_projection=.3*(best["f1"]+mat)+efficiency,
                cost_assumption=cost,
                additional_mat_tp_for_target=math.ceil(506*target_mat/(2-target_mat))-443,
                required_vulnerability_f1_at_fixed_mat=(.97-efficiency)/.3-mat)


def heldout():
    path = HERE/"heldout_parallel.csv"
    rules={int(row["row"]):any(int(v)>0 for k,v in row.items() if k.startswith("rule_"))
           for row in data(path)}
    result={}
    for role in ("test_a","test_b"):
        rows=data(HERE/role/"reference_test.csv")
        low=[row for row in rows if float(row["p_base"])<=.08]
        y=[row["label"]=="1" for row in low]
        hit=[rules[int(row["row"])] for row in low]
        result[role]=dict(below_gate_rows=len(low),weak_rules=count(y,hit))
    return result


if __name__=="__main__":
    out=dict(public=public(),heldout=heldout())
    (HERE/"error_audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
