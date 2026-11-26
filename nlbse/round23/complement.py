"""Audit coverage of errors by fixed archived predictions using the gold labels only for diagnosis."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MODEL=ROOT/"round21/mat/profile_marker.csv"
AUDIT=ROOT/"round20/conditional_audit.csv"
PROFILE=ROOT/"round22/union_public/results.json"


def read(path):
    with path.open(newline="") as stream:
        values=list(csv.DictReader(stream))
    if any(int(row["row"])!=i for i,row in enumerate(values)):
        raise ValueError(f"Row alignment changed for {path}")
    return values


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def f1(y,p):
    a=sum(a and b for a,b in zip(y,p,strict=True))
    return 2*a/(sum(y)+sum(p))


def model_decisions(path):
    rows=read(path)
    field="pred_vulnerability" if "pred_vulnerability" in rows[0] else "vulnerability"
    return [bool(int(r[field])) for r in rows]


def main():
    old_paths=json.loads((ROOT/"round15/sensitivity.json").read_text())["archive_paths"]
    pieces={label: ROOT/name for label,name in old_paths.items()}
    current=read(MODEL)
    original=[bool(int(r["pred_vulnerability"])) for r in current]
    truth=[bool(int(r["label_vulnerability"])) for r in current]
    mat_truth=[bool(int(r["label_mat"])) for r in current]
    mat=[bool(int(r["pred_MAT"])) for r in current]
    audit=read(AUDIT)
    support=json.loads(PROFILE.read_text())
    if support["baseline_predictions_sha256"]!=digest(MODEL):
        raise ValueError("Current reference differs from the profiled public archive")
    if truth!=[bool(int(r["vulnerability_gold"])) for r in audit]:
        raise ValueError("Rule audit cohort differs from the fixed predictions")
    scope=["rule_present" if int(r["any_rule"]) else "rule_absent" if int(r["valid"]) else "unobserved"
           for r in audit]
    alternatives={name:model_decisions(path) for name,path in pieces.items()}
    cost=support["stats"]["short_circuit"]["cost"]
    def score(p):
        return .3*(f1(truth,p)+f1(mat_truth,mat))+.2*(1-cost["forward_seconds"]/1000)+.2*(1-cost["analytic_gflops"]/3e6)
    if abs(score(original)-support["stats"]["short_circuit"]["score"])>1e-12:
        raise ValueError("Archived decisions and measured official score disagree")

    def possible(m,choices):
        corrected=[]
        result=Counter()
        for i,(g,p) in enumerate(zip(truth,original,strict=True)):
            if p!=g:
                available=any(q[i]==g for q in choices)
                kind="missed_positive" if g else "false_positive"
                result[(scope[i],kind,"covered" if available else "uncovered")]+=1
                corrected.append(g if available else p)
            else:
                corrected.append(p)
        groups={}
        for (cat,kind,ins),value in result.items():
            groups.setdefault(cat,{}).setdefault(kind,{})[ins]=value
        return dict(models=m,score_at_reference_cost=score(corrected),vulnerability_f1=f1(truth,corrected),cases=groups)

    summary={name:possible(1,[q]) for name,q in alternatives.items()}
    full=possible(len(alternatives),list(alternatives.values()))
    result=dict(reference_score=score(original),reference_f1=f1(truth,original),scope="gold-picking diagnostic with unchanged measured costs",
                examples=len(original),fixed_mat_f1=f1(mat_truth,mat),portfolio=full,individual=summary,
                predictions_sha256={k:digest(p) for k,p in pieces.items()},
                exact_rule_audit_sha256=digest(AUDIT))
    (HERE/"complement.json").write_text(json.dumps(result,indent=2)+"\n")
    print(full,flush=True)


if __name__=="__main__":
    main()
