"""Compare audited predictions, experimental routing and the published score target."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"round10"))
from students import ROOT, PUBLIC, TRAIN, group_codes, load, metric, write_json
from round2_report import bootstrap_delta
from score_runtime import competition_formula
from neighbor_eval import table


def read(path):
    return json.loads(path.read_text())


def main():
    public,y,columns=load(PUBLIC,audit=True)
    train,_,catalog=load(TRAIN,audit=True)
    hashes={hashlib.sha256(t.encode()).digest() for t in train}
    clean=np.asarray([hashlib.sha256(t.encode()).digest() not in hashes for t in public])
    train_files=set(zip(catalog["Projectname"],catalog["Filepath"],strict=True))
    fresh_file=np.asarray([pair not in train_files for pair in zip(columns["Projectname"],columns["Filepath"],strict=True)])
    fresh_project=~np.isin(columns["Projectname"],list(set(catalog["Projectname"])))
    del train,public,hashes,train_files
    variants={}
    sensitivity={}

    def add(label,filepath,profile,category):
        saved=table(filepath)
        if not np.array_equal(saved[:,0],np.arange(len(y))):
            raise ValueError(f"Unexpected prediction rows for {label}")
        bits=saved[:,2:4].astype(bool)
        result=metric(y,bits)
        forward=profile["forward_seconds"]
        wall=profile.get("wall_seconds")
        gflops=profile["analytic_gflops"]
        variants[label]=dict(mode=label,kind=category,mean_f1=result["mean"]["f1"],
            vulnerability_f1=result["vulnerability"]["f1"],mat_f1=result["MAT"]["f1"],
            end_to_end_seconds=wall,forward_seconds=forward,estimated_gflops=gflops,
            local_formula=competition_formula(result["mean"]["f1"],forward,gflops),
            wall_charged_formula=competition_formula(result["mean"]["f1"],wall,gflops) if wall is not None else None)
        sensitivity[label]={key:dict(rows=int(mask.sum()),metrics=metric(y[mask],bits[mask]))
            for key,mask in (("no_exact_match",clean),("file_not_in_official_train",fresh_file),
                             ("project_not_in_official_train",fresh_project)) if mask.any()}

    previous=ROOT/"round10"
    for mode,place,receipt in (("round10_compact","compact_exit/lexical_deferred","compact_exit/lexical_deferred/analysis.json"),
                               ("round10_unix","unix_context/unix_lexical","unix_context/analysis.json")):
        old=read(previous/receipt)
        cost=(old if mode.endswith("compact") else old["unix_lexical"])["cost"]
        add(mode,previous/place/"predictions.csv",cost,"single_encoder")
    for width in (128,256):
        raw=HERE/("small" if width==128 else f"small{width}")
        cascade=HERE/("cascade" if width==128 else f"cascade{width}")
        add(f"local{width}",raw/"public/predictions.csv",read(raw/"public/analysis.json")["cost"],"single_encoder")
        for mode, r in read(cascade/"analysis.json").items():
            add(f"{mode}{width}",cascade/mode/"predictions.csv",r["cost"],
                "single_encoder" if mode=="small_lexical" else "two_encoders")
        adj=HERE/"neighbors"/f"small{width}"
        if (adj/"analysis.json").is_file():
            add(f"neighbor{width}",adj/"predictions.csv",read(adj/"analysis.json")["cost"],"combined_estimate")
    for continuation in ("matched", "gold_disagreement"):
        cascade=HERE/f"cascade256_{continuation}"
        if (cascade/"analysis.json").is_file():
            for mode,r in read(cascade/"analysis.json").items():
                add(f"{mode}256_{continuation}",cascade/mode/"predictions.csv",r["cost"],"single_encoder")
    for mode,r in read(HERE/"deep_exit/analysis.json").items():
        add(f"deep_exit_{mode}",HERE/"deep_exit"/mode/"predictions.csv",r["cost"],"single_encoder")
    deployment=HERE/"cold_shift"
    if (deployment/"analysis.json").is_file():
        add("release_shift",deployment/"predictions.csv",read(deployment/"analysis.json")["cost"],"deployment_variant")
    retrieved=HERE/"neighbors"
    for mode,r in read(retrieved/"analysis.json").items():
        add(f"retrieval_{mode}",retrieved/mode/"predictions.csv",r["cost"],"combined_estimate")
    with (HERE/"results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(next(iter(variants.values()))))
        writer.writeheader()
        writer.writerows(variants.values())
    winner=lambda candidates,key: max(candidates,key=lambda label:variants[label][key])
    tested=[k for k in variants if variants[k]["kind"] in {"single_encoder","two_encoders"}]
    unified=[k for k in tested if variants[k]["kind"]=="single_encoder"]
    result=dict(score_winner=winner(tested,"local_formula"),
                score_winner_single_encoder=winner(unified,"local_formula"),
                f1_winner=winner(list(variants),"mean_f1"),
                exact_train_match_rows=int((~clean).sum()),unseen_file_rows=int(fresh_file.sum()),
                unseen_project_rows=int(fresh_project.sum()),sensitivity=sensitivity)
    if "release_shift" in variants:
        result["release_mode"]="release_shift"
        result["release_at_measured_cost"]=variants["release_shift"]
    outcomes = [HERE/f"cascade256_{suffix}/small_lexical/predictions.csv"
                for suffix in ("matched", "gold_disagreement")]
    if all(path.is_file() for path in outcomes):
        bits = [table(path)[:,2:4].astype(bool) for path in outcomes]
        groups = group_codes(zip(columns["Projectname"], columns["Filepath"], strict=True))
        result["disagreement_training_delta_vs_matched"] = bootstrap_delta(y,*bits,groups)
    winner_row=variants[result["score_winner"]]
    volume,seconds=winner_row["estimated_gflops"],winner_row["forward_seconds"]
    result["required_mean_f1_at_winning_cost"]=(.97-competition_formula(0.,seconds,volume))/.6
    write_json(HERE/"summary.json",result)
    print({k:v for k,v in result.items() if k!="sensitivity"},flush=True)


if __name__=="__main__":
    main()
