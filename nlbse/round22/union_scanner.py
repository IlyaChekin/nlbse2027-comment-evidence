"""Generate a scanner that stops after the first exact rule hit, then verify its ternary outcome."""
import argparse
import json
from pathlib import Path
import re
from statistics import median
import time

import numpy as np

from project_eval import HERE, ROOT, f, checksum, json_at, raw_values, write_json
from train import cohorts, fit_ids

SOURCE = ROOT/"round20/scan.mjs"
PROGRAM = HERE/"scan_union.mjs"
CONFIG = HERE/"union_scanner.json"


def build():
    reference = json_at(f.HERE/"factor_numeric/official/policy.json")
    if checksum(SOURCE)!=reference["scanner_sha256"]:
        raise ValueError("Round 20 scan source changed")
    source = SOURCE.read_text()
    tags = [x.removeprefix("rule_") for x in json_at(f.SOURCE/"policy.json")["rule_columns"]] + ["195"]
    timing_path = f.HERE/"subsets_numeric/cost_model.json"
    timings = json_at(timing_path)["runs"]
    startup = median(timings["startup"])
    costs = np.asarray([max(.025,median(timings[t])-startup) for t in tags])
    ids = fit_ids("official")
    raw = raw_values(ids)
    metrics = (raw>0).astype(bool)
    remaining = np.ones(len(raw),dtype=bool)
    left = set(range(len(tags)))
    order,coverage = [],{}
    while left:
        best = max(left,key=lambda i:(int((remaining&metrics[:,i]).sum())/costs[i],-i))
        order.append(tags[best])
        coverage[tags[best]]=int((remaining&metrics[:,best]).sum())
        remaining &= ~metrics[:,best]
        left.remove(best)
    front = source.index("const published =")
    back = source.index("const modules =")
    source = source[:front]+f"const active = {json.dumps(order)};\n"+source[back:]
    source = source.replace('const numeric = process.argv.includes("--bounded-numeric");',"const numeric = true;")
    front = source.index("  return [row, ...rules.map(check => {")
    back = source.index("\n}))",front)
    source = source[:front]+'''  let available = true;
  for (const check of rules) {
    try {
      const lines = check(body.slice(), comments).lineNumbers;
      if (!Array.isArray(lines)) available = false;
      else if (lines.length > 0) return [row, 1];
    } catch {
      available = false;
    }
  }
  return [row, available ? 0 : -1];'''+source[back:]
    source = source.replace("...active.map(() => -1)", "-1").replace(
        '["row", ...active.map(n => "rule_" + n)]','["row", "rule_union"]')
    source = source.replace('"../round19/weak_satd"','"../round19/weak_satd"')
    PROGRAM.write_text(source)
    write_json(CONFIG,dict(scanner_source_sha256=checksum(SOURCE),program_sha256=checksum(PROGRAM),
                           timing_sha256=checksum(timing_path),order=order,incremental_hits=coverage,
                           fit_rows=len(ids),estimated_costs=dict(zip(tags,costs.tolist(),strict=True))))


def scan(texts):
    cols,raw = f.scanner(texts,program=PROGRAM)
    if cols!=["rule_union"] or not np.isin(raw,(-1,0,1)).all():
        raise ValueError("The shortcut scanner changed its ternary contract")
    return raw[:,0]


def verify():
    config = json_at(CONFIG)
    if checksum(PROGRAM)!=config["program_sha256"]:
        raise ValueError("Generated shortcut changed")
    expected,_,cohort_sha = cohorts()
    texts,_,_,_,_ = f.inputs()
    result = np.empty(len(texts),dtype=np.int8)
    elapsed = time.perf_counter()
    for i in range(0,len(result),16000):
        segment = texts[i:i+16000]
        result[i:i+len(segment)] = scan(segment)
        end = i+len(segment)
        print(end,int((result[:end]!=expected[:end]).sum()),time.perf_counter()-elapsed,flush=True)
    failures = np.flatnonzero(result!=expected)
    dest = HERE/"union_verification.npz"
    np.savez_compressed(dest,output=result,mismatches=failures)
    write_json(HERE/"union_verification.json",dict(rows=len(result),matches=len(result)-len(failures),
                  differences=len(failures),time_sec=time.perf_counter()-elapsed,cohorts_sha256=cohort_sha,
                  program_sha256=checksum(PROGRAM),predictions_sha256=checksum(dest),
                  mismatch_rows=failures.tolist()[:250]))
    if len(failures):
        raise ValueError("Short-circuit outcomes differ from the full-scan cache")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("build","verify"))
    {"build":build,"verify":verify}[p.parse_args().stage]()
