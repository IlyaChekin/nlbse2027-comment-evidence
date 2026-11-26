"""Run the round 21 model with the verified short-circuit static check."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/"round21/mat"))
from predict_mat import predict_factor, recover

PROGRAM = HERE/"scan_union.mjs"


def decisions(text, config, model, tok, heads):
    reference = json.loads((HERE/"union_scanner.json").read_text())
    predict_factor.require_hash(PROGRAM,reference["program_sha256"])
    alternate = config | {"program":str(PROGRAM),"rule_columns":["rule_union"]}
    pred = predict_factor.decisions(text,alternate,model,tok,heads)
    pred[:,1] = recover(text,pred[:,1])
    return pred


if __name__=="__main__":
    predict_factor.main(decisions)
