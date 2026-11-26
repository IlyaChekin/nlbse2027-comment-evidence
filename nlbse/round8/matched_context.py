"""Remove only key/value context while fixing the token queries, pooling and readout."""
import argparse
import json
from pathlib import Path

import numpy as np

from late_tokens import (HERE, PORTABLE, THRESHOLDS, DeferredOffsets, bootstrap_delta, dataset,
                         group_codes, load, PUBLIC, load_checkpoint, metric, predict_bits,
                         predict_variant, save, write_json)

REFERENCE = "query_even_65"
MASKED = "drop_even_65"


def main(output):
    cfg = json.loads((HERE/"tokens_frozen/policy.json").read_text())
    original = cfg["configurations"][REFERENCE]
    cfg["configurations"][MASKED] = dict(original, spec=dict(original["spec"], attention="drop"))
    model,token=load_checkpoint(PORTABLE,"cuda")
    token=DeferredOffsets(token)
    _,_,texts,y,ids,cal=dataset(public=True)
    _,_,fields=load(PUBLIC,audit=True)
    groups=group_codes(zip(fields["Projectname"],fields["Filepath"],strict=True))
    bits, cost = {}, {}
    for name in (REFERENCE,MASKED):
        p,tally=predict_variant(model,token,texts,cfg,name)
        bits[name]=p>.5
        cost[name]=tally
        save(output/name,y,ids,cal,p,tally,(.5,.5))
    control=HERE/"tokens_final"/REFERENCE/"predictions.csv"
    if not np.array_equal(bits[REFERENCE],predict_bits(control,ids,{"thresholds":(.5,.5)})):
        raise ValueError("Fixed-query baseline differs from the interleaved public run")
    write_json(output/"analysis.json",dict(
        comparison="Only remove keys and values belonging to positions that are not updated",
        query_fraction=original["spec"]["keep"], thresholds=original["thresholds"],
        scores={key:metric(y,p) for key,p in bits.items()},
        retained_minus_dropped=bootstrap_delta(y,bits[MASKED],bits[REFERENCE],groups),
        cost=cost))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=HERE/"context_ablation")
    main(ap.parse_args().output)
