"""Extend the fixed UniXcoder teacher to all held-out consensus-negative rows."""
import argparse
from pathlib import Path

import numpy as np
import torch

from data38 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, phases, routes, pairs
from sections34 import MODEL, R30, checkpoint
from embed import tokens
from teacher35 import evaluate
from students import load_model, lexical_features

DEST = HERE/'negative_teacher'


def main(rotation):
    data,ref,base = phases(rotation)
    training = data['oof']
    texts = f.inputs()[0]
    source = R30/'tuned_teacher'/rotation
    proof = json_at(source/'fit.json')
    assert checksum(source/'oof.npz')==proof['predictions_sha256']
    assert checksum(MODEL/'pytorch_model.bin')==proof['upstream_sha256']
    active = routes(training)['negative']
    ids = training['ids'][active]
    summary = {}
    with np.load(source/'oof.npz') as z:
        common,a,b = np.intersect1d(ids,z['ids'],return_indices=True)
        assert len(common)>0
        assert np.array_equal(training['fold'][active][a],z['fold'][b])
        assert np.array_equal(training['y'][active][a],z['y'][b])
        oof = np.full(len(ids),np.nan,np.float32)
        oof[a] = z['raw'][b,2]
    pair_a,pair_b,pair_sha = pairs(rotation,training,'negative_consensus')
    ix = np.flatnonzero(active)
    assert np.isfinite(oof[np.searchsorted(ix,np.r_[pair_a,pair_b])]).all()
    result = dict(oof_ids=ids,fold=training['fold'][active],oof=oof)
    net,tok = load_model('unixcoder9')
    balance = checkpoint(net,source,proof,0)
    for role,frame in data.items():
        if role=='oof':
            continue
        ids = frame['ids'][routes(frame)['negative']]
        rows = [texts[i] for i in ids]
        seq = []
        for start in range(0,len(rows),1024):
            seq.extend(tokens(rows[start:start+1024],tok))
        side = torch.as_tensor(lexical_features(rows),device='cuda')
        result[role+'_ids'] = ids
        result[role] = evaluate(net,tok,seq,side,np.arange(len(ids)),np.log(balance))
        summary[role] = len(ids)
        print(rotation,role,len(ids),flush=True)
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    path = out/'scores.npz'
    np.savez_compressed(path,**result)
    write_json(out/'receipt.json',dict(rotation=rotation,base_sha256=base,pairs_sha256=pair_sha,
                  source_sha256=proof['predictions_sha256'],scores_sha256=checksum(path),
                  oof_coverage=int(np.isfinite(oof).sum()),oof_rows=len(oof),held=summary,code_sha256=checksum(Path(__file__))))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
