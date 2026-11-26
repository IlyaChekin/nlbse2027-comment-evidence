"""Learn query utility from actual decision switches in out-of-file forecasts."""
import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from policy45 import (HERE,ROOT,DEST as FIT,PARTS,f,checksum,json_at,load,parts,
                      write_json,experts,query_model,priority,MODES)

DEST = HERE/'switch'
MIX = .75


def main(rotation):
    start = perf_counter()
    frames,t,stamp = load(rotation)
    fit = FIT/rotation
    certificate = fit/'receipt.json'
    receipt = json_at(certificate)
    assert receipt['matrix_sha256']==stamp
    assert receipt['code_sha256']==checksum(HERE/'policy45.py')
    for f,known in receipt['files'].items():
        assert checksum(fit/f)==known
    train = frames['oof']
    x = np.load(fit/'crossfit.npz')
    assert np.array_equal(x['ids'],train['ids'])
    assert np.array_equal(x['fold'],train['fold'])
    assert np.array_equal(x['y'],train['y'])
    pair = parts(frames)
    old_cache = np.load(fit/'scores.npz')
    reference = (1.-MIX)*train['signed']+MIX*x['numeric']
    base = reference>0.
    y,accept = x['y'],x['eligible']
    data = np.column_stack((experts(train,t),reference,x['numeric'],reference-train['signed']))
    controls = []
    output = {}
    for d,p in zip(pair,('cal','held'),strict=True):
        assert np.array_equal(d['ids'],old_cache[p+'_ids'])
        u = old_cache[p+'_numeric']
        s = (1.-MIX)*d['signed']+MIX*u
        controls.append((s,np.column_stack((experts(d,t),s,u,s-d['signed']))))
        output[p+'_ids'] = d['ids']
    stats,heads = {},{}
    for arm in MODES:
        alt = (1.-MIX)*train['signed']+MIX*x[arm]
        change = (alt>0.)!=base
        gain = (y!=base)
        net = {'up': (change & ~base), 'down': (change & base)}
        stats[arm] = {k:dict(cases=int((z&accept).sum()),good=int((z&accept&gain).sum()),
                            harm=int((z&accept&~gain).sum())) for k,z in net.items()}
        mdl = query_model(data,base,gain,change,accept)
        heads[arm] = mdl
        for (score,linear),name in zip(controls,('cal','held'),strict=True):
            output[name+'_gate_'+arm] = priority(mdl,linear,score,receipt['trade'],receipt['queries'][arm]['cost'])
    out = DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'scores.npz',**output)
    joblib.dump(heads,out/'weights.joblib',compress=3)
    write_json(out/'receipt.json',dict(fit_sha256=checksum(certificate),matrix_sha256=stamp,
               mix=MIX,targets=stats,scores_sha256=checksum(out/'scores.npz'),
               weights_sha256=checksum(out/'weights.joblib'),seconds=perf_counter()-start,
               code_sha256=checksum(Path(__file__))))
    print(rotation,round(perf_counter()-start,2),stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
