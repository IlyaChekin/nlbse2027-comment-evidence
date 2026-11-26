"""Freeze the assembled expert matrix for subsequent out-of-file studies."""
import argparse
import json
from pathlib import Path

import numpy as np

from data40 import HERE, ROOT, PARTS, DEST, attach, checksum, json_at, write_json, outside_scope


def main(rotation):
    rows,ref,source=attach(rotation)
    old=json_at(HERE/'comparison'/rotation/'results.json')
    assert json.loads(json.dumps(source))==old['sources']
    out=DEST/rotation
    out.mkdir(exist_ok=True,parents=True)
    stamps={}
    for section,x in rows.items():
        output=out/f'{section}.npz'
        cols=('ids','y','length','pre','old','unqueried','queried','flag','signed','sparse','new')
        keys=('fold','duplicates') if section=='oof' else ('fit','cal','held','project')
        payload={key:x[key] for key in (*cols,*keys) if key in x}
        if 'project' in payload:
            payload['project']=np.asarray(payload['project'],dtype=str)
        payload['observed_regime']=outside_scope(x)
        np.savez_compressed(output,**payload)
        stamps[section]=dict(rows=len(x['ids']),sha256=checksum(output))
    write_json(out/'receipt.json',dict(rotation=rotation,sources=source,phases=stamps,
                  threshold=ref['cutoff'],fields=dict(counterfactual=('queried','flag'),
                    real_time=('pre','unqueried','signed','sparse','new','observed_regime')),
                  code_sha256=checksum(Path(__file__)),data_sha256=checksum(HERE/'data40.py')))
    print(rotation,stamps,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    main(parser.parse_args().rotation)
