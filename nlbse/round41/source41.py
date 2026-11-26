"""Load signed Round 40 file folds without recomputing or reusing held labels."""
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
R40=ROOT/'round40'
sys.path.insert(0,str(R40))
from sparse40 import PARTS, f, checksum, json_at, write_json
from stack40 import assemble
from data38 import crop


def load(rotation):
    folder=R40/'matrices'/rotation
    manifest=folder/'receipt.json'
    proof=json_at(manifest)
    witness=json_at(R40/'selection.json')
    previous=R40/'comparison'/rotation/'results.json'
    assert checksum(previous)==witness['source'][rotation]
    assert proof['sources']==json_at(previous)['sources']
    assert proof['data_sha256']==checksum(R40/'data40.py')
    result={}
    for name,quality in proof['phases'].items():
        p=folder/f'{name}.npz'
        assert checksum(p)==quality['sha256']
        with np.load(p) as z:
            result[name]=dict(z)
        assert len(result[name]['ids'])==quality['rows']
    return result,proof['threshold'],checksum(manifest)


def supported(frame):
    return frame['observed_regime']!=1


def parts(data):
    dev=data['dev']
    test=data.get('test',dev)
    return crop(dev,dev['cal']),crop(test,test['held'])
