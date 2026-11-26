"""Trace the first-pass probability and whether the compact encoder reread the function."""
import argparse

import numpy as np
import torch

from points33 import HERE, ROOT, PARTS, OUT, f, checksum, json_at, write_json, signed_source
from experiment33 import read_phase
from rule_model import RuleEncoder
from student_runtime import predict

DEST=HERE/'views'


def trace(model,tok,texts):
    raw,aux,_,_=predict(model,tok,texts,())
    policy=f.pinned()[0]
    score=raw[:,0]
    return np.column_stack((score,aux[:,0],(score>policy['lower']) & (score<policy['upper'])))


def main(rotation):
    folder=OUT/rotation
    signed=json_at(folder/'receipt.json')
    names=('dev','test') if rotation!='official' else ('dev',)
    corpus,_,_,_,_=f.inputs()
    tok=f.small.tokenizer()
    model=f.pipe(rotation,tok).encoder
    data={}
    for key in names:
        item=read_phase(key,folder,signed)
        ids=item['ids']
        x=trace(model,tok,[corpus[i] for i in ids])
        if not np.allclose(x[:,1],item['pre'][:,1],atol=.004,rtol=0.):
            raise ValueError('Auxiliary probability differs from the signed condition')
        data[key+'_ids']=ids
        data[key]=x[:,[0,2]]
    src,source_hash=signed_source(rotation)
    bank=read_phase('oof',folder,signed)
    assert np.array_equal(bank['ids'],src['ids']) and np.array_equal(bank['fold'],src['fold'])
    src_dir=ROOT/'round26/compact'/rotation
    config=json_at(src_dir/'manifest.json')
    assert config['sparse_oof_sha256']==source_hash
    x=np.empty((len(bank['ids']),3),dtype=np.float32)
    model=RuleEncoder(f.small.Encoder(len(tok),config['width']),conditional=True).cuda().eval()
    for block in model.encoder.local:
        block.use_shifted=True
    for fold in np.unique(bank['fold']):
        ck=src_dir/f'fold_{fold}.pt'
        assert checksum(ck)==config['folds'][str(fold)]['sha256']
        model.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'))
        use=bank['fold']==fold
        x[use]=trace(model,tok,[corpus[i] for i in bank['ids'][use]])
        print(rotation,int(fold),int(use.sum()),flush=True)
    assert np.allclose(x[:,1],bank['pre'][:,1],atol=.004,rtol=0.)
    data.update(oof=x[:,[0,2]],oof_ids=bank['ids'])
    dest=DEST/rotation
    dest.mkdir(parents=True,exist_ok=True)
    file=dest/'scores.npz'
    np.savez_compressed(file,**data)
    write_json(dest/'receipt.json',dict(code_sha256=checksum(HERE/'views33.py'),points_sha256=checksum(folder/'receipt.json'),
               source_sha256=source_hash,predictions_sha256=checksum(file),
               size={k:int((v[:,1]>.5).sum()) for k,v in data.items() if not k.endswith('_ids')}))
    print(rotation,'measurements complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rotation',choices=PARTS)
    main(p.parse_args().rotation)
