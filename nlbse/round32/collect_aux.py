"""Collect held-file predictions of individual scanner rules from already fitted encoders."""
import argparse

import numpy as np
import torch

from context32 import HERE, ROOT, PARTS, f, checksum, json_at, write_json, reference, signed_source
from rule_model import RuleEncoder
from student_runtime import predict

DEST = HERE/'aux'


def collect(rotation):
    saved = ROOT/'round26/compact'/rotation
    meta = json_at(saved/'manifest.json')
    path = saved/'scores.npz'
    source, source_sha = signed_source(rotation)
    if source_sha!=meta['sparse_oof_sha256'] or checksum(path)!=meta['sha256']:
        raise ValueError('Outside-file scores changed')
    mask = source['weak']==1
    ids, folds = source['ids'][mask],source['fold'][mask]
    corpus = f.inputs()[0]
    tok = f.small.tokenizer()
    model = RuleEncoder(f.small.Encoder(len(tok),meta['width']),conditional=True).cuda().eval()
    for block in model.encoder.local:
        block.use_shifted = True
    forecast = np.empty((len(ids),20),np.float32)
    for fold in np.unique(folds):
        path = saved/f'fold_{fold}.pt'
        if checksum(path)!=meta['folds'][str(fold)]['sha256']:
            raise ValueError('Out-of-file compact weights changed')
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        use = folds==fold
        _,aux,_,_ = predict(model,tok,[corpus[i] for i in ids[use]],())
        forecast[use] = aux[:,:20]
        print(rotation,'outside-file',int(fold),int(use.sum()),flush=True)
    del model
    with np.load(saved/'scores.npz') as z:
        mismatch = abs(forecast[:,0]-z['values'][mask,1])
        if not np.allclose(mismatch,0.,atol=.006):
            raise ValueError(f'Independent scanner predictions fail to replay {mismatch.max():.6g}')
    dev,test,*_ = reference(rotation)
    model = f.pipe(rotation,tok).encoder
    values = dict(ids=ids,fold=folds,rules=forecast)
    for name,ctx in (('dev',dev),('test',test)):
        if name=='test' and test is dev:
            continue
        use = ctx.state['known'] & ctx.state['flag']
        _,aux,_,_ = predict(model,tok,[s for s,k in zip(ctx.texts,use,strict=True) if k],())
        if not np.allclose(aux[:,0],ctx.frame['reference'][use,1],atol=.006):
            raise ValueError('Auxiliary model did not reproduce the reference scores')
        values[name] = aux[:,:20]
        values[name+'_ids'] = ctx.frame['ids'][use]
    folder = DEST/rotation
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'scores.npz'
    np.savez_compressed(path,**values)
    write_json(folder/'receipt.json',dict(predictions_sha256=checksum(path),oof_proof_sha256=checksum(saved/'manifest.json'),
               code_sha256=checksum(HERE/'collect_aux.py'),round26_max_error=float(mismatch.max())))
    print(rotation,'complete',len(ids),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    collect(parser.parse_args().rotation)
