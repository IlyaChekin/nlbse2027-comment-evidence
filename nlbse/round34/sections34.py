"""Score alternate regions with the archived file-disjoint UniXcoder teachers."""
import argparse
from bisect import bisect_left
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
R33=ROOT/'round33'
sys.path.insert(0,str(R33))
from points33 import PARTS, OUT as POINTS, f, checksum, json_at, write_json
from experiment33 import read_phase
from embed import CAP, MODEL
from section_runtime import opening
from students import load_model, lexical_features, batch, batch_plan, batch_gflops, ordered_batches

DEST=HERE/'regions'
NAMES=('head','body','middle','tail')
SPANS=1.8
RETAIN=40
R30=ROOT/'round30'


def support(data):
    p,q=data['unqueried'],data['queried']
    return (data['flag']==0) & data['old'] & (((p>0.) & (q<=0.)) | ((q>0.) & (q<SPANS)))


def layouts(text,tokens,offsets):
    body=bisect_left([start for start,_ in offsets],opening(text))
    cap=CAP-4
    head=tokens[:cap]
    # Retain the leading context when moving to the middle or end of long functions.
    other=cap-RETAIN
    if len(tokens)>cap:
        middle=max(RETAIN,(len(tokens)-other)//2)
        center=tokens[:RETAIN]+tokens[middle:middle+other]
        tail=tokens[:RETAIN]+tokens[-other:]
    else:
        center=tail=head
    return (head,tokens[body:body+cap],center,tail)


def prepare(texts,tok):
    items=tok(texts,truncation=False,add_special_tokens=False,return_offsets_mapping=True,verbose=False)
    mark=tok.convert_tokens_to_ids('<encoder-only>')
    if mark==tok.unk_token_id:
        raise ValueError('Encoder-only token is missing')
    header=[tok.bos_token_id,mark,tok.eos_token_id]
    seq=[]
    original=[]
    ix=np.empty((len(texts),len(NAMES)),np.int32)
    lengths=np.empty(len(texts),np.int32)
    for i,(text,tokens,offsets) in enumerate(zip(texts,items['input_ids'],items['offset_mapping'],strict=True)):
        windows={}
        lengths[i]=len(tokens)
        for col,contents in enumerate(layouts(text,tokens,offsets)):
            key=tuple(contents)
            if key not in windows:
                windows[key]=len(seq)
                seq.append([*header,*contents,tok.eos_token_id])
                original.append(i)
            ix[i,col]=windows[key]
    side=lexical_features(texts)[original]
    return seq,side,ix,lengths


@torch.inference_mode()
def measure(model,tok,texts,balance):
    scores=np.empty((len(texts),len(NAMES)),dtype=np.float32)
    lengths=np.empty(len(texts),dtype=np.int32)
    start=perf_counter()
    elapsed,gflops,count=0.,0.,0
    def load(offset):
        return prepare(texts[offset:offset+1024],tok)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset,(seq,side,ix,tokens) in ordered_batches(range(0,len(texts),1024),load,True):
            pred=np.empty(len(seq),np.float32)
            lex=torch.as_tensor(side,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,96):
                a,m=batch(group,seq,tok.pad_token_id)
                start_forward=perf_counter()
                pred[group]=model(a,m,lex[group])[:,0].float().cpu().numpy()-np.log(balance)
                elapsed+=perf_counter()-start_forward
                gflops+=batch_gflops('codebert',model,*a.shape)
            scores[offset:offset+len(ix)]=pred[ix]
            lengths[offset:offset+len(ix)]=tokens
            count+=len(seq)
    return scores,lengths,dict(seconds=elapsed,wall_seconds=perf_counter()-start,gflops=gflops,windows=count)


def checkpoint(net, folder, meta, fold):
    path=folder/f'upper_{fold}.pt'
    proof=meta['folds'][str(fold)]
    if checksum(path)!=proof['sha256']:
        raise ValueError('The teacher weights changed')
    parts=torch.load(path,map_location='cpu',weights_only=True)
    x=net.load_state_dict(parts,strict=False)
    if x.unexpected_keys:
        raise ValueError('The original teacher layers differ from the checkpoint')
    net.eval()
    return proof['balance']


def collect(rotation):
    ref=POINTS/rotation
    decision=json_at(ref/'receipt.json')
    src=R30/'tuned_teacher'/rotation
    config=json_at(src/'fit.json')
    if checksum(src/'oof.npz')!=config['predictions_sha256']:
        raise ValueError('The original teacher OOF table changed')
    if checksum(MODEL/'pytorch_model.bin')!=config['upstream_sha256']:
        raise ValueError('The pinned teacher base weights changed')
    texts,truth,_,_,_=f.inputs()
    bank=read_phase('oof',ref,decision)
    scope=support(bank)
    ids=bank['ids'][scope]
    folds=bank['fold'][scope]
    assert np.array_equal(bank['y'][scope],truth[ids,0])
    with np.load(src/'oof.npz') as proof:
        agreed,i,j=np.intersect1d(ids,proof['ids'],return_indices=True)
        assert len(agreed)==len(ids) and np.array_equal(folds[i],proof['fold'][j])
        prior=np.zeros(len(ids),np.float32)
        prior[i]=proof['raw'][j,2]
    net,tok=load_model('unixcoder9')
    rendered=np.empty((len(ids),len(NAMES)),np.float32)
    counts=np.empty(len(ids),np.int32)
    timing={}
    for fold in np.unique(folds):
        picked=folds==fold
        balance=checkpoint(net,src,config,int(fold))
        eligible_texts=[texts[i] for i in ids[picked]]
        s,n,elapsed=measure(net,tok,eligible_texts,balance)
        rendered[picked],counts[picked]=s,n
        timing[str(int(fold))]=elapsed
        print(rotation,'fold',int(fold),'rows',int(picked.sum()),elapsed,flush=True)
    delta=np.abs(prior-rendered[:,0])
    if np.quantile(delta,.999)>.035:
        raise ValueError(f'Head output differs from whole teacher observed OOF ({np.quantile(delta,.999)})')
    actual=dict(oof_ids=ids,oof=rendered,oof_lengths=counts)
    balance=checkpoint(net,src,config,0)
    for part in ('dev','test'):
        if part not in decision['phases']:
            continue
        view=read_phase(part,ref,decision)
        use=support(view)
        found=view['ids'][use]
        v,n,performance=measure(net,tok,[texts[i] for i in found],balance)
        actual.update({part+'_ids':found,part:v,part+'_lengths':n})
        timing[part]=performance
    base=DEST/rotation
    base.mkdir(parents=True,exist_ok=True)
    archive=base/'scores.npz'
    np.savez_compressed(archive,**actual)
    output=dict(rotation=rotation,names=NAMES,span=SPANS,fit_rows=len(ids),times=timing,
                head_absolute_error=np.quantile(delta,[.5,.95,.999,1.]).tolist(),
                teacher_sha256=config['predictions_sha256'],teacher_recipe_sha256=checksum(src/'fit.json'),
                points_sha256=checksum(ref/'receipt.json'),predictions_sha256=checksum(archive),
                code_sha256=checksum(HERE/'sections34.py'))
    write_json(base/'receipt.json',output)
    print(rotation,'complete',output['head_absolute_error'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rotation',choices=PARTS)
    collect(parser.parse_args().rotation)
