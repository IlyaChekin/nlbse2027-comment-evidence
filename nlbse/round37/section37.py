"""Read declaration and body evidence without repeating the compact encoder."""
import argparse
from bisect import bisect_left
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'round35'))
from teacher35 import PARTS, f, checksum, json_at, write_json, opening
from eval35 import loaded, reference
from broad35 import regimes
from predict_factor import lexical_features, ordered_batches, batch_plan, batch, small
from rule_model import RuleEncoder, pooling

STATE = HERE/'views'
SEAM_TOKENS = 48
NATIVE = 23
SECTIONS = 6
OUTPUTS = 4


def prepare(texts, tok):
    seq, boundaries = [], []
    for start in range(0,len(texts),1024):
        items = texts[start:start+1024]
        obj = tok(items,truncation=False,verbose=False,return_offsets_mapping=True)
        for text,ids,offsets in zip(items,obj['input_ids'],obj['offset_mapping'],strict=True):
            end = opening(text)
            # A zero offset span is a special token. Keep the original sequence intact.
            spans = [(i, low) for i,(low,high) in enumerate(offsets) if high>low]
            if not spans:
                seq.append(ids)
                boundaries.append((0,0,0))
                continue
            left = spans[0][0]
            right = spans[-1][0]+1
            boundary = bisect_left([x[1] for x in spans],end)
            cut = spans[boundary][0] if boundary < len(spans) else right
            seq.append(ids)
            boundaries.append((left,right,cut))
    return seq, np.asarray(boundaries,dtype=np.int32), lexical_features(texts)


def section_scores(hidden, mask, side, model, limits, main, aux):
    pos = torch.arange(hidden.shape[1],device=hidden.device)[None,:]
    start,stop,cut = limits.T[:,:,None]
    src = (pos>=start)&(pos<stop)&mask.bool()
    middle = (start+stop)//2
    splits = torch.cat((cut,middle,torch.minimum(stop,cut+SEAM_TOKENS)),dim=1)
    left = src[:,None,:] & (pos[:,None,:]<splits[:,:,None])
    right = src[:,None,:] & ~left
    group = torch.stack((left,right),dim=2).flatten(1,2)
    mass = group.sum(2)
    mean = group.to(hidden.dtype)@hidden/mass.clamp_min(1).unsqueeze(-1)
    native_pool = pooling(hidden,mask)
    # Preserve the native max term and substitute only the local mean. The
    # outputs keep their interpretation across file-disjoint checkpoints.
    width = hidden.shape[-1]
    global_mean,global_max = native_pool.split(width,dim=1)
    mean = torch.where(mass[:,:,None]>0,mean,global_mean[:,None,:]).to(global_max.dtype)
    joined = torch.cat((mean,global_max[:,None,:].expand_as(mean)),dim=2)
    lexical = side[:,None,:].expand(-1,SECTIONS,-1)
    local_main = model.encoder.head(torch.cat((joined,lexical),2))[:,:,0]
    local_aux = model.rule_head(joined)[:,:,[0,20,21]]
    head = torch.cat((local_main.unsqueeze(2),local_aux),dim=2)
    baseline = torch.cat((main[:,[0]],aux[:,[0,20,21]]),1)
    deltas = (head-baseline[:,None,:]).masked_fill(mass[:,:,None]==0,0.)
    score = torch.cat((main[:,[0]],aux,deltas.flatten(1),mass.float().log1p()),1)
    dimension = SECTIONS*4*width+2*SECTIONS*hidden.shape[1]*width+SECTIONS*(4*width*24+12*width)
    return score, hidden.shape[0]*dimension


@torch.inference_mode()
def collect(model, tok, texts, check_tokenization=False):
    result = np.empty((len(texts),NATIVE+SECTIONS*(OUTPUTS+1)),np.float32)
    start = perf_counter()
    def build(first):
        local = texts[first:first+1024]
        seq,lim,side = prepare(local,tok)
        if check_tokenization and first==0:
            assert seq == f.tokenize(local,tok)
        return seq,lim,side
    compute = 0.
    with torch.autocast('cuda',dtype=torch.float16):
        for first,(seq,limits,side) in ordered_batches(range(0,len(texts),1024),build,True):
            side = torch.as_tensor(side,device='cuda')
            sizes = torch.as_tensor(limits,device='cuda')
            for group in batch_plan(list(range(len(seq))),seq,small.BATCH):
                ids,mask = batch(group,seq,tok.pad_token_id)
                t = perf_counter()
                hidden = model.encoder.token_states(ids,mask)
                main = model.encoder.readout(hidden,mask,side[group])
                aux = model.rule_head(pooling(hidden,mask))
                scores,_ = section_scores(hidden,mask,side[group],model,sizes[group],main,aux)
                result[first+np.asarray(group)] = scores.float().cpu().numpy()
                compute += perf_counter()-t
    assert np.isfinite(result).all()
    return result, dict(forward_seconds=compute,wall_seconds=perf_counter()-start)


def oof_state(path,tok,manifest,fold):
    weight = path/f'fold_{fold}.pt'
    if checksum(weight)!=manifest['folds'][str(fold)]['sha256']:
        raise ValueError('Compact out-of-file weights changed')
    model = RuleEncoder(small.Encoder(len(tok),manifest['width']),conditional=True).cuda()
    model.load_state_dict(torch.load(weight,weights_only=True,map_location='cpu'))
    for block in model.encoder.local:
        block.use_shifted = True
    return model.eval()


def inputs(rotation):
    phases,ref,old = loaded(rotation)
    _,base = reference(rotation,phases,ref)
    cohorts = ROOT/'round35/broad'/rotation
    proof = json_at(cohorts/'results.json')
    signed = json_at(ROOT/'round35/broad/selection.json')
    assert checksum(cohorts/'results.json') == signed['result_sha256']['broad'][rotation]
    roster = cohorts/'cohorts.npz'
    assert checksum(roster)==proof['cohorts_sha256'] and proof['baseline_sha256']==base
    for row in phases.values():
        row['support']=row['old']&(row['flag']==0)
    oof = phases['oof']
    with np.load(roster) as z:
        for k,live in [('ids',oof['ids']),('fold',oof['fold']),('eligible',oof['support']),
                       ('regime',regimes(oof)),('truth',oof['y'])]:
            assert np.array_equal(live,z[k])
    return phases, ref, base, checksum(roster)


def main(rotation):
    phases,ref,base,cohorts = inputs(rotation)
    texts = f.inputs()[0]
    tok = small.tokenizer()
    checkpoint = ROOT/'round26/compact'/rotation
    manifest = json_at(checkpoint/'manifest.json')
    assert checksum(checkpoint/'scores.npz') == manifest['sha256']
    oof = phases['oof']
    ids,folds = [oof[k][oof['support']] for k in ('ids','fold')]
    rows = np.empty((len(ids),NATIVE+SECTIONS*(OUTPUTS+1)),np.float32)
    time = {}
    with np.load(checkpoint/'scores.npz') as x:
        assert all(np.array_equal(x[k],oof[k]) for k in ('ids','fold','y'))
        native = x['values'][oof['support']]
    for fold in np.unique(folds):
        use = folds==fold
        model = oof_state(checkpoint,tok,manifest,int(fold))
        rows[use],time[f'oof_{fold}'] = collect(model,tok,[texts[i] for i in ids[use]],True)
        del model
        print(rotation,'oof',int(fold),int(use.sum()),time[f'oof_{fold}'],flush=True)
    # Compare the native preexisting auxiliary outputs independently of the regional features.
    delta = np.abs((1./(1.+np.exp(-rows[:,[1,21,22]])))-native[:,1:])
    assert np.quantile(delta,.999) < .009
    payload = dict(oof_ids=ids,oof=rows)
    model = f.pipe(rotation,tok).encoder
    for phase in ('dev','test'):
        if phase in phases:
            sample = phases[phase]
            ids = sample['ids'][sample['support']]
            payload[phase+'_ids'] = ids
            payload[phase],time[phase] = collect(model,tok,[texts[i] for i in ids],True)
            print(rotation,phase,len(ids),time[phase],flush=True)
    dest = STATE/rotation
    dest.mkdir(parents=True,exist_ok=True)
    file = dest/'scores.npz'
    np.savez_compressed(file,**payload)
    write_json(dest/'receipt.json',dict(rotation=rotation,baseline_sha256=base,cohorts_sha256=cohorts,
                code_sha256=checksum(Path(__file__)),points=ref['phases'],state_sha256=checksum(file),
                compact_manifest_sha256=checksum(checkpoint/'manifest.json'),
                approx_quantile_error=float(np.quantile(delta,.999)),seam_tokens=SEAM_TOKENS,timings=time))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rotation',choices=PARTS)
    main(cli.parse_args().rotation)
