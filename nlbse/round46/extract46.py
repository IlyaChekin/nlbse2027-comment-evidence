"""Extract frozen representations and measure each encoder/window on the same rows."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModel, AutoConfig, RobertaModel

from views46 import HERE,ROOT,MODEL,UNIX,DEST as WINDOWS,CAPS,TYPES,f,checksum,json_at,source,write_json

DEST = HERE/'features'
BLOCK = 4096


def weights(kind):
    if kind=='qwen':
        _,known = source()
        file = MODEL/'model.safetensors'
        stamp = checksum(file)
        assert stamp==known[file.name]['lfs']['sha256'] and file.stat().st_size==known[file.name]['size']
        net = AutoModel.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',
                    local_files_only=True,trust_remote_code=False,use_safetensors=True)
        assert net.config.model_type=='qwen2'
        pad = net.config.eos_token_id
    else:
        file = UNIX/'pytorch_model.bin'
        stamp = checksum(file)
        ref = json_at(ROOT/'round32/dense/receipt.json')
        assert ref['source_sha256']==stamp
        config = AutoConfig.from_pretrained(UNIX,local_files_only=True,trust_remote_code=False)
        config._attn_implementation = 'sdpa'
        net = RobertaModel.from_pretrained(UNIX,config=config,local_files_only=True,
                                add_pooling_layer=False,dtype=torch.float16)
        pad = config.pad_token_id
    return net.cuda().eval(),int(pad),stamp


def operations(model,row,seq,causal):
    cfg = model.config
    d,layers,i = cfg.hidden_size,cfg.num_hidden_layers,cfg.intermediate_size
    if causal:
        k = d*cfg.num_key_value_heads/cfg.num_attention_heads
        # Lower triangular attention with fused multiply-add counted as two.
        per_layer = 2.*row*(2*seq*d*d+2*seq*d*k+3*seq*d*i+seq*(seq+1)*d)
    else:
        per_layer = 2.*row*(4*seq*d*d+2*seq*d*i+2*seq*seq*d)
    return layers*per_layer/1e9


def pool(hidden,mask):
    # Double-check right padding in the signed batch helper before using the final position.
    valid = mask.bool().unsqueeze(-1)
    mean = hidden.masked_fill(~valid,0.).sum(1,dtype=torch.float32)/valid.sum(1)
    highest = hidden.masked_fill(~valid,-torch.inf).amax(1)
    index = mask.sum(1).long()-1
    last = hidden[torch.arange(len(hidden),device=hidden.device),index]
    return torch.cat((mean,highest,last),1).half().cpu().numpy()


@torch.inference_mode()
def encode(seq,model,pad,kind,cap,out,mid=None):
    batch_size = (128 if cap==96 else 90) if kind=='qwen' else (192 if cap==96 else 104)
    begin = perf_counter()
    launch,calc = 0.,0.
    middle = []
    def capture(_layer,_arguments,response):
        middle.append(response if torch.is_tensor(response) else response[0])
    hook = model.layers[11].register_forward_hook(capture) if mid is not None else None
    try:
        for start in range(0,len(seq),BLOCK):
            block = seq[start:start+BLOCK]
            for ids in f.base.batch_plan(list(range(len(block))),block,batch_size):
                x,mask = f.base.batch(ids,block,pad)
                if not start and ids is not None:
                    assert torch.equal(mask.bool(),torch.arange(x.shape[1],device=x.device)[None,:]<mask.sum(1)[:,None])
                t = perf_counter()
                params = dict(input_ids=x,attention_mask=mask)
                if kind=='qwen':
                    params['use_cache'] = False
                outputs = model(**params).last_hidden_state
                result = pool(outputs,mask)
                if hook is not None:
                    assert len(middle)==1
                    state = pool(model.norm(middle.pop()),mask)
                launch += perf_counter()-t
                calc += operations(model,*x.shape,kind=='qwen')
                order = start+np.asarray(ids)
                out[order] = result
                if hook is not None:
                    mid[order] = state
            if start%(5*BLOCK)==0:
                print(kind,cap,min(start+BLOCK,len(seq)),round(perf_counter()-begin,2),flush=True)
    finally:
        if hook is not None:
            hook.remove()
    return dict(forward_seconds=launch,wall_seconds=perf_counter()-begin,analytic_gflops=calc)


def main(kind,cap):
    torch.manual_seed(46)
    proof = json_at(WINDOWS/'receipt.json')
    assert proof['code_sha256']==checksum(HERE/'views46.py')
    assert proof['roster_sha256']==checksum(WINDOWS/'rows.npz')
    name = f'{kind}_{cap}'
    packed = WINDOWS/(name+'.npz')
    assert proof['windows'][name]['sha256']==checksum(packed)
    with np.load(packed) as o:
        lengths,ends,token = o['lengths'],o['ends'],o['tokens']
    assert len(lengths)==proof['rows'] and ends[-1]==len(token)
    sequence = [token[a:b] for a,b in zip(ends[:-1],ends[1:],strict=True)]
    model,pad,weight = weights(kind)
    dest = DEST/name
    dest.mkdir(exist_ok=True,parents=True)
    partial = dest/'features.partial.npy'
    shape = (len(lengths),model.config.hidden_size*3)
    out = np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=shape)
    midpath = dest/'layer12.partial.npy'
    depth = kind=='qwen' and cap==96
    middle = np.lib.format.open_memmap(midpath,mode='w+',dtype=np.float16,shape=shape) if depth else None
    price = encode(sequence,model,pad,kind,cap,out,middle)
    out.flush()
    del out
    partial.replace(dest/'features.npy')
    arrays = {'features.npy':checksum(dest/'features.npy')}
    if middle is not None:
        middle.flush()
        del middle
        midpath.replace(dest/'layer12.npy')
        arrays['layer12.npy'] = checksum(dest/'layer12.npy')
    result = dict(name=name,window_sha256=checksum(packed),window_receipt_sha256=checksum(WINDOWS/'receipt.json'),
               model_sha256=weight,features=arrays,rows=len(lengths),cost=price,padded_formula='analytic',
               layers=model.config.num_hidden_layers,dimensions=shape[1],code_sha256=checksum(Path(__file__)),
               dtype=str(next(model.parameters()).dtype))
    write_json(dest/'receipt.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=TYPES)
    p.add_argument('cap',type=int,choices=CAPS)
    a=p.parse_args()
    main(a.kind,a.cap)
