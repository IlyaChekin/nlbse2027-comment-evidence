"""Extract frozen UniXcoder vectors on functions with a positive scanner outcome."""
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, RobertaModel

from context32 import HERE, f, checksum, write_json
from embed import MODEL, CAP, tokens
from students import ordered_batches
from train import cohorts

DEST = HERE/'dense'
BLOCK = 4096


@torch.inference_mode()
def encode(text, tok, model, output):
    def prepare(offset):
        return tokens(text[offset:offset+BLOCK],tok)
    with torch.autocast('cuda',dtype=torch.float16):
        for offset, seq in ordered_batches(range(0,len(text),BLOCK),prepare,True):
            for group in f.base.batch_plan(list(range(len(seq))),seq,96):
                ids, mask = f.base.batch(group,seq,tok.pad_token_id)
                states = model(input_ids=ids,attention_mask=mask).last_hidden_state
                mask = mask.bool().unsqueeze(-1)
                mean = states.masked_fill(~mask,0).sum(1)/mask.sum(1)
                largest = states.masked_fill(~mask,-torch.inf).amax(1)
                output[offset+np.asarray(group)] = torch.cat((mean,largest),1).half().cpu().numpy()


def encoder(path=MODEL):
    tok = AutoTokenizer.from_pretrained(path,local_files_only=True)
    cfg = AutoConfig.from_pretrained(path,local_files_only=True)
    cfg._attn_implementation = 'sdpa'
    model = RobertaModel.from_pretrained(path,config=cfg,local_files_only=True,
                                       add_pooling_layer=False).cuda().eval()
    return tok,model


def main():
    DEST.mkdir(exist_ok=True)
    weak, truth, cohort_sha = cohorts()
    corpus, y, _, _, _ = f.inputs()
    selected = np.flatnonzero(weak==1)
    assert np.array_equal(y[selected,0],truth[selected])
    tok,model = encoder()
    start = time.perf_counter()
    path = DEST/'features.npy'
    temporary = path.with_suffix('.part.npy')
    size = 2*model.config.hidden_size
    data = np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=(len(selected),size))
    encode([corpus[i] for i in selected],tok,model,data)
    data.flush()
    del data
    temporary.replace(path)
    rows = DEST/'rows.npz'
    np.savez_compressed(rows,ids=selected,y=truth[selected])
    receipt = dict(rows=len(selected),source_sha256=checksum(MODEL/'pytorch_model.bin'),
                   code_sha256=checksum(HERE/'collect_dense.py'),cohort_sha256=cohort_sha,
                   archive_sha256=checksum(path),rows_sha256=checksum(rows),cap=CAP,
                   dimensions=size,layers=model.config.num_hidden_layers,seconds=time.perf_counter()-start)
    write_json(DEST/'receipt.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
