"""Score a residual readout using the compact encoder's existing pooled states."""
import time

import numpy as np
import torch
from torch import nn

from predict_factor import (full_context, lexical_features, ordered_batches, batch_plan, batch, small)
from rule_model import pooling


def head(width):
    model = nn.Sequential(nn.Linear(2*width, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    return model


def joint(model, tokens, mask, side):
    hidden = model.encoder.token_states(tokens, mask)
    normalized = model.rule_head[0](pooling(hidden, mask))
    extra = model.rule_head[1](normalized)
    primary = model.encoder.readout(hidden, mask, side)
    return torch.cat((primary, extra), 1), normalized


def collect(model, tok, texts):
    original = torch.empty(len(texts), device='cuda', dtype=torch.float32)
    representations = torch.empty((len(texts), 2*model.width), device='cuda', dtype=torch.float16)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        def prepare(first):
            return full_context(texts[first:first+2048], tok)
        for first, seq in ordered_batches(range(0, len(texts), 2048), prepare, True):
            for group in batch_plan(list(range(len(seq))), seq, small.BATCH):
                ids, mask = batch(group, seq, tok.pad_token_id)
                v = model.rule_head[0](pooling(model.encoder.token_states(ids, mask), mask))
                sel = first+np.asarray(group)
                representations[sel] = v.to(dtype=representations.dtype)
                original[sel] = model.rule_head[1](v)[:, 20].float()
    return representations, original


@torch.inference_mode()
def infer_extra(model, tokenizer, code, heads):
    main = np.empty((len(code),2), dtype=np.float32)
    auxiliary = np.empty((len(code),22), dtype=np.float32)
    extras = np.empty((len(code),len(heads)), dtype=np.float32)
    fwd = 0.
    start = time.perf_counter()
    def prepare(first):
        segment = code[first:first+2048]
        return full_context(segment, tokenizer), lexical_features(segment)
    with torch.autocast('cuda',dtype=torch.float16):
        for first, (seq, lex) in ordered_batches(range(0,len(code),2048),prepare,True):
            main[first:first+len(seq),1] = lex[:,:4].any(axis=1)
            side = torch.as_tensor(lex,device='cuda')
            for rows in batch_plan(list(range(len(seq))),seq,small.BATCH):
                x,mask=batch(rows,seq,tokenizer.pad_token_id)
                at=time.perf_counter()
                logits, pool = joint(model, x, mask, side[rows])
                residuals = [(logits[:,22].float()+h(pool)[:,0].float()) for h in heads]
                scores = torch.column_stack((logits.float().sigmoid(), *residuals)).cpu().numpy()
                ix = first+np.asarray(rows)
                main[ix,0], auxiliary[ix], extras[ix] = scores[:, 0], scores[:, 2:24], scores[:, 24:]
                fwd += time.perf_counter()-at
    return main, auxiliary, extras, dict(forward_seconds=fwd,wall_seconds=time.perf_counter()-start)
