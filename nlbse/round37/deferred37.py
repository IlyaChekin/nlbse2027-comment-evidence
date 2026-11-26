"""Delay token-boundary work and section projections until exact routing is known."""
import numpy as np
import torch

from section37 import NATIVE, SECTIONS, OUTPUTS, prepare
from fast37 import TwoRegions
from eval37 import features


class ScopedStates:
    def __init__(self, recipe, route):
        self.project = TwoRegions(recipe,route,gated=False)

    def start(self, model, tok, texts):
        free,_ = torch.cuda.mem_get_info()
        self.limit = min(7*1024**3,int(free*.58))
        self.records = []
        self.bytes = 0
        self.ops = 0
        self.eager_count = 0
        self.late_count = 0
        self.eager = {}
        self.model,self.tok,self.texts = model,tok,texts

    @torch.inference_mode()
    def record(self, ids, hidden, mask, side, main, aux, pool):
        tensors = hidden,mask,side,main,aux,pool
        need = sum(x.numel()*x.element_size() for x in tensors)
        if self.bytes+need <= self.limit:
            self.records.append((ids,tensors))
            self.bytes += need
            return
        _,limits,_ = prepare([self.texts[i] for i in ids],self.tok)
        bound = torch.as_tensor(limits,device='cuda')
        local,ops = self.project(hidden,mask,side,self.model,bound,main,aux,pool)
        self.eager.update(zip(ids,local.float().cpu().numpy(),strict=True))
        self.eager_count += len(ids)
        self.ops += ops

    @torch.inference_mode()
    def features(self, data, threshold, mode, selection):
        # Keep an explicit input row index in the encoder output so the exact
        # scanner scope can recover the corresponding first-pass states.
        ids = selection[:,0].astype(np.int32)
        reach = np.full(len(self.texts),-1,np.int32)
        reach[ids] = np.arange(len(ids))
        part = np.full((len(ids),NATIVE+SECTIONS*(OUTPUTS+1)),np.nan,np.float32)
        _,limits,_ = prepare([self.texts[i] for i in ids],self.tok)
        bound = torch.as_tensor(limits,device='cuda')
        for row,j in zip(ids,range(len(ids)),strict=True):
            if row in self.eager:
                part[j] = self.eager[row]
        with torch.autocast('cuda',dtype=torch.float16):
            for rows,packed in self.records:
                index = reach[rows]
                need = np.flatnonzero(index>=0)
                if len(need):
                    h,m,g,p,a,q = [x[need] for x in packed]
                    at = index[need]
                    result,ops = self.project(h,m,g,self.model,bound[at],p,a,q)
                    part[at] = result.float().cpu().numpy()
                    self.ops += ops
                    self.late_count += len(need)
        design = features(data,threshold,mode,part)
        if not np.isfinite(design).all():
            raise ValueError('A chosen row has no retained or eager regional evidence')
        return design

    def finish(self, cost):
        extra = self.ops/1e9
        cost['analytic_gflops'] += extra
        cost['projection_gflops'] += extra
        cost.update(retained_bytes=self.bytes,eager_rows=self.eager_count,late_rows=self.late_count)
        self.records.clear()
        self.eager.clear()
        self.texts = None
