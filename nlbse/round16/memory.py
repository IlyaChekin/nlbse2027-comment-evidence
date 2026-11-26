"""Learn which tokens survive compression before the compact encoder's global attention."""

import torch
from torch import nn
from torch.nn import functional as F

import small


MEMORY_MODES=("control","weighted_memory","dual_memory")


class MemoryEncoder(small.Encoder):
    def __init__(self,vocab,width,mode):
        super().__init__(vocab,width)
        self.memory_mode=mode
        self.evidence=nn.Linear(width,1)
        nn.init.zeros_(self.evidence.weight)
        nn.init.zeros_(self.evidence.bias)

    def attention_memory(self,y,valid):
        b,t,d=y.shape
        pad=(-t)%4
        grouped=F.pad(y,(0,0,0,pad)).view(b,-1,4,d)
        weights=2*self.evidence(y).sigmoid().to(y.dtype)*valid.unsqueeze(-1)
        weights=F.pad(weights,(0,0,0,pad)).view(b,-1,4,1)
        count=F.pad(valid,(0,pad)).view(b,-1,4).sum(-1)
        selected=(grouped*weights).sum(2)/weights.sum(2).clamp_min(.001)
        if self.memory_mode=="weighted_memory":
            return selected,count
        uniform=grouped.masked_fill(~F.pad(valid.bool(),(0,pad)).view(b,-1,4,1),0.).sum(2)
        uniform=uniform/count.clamp_min(1).unsqueeze(-1)
        return torch.cat((uniform,selected),dim=1),torch.cat((count,count),dim=1)


def additional_operations(n,t,width,mode):
    base=n*t*(4*width+12)
    if mode=="dual_memory":
        groups=(t+3)//4
        base+=n*(4*groups*width**2+4*t*groups*width)
    return base/1e9
