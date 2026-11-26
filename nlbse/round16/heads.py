"""Low-cost token evidence readouts for the compact encoder."""

import torch
from torch import nn
from torch.nn import functional as F

import small


MODES=("control","attention","witness","segments")


class Evidence(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode=mode
        self.width=width
        self.norm=nn.LayerNorm(width)
        self.project=nn.Linear(width,8)
        dimensions={"attention":4,"witness":16,"segments":32}
        self.delta=nn.Sequential(nn.LayerNorm(dimensions[mode]),nn.Linear(dimensions[mode],12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self,x,valid):
        values=self.project(self.norm(x)).float()
        mask=valid.bool().unsqueeze(-1)
        length=valid.sum(1,keepdim=True).clamp_min(1)
        if self.mode=="attention":
            logits,values=values.chunk(2,dim=-1)
            weights=torch.softmax(logits.masked_fill(~mask,-torch.inf),dim=1)
            feature=(weights*values).sum(dim=1)
        elif self.mode=="witness":
            # Independent channels let the function readout combine local support and counter-evidence.
            masked=values.masked_fill(~mask,-torch.inf)
            k=min(6,x.shape[1])
            selected=masked.topk(k,dim=1,sorted=False).values
            mean=selected.masked_fill(~selected.isfinite(),0.).sum(1)/length.clamp_max(k)
            feature=torch.cat((mean,masked.amax(1)),dim=1)
        else:
            # Relative quarters retain prologue and tail evidence on short and long inputs.
            position=torch.arange(x.shape[1],device=x.device)
            segment=(4*position//length).clamp_max(3)
            membership=F.one_hot(segment,4).transpose(1,2)*valid.unsqueeze(1)
            pools=torch.bmm(membership.to(values.dtype),values)/membership.sum(-1,keepdim=True).clamp_min(1)
            feature=pools.flatten(1)
        return self.delta(feature).squeeze(1)


class LocalEncoder(small.Encoder):
    def __init__(self,vocab,width,mode):
        super().__init__(vocab,width)
        self.evidence=Evidence(width,mode)
        self.evidence_gain=1.

    def forward(self,tokens,valid,side):
        x=self.token_states(tokens,valid)
        original=self.readout(x,valid,side)
        if not self.evidence_gain:
            return original
        delta=self.evidence(x,valid).to(original.dtype)
        if self.evidence_gain!=1.:
            delta=delta*self.evidence_gain
        return original+torch.stack((delta,torch.zeros_like(delta)),dim=1)


def operations(n,t,width,mode):
    base=small.operations(n,t,width)
    if mode=="control":
        return base
    if mode in {"weighted_memory","dual_memory"}:
        from memory import additional_operations
        return base+additional_operations(n,t,width,mode)
    # Count projections, normalisation and reductions, including a conservative top-k comparison cost.
    aggregate={"attention":52,"witness":72,"segments":84}[mode]
    return base+n*(t*(16*width+7*width+aggregate)+1024)/1e9
