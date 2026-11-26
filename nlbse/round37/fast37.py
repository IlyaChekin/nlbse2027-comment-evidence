"""Compute only the two projections used by the frozen semantic candidate."""
import torch

from section37 import NATIVE, SECTIONS, OUTPUTS
from prefilter34 import GateFilter


class TwoRegions:
    def __init__(self, recipe, route, gated=True):
        self.gate = GateFilter(recipe,route) if gated else None
        self.gated = gated

    def __call__(self,hidden,mask,side,model,limits,main,aux,native_pool):
        batch_size,points,width = hidden.shape
        base = torch.cat((main[:,[0]],aux),1)
        profile = base.new_full((batch_size,SECTIONS*(OUTPUTS+1)),torch.nan)
        if self.gate is None:
            selected = torch.arange(batch_size,device=hidden.device)
        else:
            keep = self.gate(main.float().sigmoid(),aux.float().sigmoid())
            selected = keep.nonzero(as_tuple=True)[0]
        parts = len(selected)
        if not parts:
            return torch.cat((base,profile),1),0

        # A row contains one contiguous interval of real tokens. The regional
        # feature precedes/trails the original split and preserves the native max.
        start,stop,cut = limits[selected].T
        pos = torch.arange(points,device=hidden.device)
        left = (pos[None,:]>=start[:,None])&(pos[None,:]<cut[:,None])
        right = (pos[None,:]>=cut[:,None])&(pos[None,:]<stop[:,None])
        group = torch.stack((left,right),dim=1)
        mass = torch.stack((cut-start,stop-cut),dim=1)
        x = hidden[selected] if self.gated else hidden
        mean = group.to(x.dtype)@x/mass.clamp_min(1).unsqueeze(2)
        global_mean,global_max = native_pool[selected].split(width,dim=1)
        mean = torch.where(mass[:,:,None]>0,mean,global_mean[:,None,:]).to(global_max.dtype)
        joined = torch.cat((mean,global_max[:,None,:].expand_as(mean)),2)
        lexical = side[selected,None,:].expand(-1,2,-1)
        local_main = model.encoder.head(torch.cat((joined,lexical),2))[:,:,0]
        local_aux = model.rule_head(joined)[:,:,[0,20,21]]
        reference = torch.cat((main[selected,0,None],aux[selected][:,[0,20,21]]),1)
        evidence = torch.cat((local_main[:,:,None],local_aux),2)
        evidence = (evidence-reference[:,None,:]).masked_fill(mass[:,:,None]==0,0.)
        # The model reads only declaration/body columns. NaN in the unused
        # columns ensures accidental use of a different view fails loudly.
        profile[selected,:2*OUTPUTS] = evidence.flatten(1)
        profile[selected,SECTIONS*OUTPUTS:SECTIONS*OUTPUTS+2] = mass.float().log1p().to(profile.dtype)
        debit = parts*(2*4*width+4*points*width+2*(4*width*24+12*width))
        if self.gated:
            debit += batch_size*60
        return torch.cat((base,profile),1),debit
